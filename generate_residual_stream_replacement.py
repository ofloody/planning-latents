import torch
import torch.nn.functional as F
from nnsight import LanguageModel, CONFIG
from dotenv import load_dotenv
import os
import numpy as np

REMOTE = True
DEVICE = "cpu"  # Force CPU for local operations

# Patches residual stream activations from diff_prompt into generated tokens.
class ResidualStreamReplacementIntervention:

    def __init__(self, model: LanguageModel, x: int, diff_prompt: str):
        self.model = model
        self.x = x  # Number of generated token positions to replace
        self.diff_prompt = diff_prompt
        self.num_layers = model.config.num_hidden_layers

    # Run the intervention and return generated tokens.
    def __call__(self, original_prompt: str, max_new_tokens: int = 20):
        # Validate parameters
        if self.x > max_new_tokens:
            raise ValueError(
                f"x ({self.x}) cannot be greater than max_new_tokens ({max_new_tokens})"
            )
        nasty_add_on =  "value z = "
        # Tokenize prompts to get their lengths
        original_ids = self.model.tokenizer(
            original_prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids
        diff_ids = self.model.tokenizer(
            self.diff_prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids
        add_ids = self.model.tokenizer(
            nasty_add_on, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        original_prompt_len = original_ids.shape[1]
        diff_prompt_len = diff_ids.shape[1]
        add_prompt_len = add_ids.shape[1]
        end_plan_idx = original_prompt_len + self.x + add_prompt_len

        print(f"Starting generation: {max_new_tokens} tokens")
        print(f"Original prompt: {original_prompt}")
        print(f"Diff prompt: {self.diff_prompt}")
        print(f"Original prompt length: {original_prompt_len} tokens")
        print(f"Diff prompt length: {diff_prompt_len} tokens")
        print(f"Will replace residual at positions [{original_prompt_len}:{end_plan_idx}]")
        print()

        # Phase 1: Generate first x tokens normally using generate()
        print(f"=== Phase 1: Normal generation (first {self.x} tokens) ===")
        with self.model.generate(original_prompt, max_new_tokens=self.x, remote=REMOTE) as tracer:
            output = self.model.generator.output.save()

        output_tokens = output
        generated_text = self.model.tokenizer.decode(output_tokens[0][original_prompt_len:], skip_special_tokens=False)
        print(f"  Generated: {generated_text}")

        # Phase 2: Extract residual activations ONE TIME from diff_prompt + first x tokens
        print(f"\n=== Phase 2: Extracting residual activations ===")
        print(f"Extracting from diff_prompt + first {self.x} generated tokens")

        # Build diff context with first x generated tokens
        diff_context = self.diff_prompt + generated_text + nasty_add_on
        print(f"Diff context: {diff_context}")
        print(f"Extracting at positions [{diff_prompt_len}:{end_plan_idx}]")

        residual_acts = self._extract_residual_activations(diff_context, diff_prompt_len)
        print(f"Extracted activations from {len(residual_acts)} layers")
        print(f"Keys in residual_acts: {list(residual_acts.keys())}")
        if len(residual_acts) > 0:
            # Convert proxy objects to actual tensors for use in next trace (keep on CPU)
            residual_acts = {k: torch.tensor(v).cpu() if not isinstance(v, torch.Tensor) else v.clone().cpu() for k, v in residual_acts.items()}
            sample_shape = residual_acts[0].shape
            print(f"Activation shape per layer: {sample_shape}")
        else:
            raise RuntimeError("Failed to extract residual activations - dict is empty")

        # Phase 3: Generate remaining tokens with residual patching (single remote call)
        if max_new_tokens > self.x:
            remaining_tokens = max_new_tokens - self.x
            print(f"\n=== Phase 3: Patching generation ({remaining_tokens} tokens) ===")
            print(f"Patching at FIXED positions [{original_prompt_len}:{end_plan_idx}]")
            print()

            # Decode current tokens to use as prompt
            prompt_with_generated = original_prompt + generated_text + nasty_add_on
            num_layers = self.num_layers

            # Get the actual token length of the prompt we're sending
            prompt_tokens = self.model.tokenizer(prompt_with_generated, return_tensors="pt").input_ids
            actual_prompt_len = prompt_tokens.shape[1]
            print(f"    Actual prompt token length for Phase 3: {actual_prompt_len}")
            print(f"    Patching positions: {original_prompt_len}:{end_plan_idx}")

            with model.edit(inplace=True):
                for layer_idx in range(num_layers):
                    for i in range(self.x + add_prompt_len):
                        self.model.model.layers[layer_idx].output[0][:, i+original_prompt_len, :] = residual_acts[layer_idx][0, i]
                
            with self.model.generate(prompt_with_generated, max_new_tokens=2, remote=REMOTE) as tracer:
                output = self.model.generator.output.save()
                # Save logits from each generation step
                logits_list = list().save()
                with tracer.all():
                    # Get logits directly from lm_head output
                    step_logits = self.model.lm_head.output[:, -1, :].save()
                    logits_list.append(step_logits)

            output_tokens = output
            patched_text = self.model.tokenizer.decode(output_tokens[0][end_plan_idx-5:], skip_special_tokens=False)
            print(f"  Generated with patching: {patched_text}")

            # Print top 5 logit probs for each generated token
            print(f"\n=== Phase 3 Logit Probs: Top 5 tokens per position ===")
            self._print_top_logit_probs_from_logits(logits_list, output_tokens[0], actual_prompt_len)

            model.clear_edits()

        print(f"\n=== Phase 4: Run Both Prompts with Unedited plan latents ===")
        with self.model.generate(original_prompt, max_new_tokens=max_new_tokens, remote=REMOTE) as tracer:
            output = self.model.generator.output.save()
        
        current_tokens = output
        patched_text = self.model.tokenizer.decode(current_tokens[0], skip_special_tokens=False)
        print(f"  Generated original: {patched_text}")

        with self.model.generate(diff_prompt, max_new_tokens=max_new_tokens, remote=REMOTE) as tracer:
            output = self.model.generator.output.save()
        
        current_tokens = output
        patched_text = self.model.tokenizer.decode(current_tokens[0], skip_special_tokens=False)
        print(f"  Generated diff: {patched_text}")


        return output_tokens

    # Print and plot top 5 token probabilities per generated position.
    def _print_top_logit_probs_from_logits(self, logits_list, output_tokens, prompt_len: int):
        import matplotlib.pyplot as plt
        # Collect data for visualization
        all_positions_data = []

        for step_idx, logits in enumerate(logits_list):
            # Convert to tensor if needed
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits)

            # Get probabilities via softmax
            probs = F.softmax(logits.float(), dim=-1)

            # Get top 5 tokens and their probabilities
            top_probs, top_indices = torch.topk(probs, k=5, dim=-1)

            # Get the actual generated token at this position
            gen_token_idx = prompt_len + step_idx
            if gen_token_idx < len(output_tokens):
                actual_token_id = output_tokens[gen_token_idx].item()
                actual_token = self.model.tokenizer.decode([actual_token_id])
            else:
                actual_token = "[N/A]"
                actual_token_id = -1

            print(f"\n  Position {step_idx} (token #{gen_token_idx}): actual='{actual_token}'")

            position_data = {
                'step': step_idx,
                'token_idx': gen_token_idx,
                'actual_token': actual_token,
                'actual_token_id': actual_token_id,
                'top_tokens': [],
                'top_probs': [],
                'top_ids': []
            }

            for rank in range(5):
                token_id = top_indices[0, rank].item()
                prob = top_probs[0, rank].item()
                token_str = self.model.tokenizer.decode([token_id])
                marker = " <--" if token_id == actual_token_id else ""
                print(f"    {rank+1}. '{token_str}' (id={token_id}): {prob:.4f}{marker}")

                position_data['top_tokens'].append(token_str)
                position_data['top_probs'].append(prob)
                position_data['top_ids'].append(token_id)

            all_positions_data.append(position_data)

        # Create Plotly visualization
        self._create_logit_probs_plot(all_positions_data)

    # Plot top 5 token probabilities as bar charts.
    def _create_logit_probs_plot(self, positions_data: list):
        if len(positions_data) == 0:
            print("No data to visualize")
            return

        num_positions = len(positions_data)
        fig, axes = plt.subplots(1, num_positions, figsize=(4 * num_positions, 5))

        # Handle single subplot case
        if num_positions == 1:
            axes = [axes]

        for ax, pos_data in zip(axes, positions_data):
            tokens = []
            probs = []
            colors = []

            for i, (token, prob, token_id) in enumerate(zip(
                pos_data['top_tokens'], pos_data['top_probs'], pos_data['top_ids']
            )):
                # Clean token for display
                display_token = repr(token)[1:-1] if token.strip() == '' else token
                tokens.append(display_token)
                probs.append(prob)

                # Green for actual token, blue for others
                if token_id == pos_data['actual_token_id']:
                    colors.append('#27ae60')
                else:
                    colors.append('#3498db')

            x = np.arange(len(tokens))
            bars = ax.bar(x, probs, color=colors)

            # Add probability labels on bars
            for bar, prob in zip(bars, probs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('Probability')
            ax.set_title(f"Step {pos_data['step']}: actual='{pos_data['actual_token']}'")

        plt.suptitle('Phase 3 Logit Probabilities: Top 5 Tokens per Generated Position', fontsize=12)
        plt.tight_layout()

        # Save and show
        output_path = "logit_probs_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  Visualization saved to: {output_path}")
        plt.show()

    # Extract residual activations from all layers starting at extract_start.
    def _extract_residual_activations(self, prompt: str, extract_start: int) -> dict:
        num_layers = self.num_layers

        # Extract from all layers using nnsight's list().save() pattern
        print(f"  Extracting all {num_layers} layers...")

        with self.model.trace(prompt, remote=REMOTE):
            # Use list().save() to create a tracked list
            residual_list = list().save()
            for layer_idx in range(num_layers):
                residual_list.append(
                    self.model.model.layers[layer_idx].output[0][:, extract_start:, :]
                )

        print(f"  After trace: {len(residual_list)} entries")
        print(f"  First entry shape: {residual_list[0].shape}")

        # Convert to dict
        residual_acts = {i: residual_list[i] for i in range(len(residual_list))}
        return residual_acts


# Example usage
if __name__ == "__main__":
    # Setup
    load_dotenv()
    CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
    CONFIG.API.HOST = "https://api.ndif.us"
    CONFIG.save()
    model = LanguageModel('google/gemma-2-9b-it')
    print(model)

    # Define prompts - different contexts that should lead to different reasoning
    original_prompt =  "x = 9, y = 12, z = y / x + 2. First plan steps to solve for z in words, then use steps to find z."
    diff_prompt = "x = 4, y = 16, z = y / x + 2. First plan steps to solve for z in words, then use steps to find z."

    # Parameters
    x = 68  # Replace first 5 generated tokens' residual activations
    max_tokens = 200  # Generate 15 total tokens

    print(f"Original prompt: {original_prompt}")
    print(f"Diff prompt: {diff_prompt}")
    print(f"Will generate {x} tokens normally, extract from diff context,")
    print(f"then generate {max_tokens - x} more tokens with residual patching")
    print("=" * 70)
    print()

    # Run intervention
    intervention = ResidualStreamReplacementIntervention(model, x, diff_prompt)
    output = intervention(original_prompt, max_new_tokens=max_tokens)

    print()
    print("=" * 70)
    print(f"Final output: {model.tokenizer.decode(output[0])}")
