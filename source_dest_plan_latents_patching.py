import torch
import torch.nn.functional as F
from nnsight import LanguageModel, CONFIG
from dotenv import load_dotenv
import os
import numpy as np
from utils import visutils

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
    def __call__(self, original_prompt: str, original_info: str, max_new_tokens: int = 20):
        # Validate parameters
        if self.x > max_new_tokens:
            raise ValueError(
                f"x ({self.x}) cannot be greater than max_new_tokens ({max_new_tokens})"
            )
        add_on =  "value z = "
        # Tokenize prompts to get their lengths
        original_info_ids = self.model.tokenizer(
            original_info, add_special_tokens=True, return_tensors="pt"
        ).input_ids
        original_ids = self.model.tokenizer(
            original_prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids
        dest_ids = self.model.tokenizer(
            self.diff_prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids
        add_ids = self.model.tokenizer(
            add_on, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        original_info_len = original_info_ids.shape[1]
        original_prompt_len = original_ids.shape[1]
        dest_prompt_len = dest_ids.shape[1]
        add_prompt_len = add_ids.shape[1]
        end_plan_idx = original_prompt_len + self.x + add_prompt_len
        #end_plan_idx = original_prompt_len + self.x

        print(f"Starting generation: {max_new_tokens} tokens")
        print(f"Original prompt: {original_prompt}")
        print(f"Diff prompt: {self.diff_prompt}")
        print(f"Original prompt length: {original_prompt_len} tokens")
        print(f"Diff prompt length: {dest_prompt_len} tokens")
        print(f"Will replace residual at positions [{original_prompt_len}:{end_plan_idx}]")
        print()

        # Phase 1: Generate first x tokens normally using generate()
        print(f"=== Phase 1: Normal generation (first {self.x} tokens) ===")
        with self.model.generate(self.diff_prompt, max_new_tokens=self.x, remote=REMOTE) as tracer:
            output = self.model.generator.output.save()

        output_tokens = output
        generated_text = self.model.tokenizer.decode(output_tokens[0][original_prompt_len:], skip_special_tokens=False)
        print(f"  Generated: {generated_text}")

        # Phase 2: Extract residual activations ONE TIME from diff_prompt + first x tokens
        print(f"\n=== Phase 2: Extracting residual activations ===")
        print(f"Extracting from diff_prompt + first {self.x} generated tokens")

        # Build diff context with first x generated tokens
        diff_context = self.diff_prompt + generated_text + add_on
        #diff_context = self.diff_prompt + generated_text 
        print(f"Diff context: {diff_context}")
        print(f"Extracting at positions [{dest_prompt_len}:{end_plan_idx}]")

        residual_acts, phase2_logits = self._extract_residual_activations(diff_context, dest_prompt_len)
        print(f"Extracted activations from {len(residual_acts)} layers")
        print(f"Keys in residual_acts: {list(residual_acts.keys())}")
        if len(residual_acts) > 0:
            # Convert proxy objects to actual tensors for use in next trace (keep on CPU)
            residual_acts = {k: torch.tensor(v).cpu() if not isinstance(v, torch.Tensor) else v.clone().cpu() for k, v in residual_acts.items()}
            sample_shape = residual_acts[0].shape
            print(f"Activation shape per layer: {sample_shape}")
        else:
            raise RuntimeError("Failed to extract residual activations - dict is empty")

        # Visualize logits from Phase 2 (last layer residuals → norm → lm_head)
        print(f"\n=== Phase 2 Logits: Top-5 predictions per position ===")
        #self._plot_phase2_logits(phase2_logits, output_tokens[0], original_prompt_len)

        # Phase 3: Generate remaining tokens with residual patching (single remote call)
        if max_new_tokens > self.x:
            remaining_tokens = max_new_tokens - self.x - add_prompt_len
            #remaining_tokens = max_new_tokens - self.x 
            print(f"\n=== Phase 3: Patching generation ({remaining_tokens} tokens) ===")
            print(f"Patching at FIXED positions [{original_prompt_len}:{end_plan_idx}]")
            print()

            # Decode current tokens to use as prompt
            prompt_with_generated = original_prompt + generated_text + add_on
            #prompt_with_generated = original_prompt + generated_text
            num_layers = self.num_layers

            # Get the actual token length of the prompt we're sending
            prompt_tokens = self.model.tokenizer(prompt_with_generated, return_tensors="pt").input_ids
            actual_prompt_len = prompt_tokens.shape[1]
            print(f"    Actual prompt token length for Phase 3: {actual_prompt_len}")
            print(f"    Patching positions: {original_prompt_len}:{end_plan_idx}")

            with model.edit(inplace=True):
                for layer_idx in range(num_layers):
                    for i in range(self.x + add_prompt_len - 1):
                        self.model.model.layers[layer_idx].output[0][:, i+original_prompt_len, :] = residual_acts[layer_idx][0, i]
                
            with self.model.generate(prompt_with_generated, max_new_tokens=remaining_tokens, remote=REMOTE) as tracer:
                # Save logits from each generation step
                logits_list = list().save()
                with tracer.iter[0]:
                    # Get logits directly from lm_head output
                    step_logits = self.model.output.logits[:, -1, :].save()
                    logits_list.append(step_logits)
                
                output = self.model.generator.output.save()

            output_tokens = output
            patched_text = self.model.tokenizer.decode(output_tokens[0][end_plan_idx:], skip_special_tokens=False)
            print(f"  Generated with patching: {patched_text}")

            # Print top 5 logit probs for each generated token
            print(f"\n=== Phase 3 Logit Probs: Top 5 tokens per position ===")
            visutils.print_top_logit_probs_from_logits(logits_list, output_tokens[0], actual_prompt_len)

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
            # Also save logits (norm + lm_head applied to last layer residuals)
            logits = self.model.output.logits[:, extract_start:, :].save()

        print(f"  After trace: {len(residual_list)} entries")
        print(f"  First entry shape: {residual_list[0].shape}")

        # Convert to dict
        residual_acts = {i: residual_list[i] for i in range(len(residual_list))}
        return residual_acts, logits


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
    original_prompt =  "x = 4, y = 26, z = y / x + 5. First plan steps to solve for z in words, then use steps to find z."
    original_info =  "x = 4, y = 26"
    diff_prompt = "x = 7, y = 13, z = y / x + 5. First plan steps to solve for z in words, then use steps to find z."
    # Parameters
    x = 52  # Replace first 5 generated tokens' residual activations
    max_tokens = 200  # Generate 15 total tokens

    print(f"Original prompt: {original_prompt}")
    print(f"Diff prompt: {diff_prompt}")
    print(f"Will generate {x} tokens normally, extract from diff context,")
    print(f"then generate {max_tokens - x} more tokens with residual patching")
    print("=" * 70)
    print()

    # Run intervention
    intervention = ResidualStreamReplacementIntervention(model, x, diff_prompt)
    output = intervention(original_prompt, original_info, max_new_tokens=max_tokens)

    print()
    print("=" * 70)
    print(f"Final output: {model.tokenizer.decode(output[0])}")
