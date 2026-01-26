import torch
from nnsight import LanguageModel, CONFIG
from dotenv import load_dotenv
import os

REMOTE = True
DEVICE = "cpu"  # Force CPU for local operations

class ResidualStreamReplacementIntervention:
    """
    Residual stream replacement intervention: after generating x tokens normally,
    replace residual stream activations at positions [prompt_len:prompt_len+x]
    with values extracted from diff_prompt + first_x_generated_tokens.

    This allows you to see what happens when the model's internal representations
    for the first x generated tokens come from a different context, while maintaining
    the original prompt context and generating subsequent tokens based on these
    replaced representations.

    Args:
        model: nnsight LanguageModel instance
        x: Number of generated token positions to replace (first x after prompt)
        diff_prompt: Alternate prompt to extract activations from
    """

    def __init__(self, model: LanguageModel, x: int, diff_prompt: str):
        self.model = model
        self.x = x  # Number of generated token positions to replace
        self.diff_prompt = diff_prompt
        self.num_layers = model.config.num_hidden_layers

    def __call__(self, original_prompt: str, max_new_tokens: int = 20):
        """
        Generate tokens with residual stream replacement intervention.

        Args:
            original_prompt: Original input prompt string
            max_new_tokens: Total number of tokens to generate

        Returns:
            Generated output tensor including original prompt
        """
        # Validate parameters
        if self.x > max_new_tokens:
            raise ValueError(
                f"x ({self.x}) cannot be greater than max_new_tokens ({max_new_tokens})"
            )

        # Tokenize prompts to get their lengths
        original_ids = self.model.tokenizer(
            original_prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids
        diff_ids = self.model.tokenizer(
            self.diff_prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids

        original_prompt_len = original_ids.shape[1]
        diff_prompt_len = diff_ids.shape[1]

        print(f"Starting generation: {max_new_tokens} tokens")
        print(f"Original prompt: {original_prompt}")
        print(f"Diff prompt: {self.diff_prompt}")
        print(f"Original prompt length: {original_prompt_len} tokens")
        print(f"Diff prompt length: {diff_prompt_len} tokens")
        print(f"Will replace residual at positions [{original_prompt_len}:{original_prompt_len + self.x}]")
        print()

        # Phase 1: Generate first x tokens normally using generate()
        print(f"=== Phase 1: Normal generation (first {self.x} tokens) ===")
        with self.model.generate(original_prompt, max_new_tokens=self.x, remote=REMOTE) as tracer:
            output = self.model.generator.output.save()

        current_tokens = output
        generated_text = self.model.tokenizer.decode(current_tokens[0][original_prompt_len:], skip_special_tokens=False)
        print(f"  Generated: {generated_text}")

        # Phase 2: Extract residual activations ONE TIME from diff_prompt + first x tokens
        print(f"\n=== Phase 2: Extracting residual activations ===")
        print(f"Extracting from diff_prompt + first {self.x} generated tokens")

        # Build diff context with first x generated tokens
        diff_context = self.diff_prompt + generated_text
        print(f"Diff context: {diff_context}")
        print(f"Extracting at positions [{diff_prompt_len}:{diff_prompt_len + self.x}]")

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
            print(f"Patching at FIXED positions [{original_prompt_len}:{original_prompt_len + self.x}]")
            print()

            # Decode current tokens to use as prompt
            prompt_with_generated = self.model.tokenizer.decode(current_tokens[0])
            num_layers = self.num_layers

            # Get the actual token length of the prompt we're sending
            prompt_tokens = self.model.tokenizer(prompt_with_generated, return_tensors="pt").input_ids
            actual_prompt_len = prompt_tokens.shape[1]
            print(f"    Actual prompt token length for Phase 3: {actual_prompt_len}")
            print(f"    Patching positions: {original_prompt_len}:{original_prompt_len + self.x}")

            with self.model.generate(prompt_with_generated, max_new_tokens=remaining_tokens, remote=REMOTE) as tracer:
                # Patch only on step 0 (the prefill pass where full sequence is processed)
                with tracer.iter[0]:
                    for layer_idx in range(num_layers):
                        self.model.model.layers[layer_idx].output[0][:, original_prompt_len:original_prompt_len + self.x, :] = residual_acts[layer_idx]

                output = self.model.generator.output.save()

            current_tokens = output
            patched_text = self.model.tokenizer.decode(current_tokens[0][original_prompt_len + self.x:], skip_special_tokens=False)
            print(f"  Generated with patching: {patched_text}")

        return current_tokens

    def _extract_residual_activations(self, prompt: str, extract_start: int) -> dict:
        """
        Extract residual stream activations for positions [extract_start:extract_start+x]
        from all layers.

        Args:
            prompt: Full prompt string to run through the model
            extract_start: Starting position index for extraction

        Returns:
            dict mapping layer_idx -> residual_tensor of shape [batch, x, hidden_dim]
        """
        num_layers = self.num_layers

        # Extract from all layers using nnsight's list().save() pattern
        print(f"  Extracting all {num_layers} layers...")

        with self.model.trace(prompt, remote=REMOTE):
            # Use list().save() to create a tracked list
            residual_list = list().save()
            for layer_idx in range(num_layers):
                residual_list.append(
                    self.model.model.layers[layer_idx].output[0][:, extract_start:extract_start + self.x, :]
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
    original_prompt =  "x = 9, y = 12, z = y / x + 2. First plan steps to solve for z, then use steps to find z."
    diff_prompt = "x = 4, y = 3, z = y / x + 2. First plan steps to solve for z, then use steps to find z."

    # Parameters
    x = 40  # Replace first 5 generated tokens' residual activations
    max_tokens = 100  # Generate 15 total tokens

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
