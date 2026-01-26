import torch
from nnsight import LanguageModel, CONFIG
from dotenv import load_dotenv
import os


class CounterfactualPrefixIntervention:
    """
    Counterfactual attention intervention: for generation steps after `unmasked`,
    replace V projection outputs at positions 0:m with values from an alternate prefix.

    This allows you to see what the model would predict if the final token's attention
    used a different prefix's Value representations, while all other computations
    use the original prompt.

    Args:
        model: nnsight LanguageModel instance
        m: Number of prefix token positions to replace (positions 0:m)
        unmasked_generations: Number of tokens to generate normally before
            the V-splice intervention begins.
    """

    def __init__(self, model: LanguageModel, m: int, unmasked_generations: int):
        self.model = model
        self.m = m  # Number of prefix TOKEN POSITIONS to replace (0:m)
        self.unmasked = unmasked_generations

    def __call__(self, original_ids: torch.Tensor, alt_prefix_ids: torch.Tensor,
                 max_new_tokens: int = 20):
        """
        Generate tokens with V-splice intervention.

        Args:
            original_ids: Original input prompt IDs [batch, seq_len]
            alt_prefix_ids: Alternate prefix token IDs [batch, seq_len] - must match original length
            max_new_tokens: Number of tokens to generate

        Returns:
            Generated output tensor including original prompt
        # """
        # assert original_ids.shape[1] == alt_prefix_ids.shape[1], \
        #     f"Original and alt prefix must have same length, got {original_ids.shape[1]} vs {alt_prefix_ids.shape[1]}"
        assert original_ids.shape[1] >= self.m, \
            f"Input must be at least {self.m} tokens, got {original_ids.shape[1]}"

        return self._generate_with_counterfactual(original_ids, alt_prefix_ids, max_new_tokens)

    def _extract_alt_v_for_step(self, alt_prompt: str) -> dict:
        """
        Run alt prefix + generation_so_far, extract v_proj.output[:, :m, :] at each layer.

        Args:
            alt_prompt: The alternate prefix prompt string (with any generated tokens appended)

        Returns:
            dict mapping layer_idx -> v_tensor of shape [batch, m, hidden_dim]
        """
        alt_v = {}
        with self.model.trace(alt_prompt, remote=True):
            for layer_idx, layer in enumerate(self.model.model.layers):
                # Get V projection for first m token positions
                v_out = layer.self_attn.v_proj.output[:, :self.m, :].save()
                alt_v[layer_idx] = v_out
        return alt_v

    def _generate_one_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Normal single token generation without intervention.

        Args:
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            Next token ID tensor
        """
        prompt = self.model.tokenizer.decode(input_ids[0])

        with self.model.trace(prompt, remote=True):
            logits = self.model.lm_head.output[:, -1, :].save()

        next_token = logits.argmax(dim=-1)
        return next_token
    
    def _generate_x_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Normal single token generation without intervention.

        Args:
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            Next token ID tensor
        """
        prompt = self.model.tokenizer.decode(input_ids[0])

        with self.model.generate(prompt, max_new_tokens=self.unmasked, remote=True, do_sample=False) as gen:
            saved = self.model.generator.output.save()

        return saved[0]

    def _generate_one_token_with_v_splice(self, input_ids: torch.Tensor,
                                           alt_v: dict) -> torch.Tensor:
        """
        Generate one token, replacing V at positions 0:m with alt values.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            alt_v: dict mapping layer_idx -> V tensor for positions 0:m

        Returns:
            Next token ID tensor
        """
        prompt = self.model.tokenizer.decode(input_ids[0])

        with self.model.trace(prompt, remote=True):
            for layer_idx, layer in enumerate(self.model.model.layers):
                # Splice: replace V vectors at first m token positions
                layer.self_attn.v_proj.output[:, :self.m, :] = alt_v[layer_idx]

            logits = self.model.lm_head.output[:, -1, :].save()

        next_token = logits.argmax(dim=-1)
        return next_token

    def _generate_with_counterfactual(self, original_ids: torch.Tensor,
                                       alt_prefix_ids: torch.Tensor,
                                       max_new_tokens: int) -> torch.Tensor:
        """
        Main generation loop with dual-run V-splicing.

        For steps < unmasked: normal generation with original prompt
        For steps >= unmasked: extract alt V values, then generate with V-splice

        Args:
            original_ids: Original input token IDs
            alt_prefix_ids: Alternate prefix token IDs
            max_new_tokens: Number of tokens to generate

        Returns:
            Full sequence including original prompt and generated tokens
        """
        generated_tokens = []
        current_original = original_ids.clone()
        current_alt = alt_prefix_ids.clone()

        print(f"Starting generation: {max_new_tokens} tokens")
        print(f"Original prompt: {self.model.tokenizer.decode(original_ids[0])}")
        print(f"Alt prefix: {self.model.tokenizer.decode(alt_prefix_ids[0])}")
        print(f"Unmasked steps: {self.unmasked}, m (positions to splice): {self.m}")

        for step in range(max_new_tokens):
            if step < self.unmasked:
                # Normal generation - just run original
                print(f"Step {step}: Normal generation")
                next_token = self._generate_one_token(current_original)
            else:
                # Dual-run with V-splice
                print(f"Step {step}: V-splice generation")

                # 1. Get alt V values for positions 0:m
                alt_prompt = self.model.tokenizer.decode(current_alt[0])
                alt_v = self._extract_alt_v_for_step(alt_prompt)

                # 2. Generate with original, splicing in alt V values at positions 0:m
                next_token = self._generate_one_token_with_v_splice(current_original, alt_v)

            # Handle tensor dimensions for concatenation
            if next_token.dim() == 0:
                next_token = next_token.unsqueeze(0)
            if next_token.dim() == 1:
                next_token = next_token.unsqueeze(0)

            generated_tokens.append(next_token)
            print(f"  Generated: {self.model.tokenizer.decode(next_token[0])}")

            # Append token to both sequences for next iteration
            current_original = torch.cat([current_original, next_token], dim=1)
            current_alt = torch.cat([current_alt, next_token], dim=1)

        return current_original


# Example usage
if __name__ == "__main__":
    # Setup
    load_dotenv()
    CONFIG.set_default_api_key(os.environ["HF_TOKEN"])
    CONFIG.API.HOST = "https://api.ndif.us"
    CONFIG.save()
    model = LanguageModel('google/gemma-2-9b-it')

    # Original prompt and alternate prefix
    original_prompt = "x = 9, y = 12, z = (x + y) * (y / x). Write the steps for solving for z:"
    alt_prefix = "x = 4, y = 3, z = (x + y) * (y / x). Write the steps for solving for z:"

    original_ids = model.tokenizer(original_prompt, add_special_tokens=True, return_tensors="pt").input_ids
    alt_ids = model.tokenizer(alt_prefix, add_special_tokens=True, return_tensors="pt").input_ids

    print(f"Original tokens: {original_ids.shape[1]}")
    print(f"Alt tokens: {alt_ids.shape[1]}")

    # Ensure same length (pad if needed or use prompts of same token count)
    # assert original_ids.shape[1] == alt_ids.shape[1], \
    #     "Prompts must have same number of tokens for this example"

    m = 2  # Replace V at first 2 token positions (BOS + first word token)
    unmasked_generations = 3  # Generate first 3 tokens normally
    max_tokens = 10  # Total tokens to generate

    print(f"\nOriginal prompt: {original_prompt}")
    print(f"Alternate prefix: {alt_prefix}")
    print(f"First {unmasked_generations} tokens generated normally")
    print(f"V-splice at positions 0:{m} for remaining tokens")

    intervention = CounterfactualPrefixIntervention(model, m, unmasked_generations)
    output = intervention(original_ids, alt_ids, max_new_tokens=max_tokens)

    print(f"\nFinal output: {model.tokenizer.decode(output[0])}")
