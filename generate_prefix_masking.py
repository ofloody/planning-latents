import torch
from nnsight import LanguageModel, CONFIG
import os
from dotenv import load_dotenv


# Cleanest approach: Use nnsight's intervention API
class PrefixMaskIntervention:
    """
    Reusable intervention for prefix masking across any compatible model.
    """
    def __init__(self, model: LanguageModel, m: int, unmasked_generations: int):
        self.model = model
        self.m = m
        self.unmasked = unmasked_generations
        self.x = None  # Set dynamically based on input length
    
    def __call__(self, input_ids: torch.Tensor, max_new_tokens: int = 20):
        self.x = input_ids.shape[1] + self.unmasked
        return self._generate(input_ids, max_new_tokens)
    
    def _generate(self, input_ids, max_new_tokens):
        prompt = self.model.tokenizer.decode(input_ids[0])

        with self.model.generate(prompt, max_new_tokens=max_new_tokens, remote=True) as tracer:
            # First self.unmasked iterations: no masking
            with tracer.iter[:self.unmasked]:
                pass

            # Remaining iterations: apply mask
            with tracer.iter[self.unmasked:]:
                for layer in self.model.model.layers:
                    self._apply_mask(layer.self_attn, input_ids.shape[1] + max_new_tokens)

            output = self.model.generator.output.save()
            print(output)
    

        return output
    
    def _apply_mask(self, attn_module, seq_len):
        """Apply prefix mask to attention scores."""
        n = seq_len - 1
        
        # Create and apply mask
        # Access attention weights - exact path depends on model
        try:
            weights = attn_module.o_proj.input[0][0]
            mask = torch.zeros((1, 1, seq_len, seq_len))
            if self.m > 0 and self.x <= n:
                mask[:, :, self.x:n+1, :self.m] = float('-inf')
            weights[:] = weights + mask
        except Exception:
            pass  # Handle models with different architecture


# Example usage
if __name__ == "__main__":
    # Setup
    load_dotenv()
    CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
    CONFIG.API.HOST = "https://api.ndif.us"
    CONFIG.save()
    model = LanguageModel('meta-llama/Llama-3.1-8B')
    
    prompt = "What is the Eiffle Tower?"
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids
    
    m = 2  # Mask first 2 tokens
    unmasked_generations=5 # Allow next 5 tokens to attend to all prior tokens
    
    intervention = PrefixMaskIntervention(model, m, unmasked_generations)
    output = intervention(input_ids, max_new_tokens=8)
    print(f"Generated: {model.tokenizer.decode(output[0])}")
