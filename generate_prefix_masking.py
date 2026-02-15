import torch
from nnsight import LanguageModel, CONFIG
import os
from dotenv import load_dotenv

# Cleanest approach: Use nnsight's intervention API
class PrefixMaskIntervention:
    """
    Reusable intervention for prefix masking across any compatible model.
    """
    def __init__(self, model: LanguageModel, m: int, delimiter: str = "##"):
        self.model = model
        self.m = m
        self.delimiter = delimiter
        self.unmasked = None  # Determined automatically via calibration
        self.x = None  # Set dynamically based on input length

    def _calibrate(self, input_ids, calibration_tokens: int = 250):
        """Generate once to find where the second delimiter appears, and set unmasked_generations."""
        with self.model.generate(input_ids, do_sample=False, max_new_tokens=calibration_tokens, remote=False) as tracer:
            calibration_output = self.model.generator.output.save()

        prompt_len = input_ids.input_ids.shape[1]
        generated_token_ids = calibration_output[0, prompt_len:]

        # Walk through generated tokens, decoding each, to find the second delimiter
        count = 0
        for i, tok_id in enumerate(generated_token_ids):
            tok_text = self.model.tokenizer.decode(tok_id)
            if self.delimiter in tok_text:
                count += 1
                if count == 2:
                    # Stop right before the second delimiter
                    self.unmasked = i
                    print(f"Calibration: found second '{self.delimiter}' at generated token {i}, setting unmasked_generations={self.unmasked}")
                    return

        raise ValueError(f"Second '{self.delimiter}' not found in first {calibration_tokens} generated tokens. "
                         f"Try increasing calibration_tokens or checking the delimiter.")

    def __call__(self, input_ids: torch.Tensor, max_additional_tokens: int = 20):
        self._calibrate(input_ids)
        self.x = input_ids.input_ids.shape[1] + self.unmasked
        return self._generate(input_ids, max_additional_tokens)
    
    def _generate(self, input_ids, max_additional_tokens):

        # Unmasked steps: generate the plan (up to second delimiter)
        if self.unmasked > 0:
            with self.model.generate(input_ids, do_sample=False, max_new_tokens=self.unmasked, remote=False) as tracer:
                generated = self.model.generator.output.save()
        else:
            generated = input_ids.clone()

        print(self.model.tokenizer.decode(generated[0], skip_special_characters=False))
        generated_pp = self.model.tokenizer.decode(generated[0])
        # Masked steps: generate beyond the plan with prompt masked
        for step in range(max_additional_tokens):
            with self.model.trace(generated_pp, do_sample=False, remote=False) as tracer:
                for layer in self.model.model.layers:
                    attn_weights = layer.self_attn.source.attention_interface_0.source.torch_matmul_0.output
                    attn_weights[:, :, self.x:, :self.m] = float('-inf') #prompt might be 1 shorter w/out eos (m-1)
                    layer.self_attn.source.attention_interface_0.source.torch_matmul_0.output = attn_weights

                next_token = self.model.lm_head.output[0, -1].argmax(dim=-1).save()
            new_char = self.model.tokenizer.decode(next_token, skip_special_characters=False)
            generated_pp += new_char

        return generated_pp


# Example usage
if __name__ == "__main__":
    # Setup
    load_dotenv()
    model = LanguageModel('google/gemma-2-9b-it', dispatch=True, attn_implementation="eager")
    
    prompt = "What is the Eiffel Tower?"
    prompt =  "x = 4, y = 26, z = y / x + 5. First plan steps to solve for z in words, then use steps to find z."
    unmasked_generations = 78
    prompt =  "x = , y = 26, z = y / x + 5. First plan steps to solve for z in words, then use steps to find z."
    prompt =  "x = 4, y = 26, z = y / x + 5. First plan steps to solve for z in words, then use steps to find z."
    prompt = "I want to go to Bali, Indonesia and go swimming. First name the universal steps to plan any international trip (short list of short phrases), then use the steps to book the trip."

    input_ids = model.tokenizer(prompt, return_tensors="pt", )

    print(model.tokenizer.decode(input_ids.input_ids, skip_special_characters=False))
    
    m = input_ids.input_ids.shape[1]  # Mask entire prompt

    intervention = PrefixMaskIntervention(model, m)
    output = intervention(input_ids, max_additional_tokens=150)
    print(f"Generated: {output}")
