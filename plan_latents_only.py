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

    def __init__(self, model: LanguageModel, x: int):
        self.model = model
        self.x = x  # Number of generated token positions to replace
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
        add_ids = self.model.tokenizer(
            add_on, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        original_info_len = original_info_ids.shape[1]
        original_prompt_len = original_ids.shape[1]
        add_prompt_len = add_ids.shape[1]
        #end_plan_idx = self.x + add_prompt_len
        end_plan_idx = self.x

        # Constructing padding of length size(prompt_ids)
        space_ids = self.model.tokenizer(" ", add_special_tokens=False, return_tensors="pt").input_ids
        print(space_ids)  # tensor([[...]])
        print(self.model.tokenizer.decode(space_ids[0]))  # verify it's a space
        token_id = space_ids[0, 0].item()
        pad_prompt = space_ids.repeat(1, original_prompt_len)
        print(f"Token ID: {token_id}")
        print(f"Token repr: {repr(self.model.tokenizer.decode([token_id]))}")
        end_plan_idx = self.x + add_prompt_len

        print(f"Starting generation: {max_new_tokens} tokens")
        print(f"Original prompt: {original_prompt}")
        print(f"Original prompt length: {original_prompt_len} tokens")
        print(f"Will replace residual at positions [{original_prompt_len}:{end_plan_idx}]")
        print()

        # Phase 1: Generate first x tokens AND extract residual activations during generation
        print(f"=== Phase 1: Generate {self.x} tokens and extract residual activations ===")
        num_layers = self.num_layers

        with self.model.generate(original_prompt, max_new_tokens=self.x, remote=REMOTE) as tracer:
            # Initialize storage for activations at each generation step
            all_step_residuals = [list().save() for _ in range(self.x)]

            for step_idx in tracer.iter:
                # At each generation step, save activations from all layers for the current (last) token
                for layer_idx in range(num_layers):
                    all_step_residuals[step_idx].append(
                        self.model.model.layers[layer_idx].output[0][:, -1, :].clone()
                    )

            output = self.model.generator.output.save()

        plan_tokens = output[0][original_prompt_len:]
        generated_text = self.model.tokenizer.decode(plan_tokens, skip_special_tokens=False)
        print(f"  Generated: {generated_text}")

        # Reorganize activations: from [step][layer] to {layer: tensor[1, x, hidden_dim]}
        residual_acts = {}
        for layer_idx in range(num_layers):
            layer_acts = [all_step_residuals[step][layer_idx] for step in range(self.x)]
            # Stack along position dimension and convert to tensor
            stacked = torch.stack([
                torch.tensor(a).cpu() if not isinstance(a, torch.Tensor) else a.clone().cpu()
                for a in layer_acts
            ], dim=1)
            residual_acts[layer_idx] = stacked

        print(f"Extracted activations from {len(residual_acts)} layers")
        print(f"Keys in residual_acts: {list(residual_acts.keys())}")
        if len(residual_acts) > 0:
            sample_shape = residual_acts[0].shape
            print(f"Activation shape per layer: {sample_shape}")
        else:
            raise RuntimeError("Failed to extract residual activations - dict is empty")

        # Phase 2: Generate remaining tokens with residual patching (single remote call)
        if max_new_tokens > self.x:
            #remaining_tokens = max_new_tokens - self.x - add_prompt_len
            remaining_tokens = max_new_tokens - self.x
            print(f"\n=== Phase 2: Patching generation ({remaining_tokens} tokens) ===")
            print(f"Patching at FIXED positions [{original_prompt_len}:{end_plan_idx}]")
            print()

            # Decode current tokens to use as prompt
            #generated_plan_added = generated_text + add_on
            generated_plan_added = torch.cat([pad_prompt, torch.unsqueeze(plan_tokens, dim=0)], dim=1)
            num_layers = self.num_layers

            # Get the actual token length of the prompt we're sending
            prompt_tokens = self.model.tokenizer(generated_text, return_tensors="pt").input_ids
            actual_prompt_len = prompt_tokens.shape[1]
            print(f"    Actual prompt token length for Phase 2: {actual_prompt_len}")
            print(f"    Patching positions: {original_prompt_len}:{end_plan_idx}")

            with model.edit(inplace=True):
                for layer_idx in range(num_layers):
                    for i in range(self.x):
                        self.model.model.layers[layer_idx].output[0][:, i+original_prompt_len, :] = residual_acts[layer_idx][0, i]
                
            with self.model.generate(generated_plan_added, max_new_tokens=remaining_tokens, remote=REMOTE) as tracer:
                            # First self.unmasked iterations: no masking
                with tracer.iter[:self.unmasked]:
                    pass

                # Remaining iterations: apply mask
                with tracer.iter[self.unmasked:]:
                    for layer in self.model.model.layers:
                        self._apply_mask(layer.self_attn, original_prompt_len + max_new_tokens)
                
                output_patched = self.model.generator.output.save()

            output_patchd_tokens = output_patched

            #end_plan_chars = len(generated_text) + len(original_prompt)
            patched_text = self.model.tokenizer.decode(output_patchd_tokens[0], skip_special_tokens=False)
            print(f"  Generated with patching: {patched_text}")

            model.clear_edits()

        print(f"\n=== Phase 3: Run Prompt with Unedited plan latents ===")
        with self.model.generate(original_prompt, max_new_tokens=max_new_tokens, remote=REMOTE) as tracer:
            output = self.model.generator.output.save()

        current_tokens = output
        patched_text = self.model.tokenizer.decode(current_tokens[0], skip_special_tokens=False)
        print(f"  Generated Original: {patched_text}")


        return output_patchd_tokens
    def _apply_mask(self, attn_module, seq_len):
        """Apply prefix mask to attention scores."""
        n = seq_len
        prior = self.m+self.x
        
        # Create and apply mask
        # Access attention weights - exact path depends on model
        try:
            weights = attn_module.o_proj.input[0][0]
            mask = torch.zeros((1, 1, seq_len, seq_len))
            if self.m > 0 and self.x <= n:
                mask[:, :, prior:n, :self.m] = float('-inf')
            weights[:] = weights + mask
        except Exception:
            pass  # Handle models with different architecture`



# Patches residual stream activations from diff_prompt into generated tokens.
class NoPadPlanResidualStreamIntervention:

    def __init__(self, model: LanguageModel, x: int):
        self.model = model
        self.x = x  # Number of generated token positions to replace
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
        add_ids = self.model.tokenizer(
            add_on, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        

        original_info_len = original_info_ids.shape[1]
        original_prompt_len = original_ids.shape[1]
        add_prompt_len = add_ids.shape[1]
        #end_plan_idx = self.x + add_prompt_len
        end_plan_idx = self.x

        # Constructing padding of length size(prompt_ids)
        # space_ids = self.model.tokenizer(" ", add_special_tokens=False, return_tensors="pt").input_ids
        # print(space_ids)  # tensor([[...]])
        # print(self.model.tokenizer.decode(space_ids[0]))  # verify it's a space
        # token_id = space_ids[0, 0].item()
        # pad_prompt = space_ids.repeat(1, original_prompt_len)
        # print(f"Token ID: {token_id}")
        # print(f"Token repr: {repr(self.model.tokenizer.decode([token_id]))}")
        # end_plan_idx = self.x + add_prompt_len

        print(f"Starting generation: {max_new_tokens} tokens")
        print(f"Original prompt: {original_prompt}")
        print(f"Original prompt length: {original_prompt_len} tokens")
        print(f"Will replace residual at positions [{original_prompt_len}:{end_plan_idx}]")
        print()

        # Phase 1: Generate first x tokens normally using generate()
        print(f"=== Phase 1: Normal generation (first {self.x} tokens) ===")
        with self.model.generate(original_prompt, max_new_tokens=self.x, remote=REMOTE) as tracer:
            output = self.model.generator.output.save()

        plan_tokens = output[0][original_prompt_len:]
        generated_text = self.model.tokenizer.decode(plan_tokens, skip_special_tokens=False)
        print(f"  Generated: {generated_text}")

        # Phase 2: Extract residual activations ONE TIME from diff_prompt + first x tokens
        print(f"\n=== Phase 2: Extracting residual activations ===")
        print(f"Extracting from diff_prompt + first {self.x} generated tokens")

        # Build diff context with first x generated tokens
        #diff_context = self.diff_prompt + generated_text + add_on
        diff_context = original_prompt + generated_text 
        print(f"Diff context: {diff_context}")
        print(f"Extracting at positions [{original_prompt_len}:{end_plan_idx}]")

        residual_acts, phase2_logits = self._extract_residual_activations(diff_context, original_prompt_len)
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
            #remaining_tokens = max_new_tokens - self.x - add_prompt_len
            remaining_tokens = max_new_tokens - self.x 
            print(f"\n=== Phase 3: Patching generation ({remaining_tokens} tokens) ===")
            print(f"Patching at FIXED positions [{original_prompt_len}:{end_plan_idx}]")
            print()

            num_layers = self.num_layers

            # Get the actual token length of the prompt we're sending
            prompt_tokens = self.model.tokenizer(generated_text, return_tensors="pt").input_ids
            actual_prompt_len = prompt_tokens.shape[1]
            print(f"    Actual prompt token length for Phase 3: {actual_prompt_len}")
            print(f"    Patching positions: {original_prompt_len}:{end_plan_idx}")

            with model.edit(inplace=True):
                for layer_idx in range(num_layers):
                    for i in range(self.x):
                        self.model.model.layers[layer_idx].output[0][:, i, :] = residual_acts[layer_idx][0, i]
                
            with self.model.generate(generated_text, max_new_tokens=remaining_tokens, remote=REMOTE) as tracer:
                output_patched = self.model.generator.output.save()

            output_patchd_tokens = output_patched

            #end_plan_chars = len(generated_text) + len(original_prompt)
            patched_text = self.model.tokenizer.decode(output_patchd_tokens[0], skip_special_tokens=False)
            print(f"  Generated with patching: {patched_text}")

            model.clear_edits()

        print(f"\n=== Phase 4: Run Prompt with Unedited plan latents ===")
        with self.model.generate(original_prompt, max_new_tokens=max_new_tokens, remote=REMOTE) as tracer:
            output = self.model.generator.output.save()
        
        current_tokens = output
        patched_text = self.model.tokenizer.decode(current_tokens[0], skip_special_tokens=False)
        print(f"  Generated Original: {patched_text}")

        return output_patchd_tokens
    
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
    original_prompt = "x = 7, y = 9, z = y / x + 5. First plan steps to solve for z in words, then use steps to find z." # perfect
    original_prompt = "x = 7, y = 2, z = y / x + 5. First plan steps to solve for z in words only, then use steps to find z." # thinks x = 3
    original_prompt = "I want to go to Bali, Indonesia and go swimming. First name the universal steps to plan any international trip (short list of short phrases), then use the steps to book the trip."
    # Parameters
    x = 114  # Replace prompt tokens' residual activations
    max_tokens = 300  # Generate 15 total tokens

    print(f"Original prompt: {original_prompt}")
    print(f"Will generate {x} tokens normally, extract from diff context,")
    print(f"then generate {max_tokens - x} more tokens with residual patching")
    print("=" * 70)
    print()

    # Run intervention
    #intervention = ResidualStreamReplacementIntervention(model, x)
    intervention = NoPadPlanResidualStreamIntervention(model, x)
    output = intervention(original_prompt, original_info, max_new_tokens=max_tokens)

    print()
    print("=" * 70)
    print(f"Final output: {model.tokenizer.decode(output[0])}")
