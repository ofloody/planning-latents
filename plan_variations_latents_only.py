import torch
import torch.nn.functional as F
from nnsight import LanguageModel, CONFIG
from dotenv import load_dotenv
import os
import numpy as np

REMOTE = False
DEVICE = "cuda"
OFFSET = 0  # Number of final layers to skip when patching (0 = patch all)

# ============================================================
# Plan Variation Definitions
# ============================================================

# Human-written plan variations for travel planning prompts.
# Format should roughly match model output style (numbered steps).
# These get concatenated with the original prompt for residual extraction.

TRAVEL_PLAN_VARIATIONS_HUMAN = {
    "rephrased_human": (
        "\n\n## Universal Steps for Planning an International Trip\n\n"
        "1. Area and Dates - Where and when the trip is happening\n"
        "2. Max Budget - Most you are able to spend and is that feasible?\n"
        "3. Book Flights and Lodging - Find good prices and reasonable options at around 6-8 weeks out\n"
        "4. Activities - Priorities to book out and plan around\n"
        "5. Visa and Passport - Research local visa requirements and check your passport\n"
        "6. Packing List - plan out what you have and what you need to get\n"
        "7. Travel Insurence - consider travel insurence for conflicts, theft, and tight connections\n"
        "8. Currency - consider preordering from you bank for better exchange rates\n"
        "9. Emergency Considerations - what could go wrong and who can you call in the area?\n\n"
    ),
    "different_human": (
        "\n\n## Universal Steps for Planning an International Trip\n\n"
        "1. Set a realistic travel budget and timeline\n"
        "2. Learn important phrases in the local language\n"
        "3. Research transportation options and safety guidence to and from each location\n"
        "4. Pack weather-appropriate clothing and travel basics in suitcase or backpack\n"
        "5. Notify your bank and set up international phone service\n"
        "6. Pick activities to plan your day to day around\n\n"
    ),
    "opposite_human": (
        "\n\n## Universal Steps for Planning an International Trip\n\n"
        "1. Do Not Disturb - Set your status as away and turn your phone off\n"
        "2. Clear Your Calendar - stop making any plans and cancel everything that week\n"
        "3. Indulgent Activities - Get your watchlist ready to roll non-stop! Find all the books you've been wanting to finish\n"
        "4. Prep with Chores - make your home clean and comfortable to hole up in\n"
        "5. Bucket List - Only make one commitment to yourself of a hobby you've been wanting to try or something you've been putting off\n\n"
    ),
}

MATH_PLAN_VARIATIONS_HUMAN = {
    "rephrased_human": (
        "\n\n## Steps to Solve for z\n\n"
        "1. Note down the given values of x and y\n"
        "2. Compute y divided by x\n"
        "3. Add 5 to the result from step 2\n"
        "4. The sum is the value of z\n\n"
    ),
    "different_human": (
        "\n\n## Steps to Solve for z\n\n"
        "1. Rewrite the equation z = y/x + 5 with known values\n"
        "2. Simplify the fraction y/x to a decimal\n"
        "3. Combine the decimal with 5 to get z\n\n"
    ),
    "opposite_human": (
        "\n\n## Steps to Solve for z\n\n"
        "1. Ignore the given values of x and y\n"
        "2. Multiply x by y instead of dividing\n"
        "3. Subtract 5 instead of adding it\n"
        "4. Use the wrong formula entirely\n\n"
    ),
}

# ============================================================
# LLM Plan Variation Generation
# ============================================================

LLM_VARIATION_PROMPTS = {
    "rephrased": (
        "Below is a plan with numbered steps. Rephrase each step using "
        "different words while keeping the exact same meaning and order. "
        "Output ONLY the rephrased plan, with header ## Universal Steps for Planning an International Trip\n\n"
    ),
    "different": (
        "Below is a plan with numbered steps for a task. Write a completely "
        "different set of steps (different approach, different order) that "
        "still achive the same goal as instructed in prompt."
        "Output ONLY this plan, with header ## Universal Steps for Planning an International Trip\n\n"
    ),
    "opposite": (
        "Below is a plan with numbered steps. Write the opposite plan - steps "
        "that would lead to an opposite goal. Each step should "
        "be good advice."
        "Output ONLY this plan, with header ## Universal Steps for Planning an International Trip\n\n"
    ),
}


def generate_llm_plan_variation(model, own_plan_text, variation_type):
    """Generate a single plan variation using the model itself.

    Args:
        model: The LanguageModel instance.
        own_plan_text: The model's own generated plan text.
        variation_type: One of "rephrased", "different", "opposite".

    Returns:
        The generated variation text string.
    """
    prompt = LLM_VARIATION_PROMPTS[variation_type].format(plan_text=own_plan_text)

    with model.generate(prompt, do_sample=False, max_new_tokens=200, remote=REMOTE) as tracer:
        output = model.generator.output.save()

    full_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    variation_text = full_text[len(prompt):]

    # Trim at a natural stopping point (double newline or next section header)
    for stop in ["\n\n##", "\n\n**", "\n\nNow", "\n\nApplying", "\n\nUsing"]:
        idx = variation_text.find(stop)
        if idx != -1:
            variation_text = variation_text[:idx]
            break

    return variation_text.strip()


def generate_all_llm_variations(model, own_plan_text):
    """Generate rephrased, different, and opposite plans using the model.

    Returns:
        Dict mapping variation names to plan text strings.
    """
    llm_variations = {}
    for vtype in ["rephrased", "different", "opposite"]:
        key = f"{vtype}_llm"
        print(f"  Generating LLM variation: {vtype}...")
        text = generate_llm_plan_variation(model, own_plan_text, vtype)
        llm_variations[key] = "\n\n" + text + "\n\n"
        print(f"    Generated {len(text)} chars")
    return llm_variations


# ============================================================
# Intervention Class
# ============================================================

class PlanVariationIntervention:
    """Patches residual stream activations from plan variations.

    Workflow:
        1. Trace prompt + plan_variation to extract residual activations.
        2. Tokenize plan_variation to get plan tokens and their count.
        3. Patch extracted residuals into first token positions.
        4. Generate execution given the plan_variation tokens with patched residuals.
    """

    def __init__(self, model: LanguageModel):
        self.model = model
        self.num_layers = model.config.num_hidden_layers

    def _extract_residual_activations(self, prompt: str, extract_start: int) -> dict:
        """Extract residual activations from all layers starting at extract_start."""
        num_layers = self.num_layers
        print(f"  Extracting {num_layers} layers...")

        with self.model.trace(prompt, do_sample=False, add_special_tokens=True, remote=REMOTE):
            residual_list = list().save()
            for layer_idx in range(num_layers):
                residual_list.append(
                    self.model.model.layers[layer_idx].output[:, extract_start:, :]
                )

        print(f"  After trace: {num_layers} entries")
        print(f"  First entry shape: {residual_list[0].shape}")

        residual_acts = {i: residual_list[i] for i in range(len(residual_list))}
        return residual_acts

    def run(self, original_prompt: str, plan_text: str,
            variation_name: str = "own_plan", max_additional_tokens: int = 500):
        """Run the patching intervention with a given plan variation.

        Args:
            original_prompt: The task prompt.
            plan_text: The plan text to extract residuals from and feed as input.
            variation_name: Label for this variation (for logging).
            max_additional_tokens: Tokens to generate after plan section.

        Returns:
            Dict with 'patched_output', 'variation_name', 'plan_text', 'plan_token_len'.
        """
        print(f"\n{'='*70}")
        print(f"VARIATION: {variation_name}")
        print(f"{'='*70}")

        # --- Tokenize prompt and plan ---
        original_prompt_len = self.model.tokenizer(
            original_prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids.shape[1]

        plan_token_ids = self.model.tokenizer(
            plan_text, return_tensors="pt", add_special_tokens=False
        ).input_ids
        plan_token_len = plan_token_ids.shape[1]

        print(f"  Prompt tokens: {original_prompt_len}")
        print(f"  Plan tokens: {plan_token_len}")
        print(f"  Plan text: {plan_text[:150]}...")

        # --- Phase 1: Extract residuals from prompt + plan_text ---
        print(f"\n=== Phase 1: Extracting residuals from prompt + plan_variation ===")
        diff_context = original_prompt + plan_text
        print(f"  Extracting at positions [{original_prompt_len}:]")

        residual_acts = self._extract_residual_activations(diff_context, original_prompt_len)

        if len(residual_acts) > 0:
            residual_acts = {
                k: torch.tensor(v).cpu() if not isinstance(v, torch.Tensor) else v.clone().cpu()
                for k, v in residual_acts.items()
            }
            print(f"  Extracted activations from {len(residual_acts)} layers")
            print(f"  Activation shape per layer: {residual_acts[0].shape}")
        else:
            raise RuntimeError("Failed to extract residual activations - dict is empty")

        # --- Phase 2: Patch residuals and generate execution ---
        print(f"\n=== Phase 2: Patching {plan_token_len} positions, generating {max_additional_tokens} tokens ===")

        num_layers = self.num_layers

        with self.model.edit(inplace=True):
            for layer_idx in range(num_layers - OFFSET):
                for i in range(plan_token_len):
                    self.model.model.layers[layer_idx].output[:, i, :] = residual_acts[layer_idx][0, i]

        with self.model.generate(plan_token_ids, do_sample=False, max_new_tokens=max_additional_tokens, remote=REMOTE) as tracer:
            output_patched = self.model.generator.output.save()

        self.model.clear_edits()

        patched_text = self.model.tokenizer.decode(output_patched[0], skip_special_tokens=True)
        print(f"\nPatched output:\n{patched_text}")

        return {
            "variation_name": variation_name,
            "plan_text": plan_text,
            "patched_output": patched_text,
            "plan_token_len": plan_token_len,
        }


# ============================================================
# Experiment Runner
# ============================================================

def get_own_plan(model, prompt, delimiter="\n##", calibration_tokens=200):
    """Generate with the model to get its own plan text.

    Returns:
        The model's own plan text (up to the second delimiter).
    """
    with model.generate(prompt, do_sample=False, max_new_tokens=calibration_tokens, remote=REMOTE) as tracer:
        output = model.generator.output.save()

    full_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    generated_only = full_text[len(prompt):]

    first = generated_only.find(delimiter)
    second = generated_only.find(delimiter, first + len(delimiter))
    if second == -1:
        first = generated_only.find('\n**')
        second = generated_only.find('\n**', first + len(delimiter))
        if second == -1:
            raise ValueError(f"Second '{delimiter}' not found in first {calibration_tokens} generated tokens.")

    plan_text = generated_only[:second]
    print(f"Own plan ({len(plan_text)} chars):\n{plan_text}")
    return plan_text


def run_all_variations(model, prompt, human_variations, max_additional_tokens=500,
                       generate_llm_variations_flag=True):
    """Run all plan variations for a single prompt.

    Returns:
        Dict mapping variation names to result dicts.
    """
    intervention = PlanVariationIntervention(model)
    results = {}

    # 1. Get the model's own plan
    print("\n" + "=" * 70)
    print("GETTING MODEL'S OWN PLAN")
    print("=" * 70)
    own_plan_text = get_own_plan(model, prompt)

    # 2. Generate LLM variations from the model's own plan
    llm_variations = {}
    if generate_llm_variations_flag:
        print("\n" + "=" * 70)
        print("GENERATING LLM PLAN VARIATIONS")
        print("=" * 70)
        llm_variations = generate_all_llm_variations(model, own_plan_text)
        for key, text in llm_variations.items():
            print(f"\n  {key}:\n{text}")

    # 3. Combine all variations: own plan + human + LLM
    all_variations = {"own_plan": own_plan_text}
    all_variations.update(human_variations)
    all_variations.update(llm_variations)

    # 4. Run each variation
    for var_name, plan_text in all_variations.items():
        result = intervention.run(
            prompt,
            plan_text=plan_text,
            variation_name=var_name,
            max_additional_tokens=max_additional_tokens,
        )
        results[var_name] = result

    return results


def print_summary(results):
    """Print a comparison summary of all variation results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    for var_name, result in results.items():
        print(f"\n--- {var_name} ---")
        print(f"  Plan tokens patched: {result['plan_token_len']}")
        print(f"  Plan: {result['plan_text'][:100]}...")
        print(f"  Patched output (first 200 chars): {result['patched_output'][:200]}")
        print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    load_dotenv()

    model = LanguageModel('google/gemma-2-9b-it', dispatch=True)

    # --- Travel planning prompts ---
    travel_prompts = [
        "My friends and I are going to the Berlin, Germany and go surfing. First name the universal steps to plan any international trip (short list of short phrases), then use the steps to book the trip.",
        "I want to go to Bali, Indonesia and go swimming. First name the universal steps to plan any international trip (short list of short phrases), then use the steps to book the trip.",
        "I want to travel to Fiji and go skiing. First name the universal steps to plan any international trip (short list of short phrases), then use the steps to book the trip.",
    ]

    # --- Math prompts ---
    math_prompts = [
        "x = 7, y = 9, z = y / x + 5. First plan steps to solve for z in words, then use steps to find z.",
        "x = 4, y = 26, z = y / x + 5. First plan steps to solve for z in words, then use steps to find z.",
    ]

    max_additional_tokens = 500

    all_results = {}

    # Run travel prompt experiments
    for prompt in travel_prompts:
        print(f"\n{'#'*70}")
        print(f"PROMPT: {prompt}")
        print(f"{'#'*70}")
        results = run_all_variations(
            model, prompt,
            human_variations=TRAVEL_PLAN_VARIATIONS_HUMAN,
            max_additional_tokens=max_additional_tokens,
            generate_llm_variations_flag=True,
        )
        all_results[prompt] = results
        print_summary(results)

    # Run math prompt experiments
    for prompt in math_prompts:
        print(f"\n{'#'*70}")
        print(f"PROMPT: {prompt}")
        print(f"{'#'*70}")
        results = run_all_variations(
            model, prompt,
            human_variations=MATH_PLAN_VARIATIONS_HUMAN,
            max_additional_tokens=max_additional_tokens,
            generate_llm_variations_flag=True,
        )
        all_results[prompt] = results
        print_summary(results)
