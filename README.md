# Planning Latents: Prefix Intervention Experiments

This repository contains two experiments for investigating how language models use prefix information during text generation. Both experiments use the [nnsight](https://github.com/ndif-team/nnsight) library to intervene on model internals during generation.

## See Results in results

Next

1. result is torn!
2. Patch activations accross clean and corrupted plan latents

## Setup

### Requirements

```bash
pip install torch nnsight python-dotenv
```

### Configuration

Both experiments use the NDIF remote inference API. Configure your API key:

```python
from nnsight import CONFIG

CONFIG.set_default_api_key("your-api-key")
CONFIG.API.HOST = "https://api.ndif.us"
CONFIG.save()
```

Or use a `.env` file with `python-dotenv`.

## Experiments

### 1. Prefix Masking (`generate_prefix_masking.py`)

**Goal:** Investigate what happens when generated tokens cannot attend to the first `m` tokens of the input prompt.

**Mechanism:** After generating `unmasked_generations` tokens normally, all subsequent tokens have their attention masked so they cannot attend to positions `0:m` in the input. This is done by adding `-inf` to the attention scores for those positions.

**Parameters:**

- `m`: Number of prefix tokens to mask (e.g., mask "What is" in "What is the Eiffel Tower?")
- `unmasked_generations`: Number of tokens to generate normally before masking begins

````python
n = seq_len                      # Total sequence length
prior = self.m + self.unmasked   # Where masking STARTS (row-wise)
self.m                           # First m tokens (e.g., prompt/question)
self.unmasked                    # Tokens allowed to see the prompt (e.g., exact answer tokens)
```
## Visual Representation

Say `seq_len=12`, `self.m=4` (prompt), `self.unmasked=3` (exact answer tokens):

        Columns (Keys/Values being attended TO)

        |←── self.m ──→|
        |   (prompt)   |
        0   1   2   3   4   5   6   7   8   9  10  11
      ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
    0 │   │   │   │   │   │   │   │   │   │   │   │   │ ┐
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ │ self.m
    1 │   │   │   │   │   │   │   │   │   │   │   │   │ │ (PROMPT tokens)
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ │ CAN attend anywhere (trivial)
    2 │   │   │   │   │   │   │   │   │   │   │   │   │ │
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ ┘
    3 │   │   │   │   │   │   │   │   │   │   │   │   │
R     ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
o   4 │   │   │   │   │   │   │   │   │   │   │   │   │ ┐
w     ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ │ self.unmasked (PLAN)
s   5 │   │   │   │   │   │   │   │   │   │   │   │   │ │ CAN attend to PROMPT
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ │
(Q) 6 │   │   │   │   │   │   │   │   │   │   │   │   │ ┘
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ ← prior = self.m + self.unmasked
    7 │ X │ X │ X │ X │   │   │   │   │   │   │   │   │ ┐
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ │
    8 │ X │ X │ X │ X │   │   │   │   │   │   │   │   │ │ SOLUTION
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ │ CANNOT attend to PROMPT
    9 │ X │ X │ X │ X │   │   │   │   │   │   │   │   │ │ CAN attend to PLAN
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ │
   10 │ X │ X │ X │ X │   │   │   │   │   │   │   │   │ │
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ │
   11 │ X │ X │ X │ X │   │   │   │   │   │   │   │   │ ┘
      └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

**Example:**

```python
from nnsight import LanguageModel
from generate_prefix_masking import PrefixMaskIntervention

model = LanguageModel('meta-llama/Llama-3.1-8B')

prompt = "What is the Eiffel Tower?"
input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids

m = 2  # Mask first 2 tokens
unmasked_generations = 5  # Generate 5 tokens normally first

intervention = PrefixMaskIntervention(model, m, unmasked_generations)
output = intervention(input_ids, max_new_tokens=8)
````

**What this tests:** How much information about the prefix is "baked in" to the latent representations of later tokens vs. retrieved via attention at generation time.

---

### 2. Counterfactual Prefix - !Wrong Needs Custom Attention! (`generate_counterfactual_prefix.py`)

**Goal:** Investigate what happens when specific generated tokens attend to an _alternate_ prefix instead of the original, while all other tokens (including earlier generated tokens) continue to use the original prompt's representations.

**Mechanism:**

1. Pre-compute K/V representations for the alternate prefix by running it through the model
2. Generate normally with the original prompt for `unmasked_generations` tokens
3. For remaining tokens, manually recompute attention for _only_ those query positions using the alternate K/V for positions `0:m`, then patch the attention output

**Key Innovation:** Since nnsight doesn't use KV caching (the full attention is recomputed each step), simply swapping K/V would affect _all_ tokens. Instead, we:

- Extract Q, K, V from the attention module
- Manually compute attention scores for only the affected query positions
- Apply causal masking
- Compute the weighted sum with modified V
- Patch only those positions in the attention output (`o_proj.input`)

This ensures tokens before `warped_prefix_start` see the original prefix, while later tokens see the alternate prefix.

**Parameters:**

- `m`: Number of prefix tokens to swap
- `unmasked_generations`: Number of tokens to generate normally before swapping begins
- `alt_prefix_ids`: Token IDs for the alternate prefix

**Example:**

```python
from nnsight import LanguageModel
from generate_counterfactual_prefix import CounterfactualPrefixIntervention

model = LanguageModel('meta-llama/Llama-3.1-8B')

original_prompt = "What is the Eiffel Tower?"
alt_prefix = "Where"  # Will replace "What"

input_ids = model.tokenizer(original_prompt, add_special_tokens=True, return_tensors="pt").input_ids
alt_prefix_ids = model.tokenizer(alt_prefix, add_special_tokens=True, return_tensors="pt").input_ids

m = alt_prefix_ids.shape[1]  # Number of tokens in alternate prefix (including BOS)
unmasked_generations = 5

intervention = CounterfactualPrefixIntervention(model, m, unmasked_generations)
output = intervention(input_ids, alt_prefix_ids, max_new_tokens=20)
```

**What this tests:** Whether the model can be "steered" by changing what specific generated tokens attend to, even when all intermediate latents were computed with the original prefix. This probes the role of attention-based retrieval vs. feedforward computation in determining model outputs.

---

## Key Differences

| Aspect                   | Prefix Masking                            | Counterfactual Prefix                           | Residual Stream Replacement                                       |
| ------------------------ | ----------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| Intervention type        | Mask attention (add `-inf`)               | Manual attention recomputation                  | Direct residual stream patching                                   |
| What gets replaced       | Attention weights (set to 0)              | V projections for prefix positions              | Full residual stream for generated tokens                         |
| Source of replacement    | N/A (blocking)                            | Alternate prefix alone                          | Diff prompt + generated tokens                                    |
| Affected positions       | Prefix (0:m)                              | Prefix (0:m)                                    | Generated tokens (prompt_len:prompt_len+x)                        |
| When intervention occurs | During generation (steps >= unmasked)     | During generation (steps >= unmasked)           | After phase 1, ONE-TIME extraction + repeated patching            |
| Scope of impact          | Attention mechanism only                  | Attention mechanism only                        | All downstream computation                                        |
| Research question        | Is prefix info needed at generation time? | Can we redirect attention to different content? | Can the model compensate for mismatched residual representations? |

## Technical Details

All experiments leverage nnsight's tracing API:

- `model.generate(..., remote=True)` enables intervention during generation (Prefix Masking)
- `model.trace(..., remote=True)` enables intervention during single forward pass (Counterfactual Prefix, Residual Stream Replacement)
- `tracer.iter[start:end]` applies interventions only during specific generation steps
- `layer.self_attn.q_proj.output` / `k_proj.output` / `v_proj.output` accesses Q/K/V projections
- `attn_module.o_proj.input[0][0]` accesses attention output (before output projection)
- `layer.output[0]` accesses the residual stream after a complete transformer layer

### Counterfactual Prefix: Manual Attention

The counterfactual experiment manually computes attention to selectively affect only certain query positions:

```python
# For queries at positions >= warped_prefix_start:
# 1. Get Q, K, V from projections
# 2. Replace K[:, :m, :] and V[:, :m, :] with alternate prefix values
# 3. Compute: attn_scores = Q @ K.T / sqrt(head_dim)
# 4. Apply causal mask
# 5. attn_weights = softmax(attn_scores)
# 6. attn_output = attn_weights @ V
# 7. Patch o_proj.input for affected positions
```

This handles Grouped Query Attention (GQA) by repeating KV heads as needed.

### Residual Stream Replacement: Three-Phase Generation

The residual stream replacement experiment uses a three-phase approach:

```python
# Phase 1: Normal generation (first x tokens)
for step in range(x):
    next_token = generate_one_token(current_tokens)
    current_tokens = concat(current_tokens, next_token)

# Phase 2: ONE-TIME extraction from diff context
diff_context = diff_prompt + decode(generated_tokens[0:x])
residual_acts = {}
with model.trace(diff_context, remote=True):
    for layer in model.model.layers:
        residual_acts[layer_idx] = layer.output[0][:, diff_prompt_len:diff_prompt_len+x, :].save()

# Phase 3: Generate remaining tokens with FIXED position patching
for step in range(x, max_new_tokens):
    with model.trace(current_prompt, remote=True):
        for layer in model.model.layers:
            # Patch SAME positions each time
            layer.output[0][:, original_prompt_len:original_prompt_len+x, :] = residual_acts[layer_idx]
        next_token = generate_from_logits()
    current_tokens = concat(current_tokens, next_token)
```

Key points:

- Extraction happens **once** after Phase 1 completes
- The same extracted activations are reused for all subsequent generation steps
- Patch positions `[original_prompt_len:original_prompt_len+x]` remain fixed as the sequence grows
- This creates a persistent mismatch between the residual representations at those positions and the actual context

---

### 3. Residual Stream Replacement (`generate_residual_stream_replacement.py`)

**Goal:** Investigate what happens when the residual stream representations for the first `x` generated tokens come from a different context, while the original prompt context is preserved.

**Mechanism:**

1. Generate `x` tokens normally with the original prompt
2. Extract residual stream activations from `diff_prompt + first_x_generated_tokens` at positions `[diff_prompt_len:diff_prompt_len+x]` across all layers (ONE TIME)
3. For each subsequent token generation, patch the residual streams at FIXED positions `[original_prompt_len:original_prompt_len+x]` with the extracted activations
4. These patched positions remain constant as new tokens are generated

**Parameters:**

- `x`: Number of generated token positions to replace (first x after prompt)
- `diff_prompt`: Alternate prompt to extract residual activations from
- `max_new_tokens`: Total number of tokens to generate

**Example:**

```python
from nnsight import LanguageModel
from generate_residual_stream_replacement import ResidualStreamReplacementIntervention

model = LanguageModel('meta-llama/Llama-3.1-8B')

original_prompt = "To solve this math problem, first we"
diff_prompt = "To write a creative story, first we"

x = 5  # Replace first 5 generated tokens' residual activations
max_tokens = 15  # Generate 15 total tokens

intervention = ResidualStreamReplacementIntervention(model, x, diff_prompt)
output = intervention(original_prompt, max_new_tokens=max_tokens)
```

**What this tests:** How much does replacing residual stream representations (the full internal state after each layer, not just attention) affect subsequent generation? This probes whether the model can recover from "context-mismatched" intermediate representations, and whether later layers depend critically on earlier residual streams or can compensate for the mismatch.

**Key difference from Counterfactual Prefix:** Instead of swapping attention K/V projections, this replaces the complete residual stream (post-layer outputs) for specific token positions. This affects all downstream computation, not just attention-based retrieval.

---

## Model Compatibility

These experiments are designed for Llama-style architectures (tested with `meta-llama/Llama-3.1-8B`). Other architectures may require adjusting the attention module access paths.
