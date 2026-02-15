import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Print and plot top 5 token probabilities per generated position.
def print_top_logit_probs_from_logits(self, logits_list, output_tokens, prompt_len: int):
    
    
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
def create_logit_probs_plot(self, positions_data: list):

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