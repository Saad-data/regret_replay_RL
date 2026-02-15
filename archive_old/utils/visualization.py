"""
Utility functions for visualization and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_ablation_results(all_results, save_path='plots/ablation_study.png'):
    """
    Plot bar chart comparing ablation configurations.

    Args:
        all_results: Dictionary mapping config_name -> {mean_reward, std_reward, ...}
        save_path: Path to save figure
    """
    configs = list(all_results.keys())
    means = [all_results[c]['mean_reward'] for c in configs]
    stds = [all_results[c]['std_reward'] for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Mean Evaluation Reward')
    ax.set_title('RMER Ablation Study: Component Contributions')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves(learning_curves_data, env_name, save_path='plots/learning_curves.png'):
    """
    Plot learning curves for each ablation configuration.

    Args:
        learning_curves_data: Dictionary mapping config_name -> {episode_rewards, eval_rewards}
        env_name: Environment name (for title)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Episode rewards (training)
    ax = axes[0]
    for config_name, data in learning_curves_data.items():
        rewards = np.array(data['episode_rewards'])
        if len(rewards) > 100:
            smoothed = np.convolve(rewards, np.ones(100) / 100, mode='valid')
            x = np.arange(len(smoothed))
            ax.plot(x, smoothed, label=config_name, alpha=0.9)
        elif len(rewards) > 0:
            ax.plot(rewards, label=config_name, alpha=0.9)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (smoothed)')
    ax.set_title(f'Training Rewards - {env_name}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Evaluation rewards over steps
    ax = axes[1]
    for config_name, data in learning_curves_data.items():
        eval_rewards = data.get('eval_rewards', [])
        if eval_rewards:
            steps, rewards = zip(*eval_rewards)
            ax.plot(steps, rewards, marker='o', markersize=3, label=config_name, alpha=0.9)
    ax.set_xlabel('Step')
    ax.set_ylabel('Evaluation Reward')
    ax.set_title(f'Evaluation Performance - {env_name}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_table(all_results, metrics, save_path='plots/comparison_table.png'):
    """
    Plot a table comparing methods across metrics.

    Args:
        all_results: Dictionary mapping method_name -> {metric: value, ...}
        metrics: List of metric keys to display
        save_path: Path to save figure
    """
    method_names = list(all_results.keys())
    n_methods = len(method_names)
    n_metrics = len(metrics)

    # Build cell text: rows = methods, cols = metrics
    cell_text = []
    for method_name in method_names:
        row = []
        for m in metrics:
            val = all_results[method_name].get(m, np.nan)
            if isinstance(val, (int, np.integer)):
                row.append(f'{val}')
            elif isinstance(val, float):
                row.append(f'{val:.2f}')
            else:
                row.append(str(val))
        cell_text.append(row)

    # Column headers: metric names (nicely formatted)
    col_labels = [m.replace('_', ' ').title() for m in metrics]
    row_labels = method_names

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), max(4, n_methods * 0.8)))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        rowLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    plt.title('Method Comparison')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
