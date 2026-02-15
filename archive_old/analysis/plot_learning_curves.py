"""Plot learning curves to understand failures (success vs failed seeds)."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Install tensorboard: pip install tensorboard")
    exit(1)

EVAL_TAGS = ['Reward/eval', 'eval/reward', 'eval/final_reward', 'Eval/Reward']


def get_learning_curve(run_name):
    """Extract learning curve from TensorBoard. Returns (steps, values) or (None, None)."""
    run_path = os.path.join("runs", run_name)
    if not os.path.exists(run_path):
        return None, None

    try:
        ea = event_accumulator.EventAccumulator(run_path)
        ea.Reload()
        scalars = ea.Tags().get('scalars', [])
        for tag in EVAL_TAGS:
            if tag in scalars:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                return steps, values
    except Exception:
        pass
    return None, None


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    methods = [
        ('PER', 'per_CartPole-v1_seed{}'),
        ('Full-RMER', 'rmer_CartPole-v1_seed{}'),
        ('TD-only', 'rmer_CartPole-v1_seed{}_noLFIW_noTCE'),
        ('Uniform', 'baseline_CartPole-v1_seed{}'),
    ]

    for idx, (method_name, pattern) in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        failed_handles, failed_labels = [], []

        for seed in range(10):
            run_name = pattern.format(seed)
            steps, rewards = get_learning_curve(run_name)

            if not steps or not rewards:
                continue

            final = rewards[-1]
            success = final > 450
            color = 'green' if success else 'red'
            alpha = 0.7 if success else 1.0
            linewidth = 1 if success else 2

            line, = ax.plot(steps, rewards, color=color, alpha=alpha, linewidth=linewidth, label=f'Seed {seed} ({final:.0f})')
            if not success:
                failed_handles.append(line)
                failed_labels.append(f'Seed {seed} ({final:.0f})')

        ax.set_xlabel('Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Eval Reward', fontsize=12, fontweight='bold')
        ax.set_title(
            f'{method_name} Learning Curves\n(Red = Failed, Green = Success)',
            fontsize=13, fontweight='bold',
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(y=450, color='black', linestyle='--', alpha=0.5, label='Success (450)')
        ax.set_ylim([0, 550])

        if failed_handles:
            th_line = Line2D([0], [0], color='black', linestyle='--', alpha=0.5, label='Success (450)')
            ax.legend(failed_handles + [th_line], failed_labels + ['Success (450)'], loc='lower right', fontsize=8)
        else:
            ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out = 'plots/learning_curves_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
