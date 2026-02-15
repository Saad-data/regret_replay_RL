"""Create MountainCar vs CartPole comparison plots."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

plt.rcParams['font.size'] = 11

# Load MountainCar results
mc_path = 'results/mountaincar_results.json'
if not os.path.isfile(mc_path):
    print(f"Missing {mc_path}. Run: python collect_mountaincar_results.py")
    exit(1)

with open(mc_path) as f:
    mc_results = json.load(f)

# Load CartPole results (optional for side-by-side)
cp_path = 'results/cartpole_final_results.json'
if not os.path.isfile(cp_path):
    cp_path = 'results/cartpole_results.json'
if not os.path.isfile(cp_path):
    cp_results = None
else:
    with open(cp_path) as f:
        cp_results = json.load(f)

methods = ['Uniform', 'PER', 'TD-only', 'Full-RMER']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
os.makedirs('plots', exist_ok=True)

if cp_results is not None:
    # Side-by-side: CartPole vs MountainCar
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CartPole
    ax1 = axes[0]
    cp_means = [cp_results[m]['mean'] for m in methods]
    cp_stds = [cp_results[m]['std'] for m in methods]
    x = np.arange(len(methods))
    bars1 = ax1.bar(x, cp_means, yerr=cp_stds, capsize=8, alpha=0.75,
                    color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Final Reward (Higher = Better)', fontsize=13, fontweight='bold')
    ax1.set_title('CartPole-v1\n(Dense Rewards)', fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 550])
    for bar, mean, std in zip(bars1, cp_means, cp_stds):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 10,
                 f'{mean:.0f}±{std:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # MountainCar
    ax2 = axes[1]
    mc_means = [mc_results[m]['mean'] for m in methods]
    mc_stds = [mc_results[m]['std'] for m in methods]
    bars2 = ax2.bar(x, mc_means, yerr=mc_stds, capsize=8, alpha=0.75,
                    color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Final Reward (Higher = Better)', fontsize=13, fontweight='bold')
    ax2.set_title('MountainCar-v0\n(Sparse Rewards)', fontsize=15, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=-110, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Success (-110)')
    ax2.legend(loc='lower right')
    for bar, mean, std in zip(bars2, mc_means, mc_stds):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + (abs(std) if std != 0 else 5) + 5,
                 f'{mean:.0f}±{abs(std):.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/cartpole_vs_mountaincar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/cartpole_vs_mountaincar.png")

    # Ranking comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    cp_means = [cp_results[m]['mean'] for m in methods]
    mc_means = [mc_results[m]['mean'] for m in methods]
    cp_ranks = {m: i for i, (m, _) in enumerate(sorted(zip(methods, cp_means), key=lambda x: -x[1]), 1)}
    mc_ranks = {m: i for i, (m, _) in enumerate(sorted(zip(methods, mc_means), key=lambda x: -x[1]), 1)}

    for i, method in enumerate(methods):
        ax.plot([0, 1], [cp_ranks[method], mc_ranks[method]],
                marker='o', markersize=12, linewidth=3, color=colors[i], label=method, alpha=0.7)
        ax.text(-0.05, cp_ranks[method], method, ha='right', va='center', fontsize=11, fontweight='bold')
        ax.text(1.05, mc_ranks[method], method, ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim([-0.3, 1.3])
    ax.set_ylim([0.5, 4.5])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CartPole-v1\n(Dense)', 'MountainCar-v0\n(Sparse)'], fontsize=13, fontweight='bold')
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['1st', '2nd', '3rd', '4th'])
    ax.invert_yaxis()
    ax.set_ylabel('Rank (1 = Best)', fontsize=13, fontweight='bold')
    ax.set_title('Method Rankings: Environment-Specific Performance', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/ranking_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/ranking_comparison.png")
else:
    # MountainCar only: bar + box
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(methods))
    mc_means = [mc_results[m]['mean'] for m in methods]
    mc_stds = [mc_results[m]['std'] for m in methods]
    mc_data = [mc_results[m]['rewards'] for m in methods]

    ax1 = axes[0]
    bars = ax1.bar(x, mc_means, yerr=mc_stds, capsize=8, alpha=0.75, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Final Reward')
    ax1.set_title('MountainCar-v0 (200k steps, 10 seeds)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.axhline(y=-110, color='red', linestyle='--', alpha=0.7, label='Success (-110)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, mean, std in zip(bars, mc_means, mc_stds):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + abs(std) + 5,
                 f'{mean:.0f}±{abs(std):.0f}', ha='center', va='bottom', fontsize=10)

    ax2 = axes[1]
    bp = ax2.boxplot(mc_data, tick_labels=methods, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Final Reward')
    ax2.set_title('MountainCar-v0 Distribution')
    ax2.axhline(y=-110, color='red', linestyle='--', alpha=0.7)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('plots/mountaincar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/mountaincar_comparison.png")

print("\n✅ All MountainCar plots created.")
