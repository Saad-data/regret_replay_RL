"""
Create final comparison plots (bar, box, optional scatter).
Reads results/cartpole_results.json or results/cartpole_final_results.json.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for name in ['results/cartpole_final_results.json', 'results/cartpole_results.json']:
    if os.path.isfile(name):
        results_file = name
        break
else:
    print("Missing results. Run: python collect_cartpole_results.py  (or collect_all_results.py)")
    exit(1)

with open(results_file, 'r') as f:
    results = json.load(f)

methods = ['Uniform', 'PER', 'TD-only', 'Full-RMER']
means = [results[m]['mean'] for m in methods]
stds = [results[m]['std'] for m in methods]
data_points = [results[m]['rewards'] for m in methods]
n_seeds = len(results[methods[0]]['rewards']) if results[methods[0]]['rewards'] else 0

os.makedirs('plots', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Bar chart
ax1 = axes[0]
x = np.arange(len(methods))
bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
ax1.set_xlabel('Method', fontsize=14, fontweight='bold')
ax1.set_ylabel('Final Evaluation Reward', fontsize=14, fontweight='bold')
ax1.set_title(f'CartPole-v1 Final Comparison\n({n_seeds} seeds)', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=15, ha='right')
ax1.grid(True, alpha=0.3, axis='y')
for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 5,
             f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Box plot
ax2 = axes[1]
bp = ax2.boxplot(data_points, tick_labels=methods, patch_artist=True, showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_xlabel('Method', fontsize=14, fontweight='bold')
ax2.set_ylabel('Final Evaluation Reward', fontsize=14, fontweight='bold')
ax2.set_title(f'Distribution ({n_seeds} seeds)', fontsize=16, fontweight='bold')
ax2.set_xticklabels(methods, rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/cartpole_final_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plots/cartpole_final_comparison.png")

# Seed scatter: one subplot per method, x=seed, y=reward
fig2, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for i, method in enumerate(methods):
    rewards = results[method]['rewards']
    seeds = results[method]['seeds']
    axes[i].scatter(seeds, rewards, alpha=0.8, s=60, c=colors[i])
    axes[i].axhline(results[method]['mean'], color=colors[i], linestyle='--', alpha=0.7, label=f"mean={results[method]['mean']:.1f}")
    axes[i].set_xlabel('Seed')
    axes[i].set_ylabel('Final reward')
    axes[i].set_title(method)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
plt.suptitle('CartPole-v1: Final reward by seed', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/cartpole_seed_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plots/cartpole_seed_scatter.png")
print("All final plots created.")
