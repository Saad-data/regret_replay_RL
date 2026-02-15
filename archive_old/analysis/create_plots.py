"""
Create publication-quality plots for CartPole results.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

results_file = 'results/cartpole_results.json'
if not os.path.isfile(results_file):
    print(f"Missing {results_file}. Run collect_cartpole_results.py first.")
    exit(1)

with open(results_file, 'r') as f:
    results = json.load(f)

methods = ['Uniform', 'PER', 'TD-only', 'Full-RMER']
means = [results[m]['mean'] for m in methods]
stds = [results[m]['std'] for m in methods]
data_points = [results[m]['rewards'] for m in methods]

os.makedirs('plots', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Bar chart with error bars
ax1 = axes[0]
x = np.arange(len(methods))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
ax1.set_xlabel('Method', fontsize=14, fontweight='bold')
ax1.set_ylabel('Final Evaluation Reward', fontsize=14, fontweight='bold')
ax1.set_title('CartPole-v1 Performance Comparison\n(100k steps, 10 seeds)', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=15, ha='right')
ax1.grid(True, alpha=0.3, axis='y')
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 5,
             f'{mean:.1f}+/-{std:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Box plot
ax2 = axes[1]
bp = ax2.boxplot(data_points, labels=methods, patch_artist=True, showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_xlabel('Method', fontsize=14, fontweight='bold')
ax2.set_ylabel('Final Evaluation Reward', fontsize=14, fontweight='bold')
ax2.set_title('Distribution of Final Rewards (10 seeds)', fontsize=16, fontweight='bold')
ax2.set_xticklabels(methods, rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/cartpole_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plots/cartpole_comparison.png")

# Learning curves placeholder (use TensorBoard for full curves)
fig2, ax = plt.subplots(figsize=(12, 6))
ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('Evaluation Reward', fontsize=14, fontweight='bold')
ax.set_title('Learning Curves: CartPole-v1 (run TensorBoard for full curves)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/cartpole_learning_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plots/cartpole_learning_curves.png (placeholder; use TensorBoard for full curves)")
print("All plots created.")
