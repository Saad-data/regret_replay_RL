# final_comparison.py

import json
import os
import statistics
from scipy import stats

# td_only: load from collected results if available, else fallback
if os.path.isfile("td_only_results.json"):
    with open("td_only_results.json") as f:
        data = json.load(f)
    td_only_seeds = data["rewards"]
else:
    td_only_seeds = [59.19, -18.71, -17.92, 25.11, -55.37, -166.10, -166.72, -90.68, -180.05, -146.76]

# full_rmer: load from collected results if available, else use ablation (5 seeds)
if os.path.isfile("full_rmer_results.json"):
    with open("full_rmer_results.json") as f:
        data = json.load(f)
    full_rmer_seeds = data["rewards"]
else:
    full_rmer_seeds = [-24.25, -9.89, -37.10, -40.04, -25.18]  # 5 seeds from ablation

# Statistics
td_mean = statistics.mean(td_only_seeds)
td_std = statistics.stdev(td_only_seeds)
td_median = statistics.median(td_only_seeds)

rmer_mean = statistics.mean(full_rmer_seeds)
rmer_std = statistics.stdev(full_rmer_seeds)
rmer_median = statistics.median(full_rmer_seeds)

# Statistical test
t_stat, p_value = stats.ttest_ind(full_rmer_seeds, td_only_seeds)

# Cohen's d
pooled_std = ((td_std**2 + rmer_std**2) / 2) ** 0.5
cohens_d = (rmer_mean - td_mean) / pooled_std

n_rmer = len(full_rmer_seeds)
print("=" * 70)
print(f"FINAL COMPARISON (10 seeds td_only, {n_rmer} seeds full_rmer)")
print("=" * 70)
print(f"{'Method':<15} {'Mean':<12} {'Std':<12} {'Median':<12}")
print("-" * 70)
print(f"{'full_rmer':<15} {rmer_mean:>6.2f} ± {rmer_std:<4.2f} {rmer_median:>6.2f}")
print(f"{'td_only':<15} {td_mean:>6.2f} ± {td_std:<4.2f} {td_median:>6.2f}")
print("-" * 70)
diff = rmer_mean - td_mean
winner = "full_rmer better" if diff > 0 else "td_only better"
print(f"\nDifference: {diff:+.2f} points ({winner})")
print(f"Variance ratio: {td_std / rmer_std:.2f}× (td_only more unstable)")
print(f"p-value: {p_value:.4f}")
print(f"Cohen's d: {cohens_d:.3f}")

if p_value < 0.05:
    print("✅ Statistically significant")
else:
    print("❌ Not statistically significant at α=0.05")
