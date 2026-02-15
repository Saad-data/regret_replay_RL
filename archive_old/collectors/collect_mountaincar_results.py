"""Collect MountainCar-v0 results from TensorBoard (10 seeds × 4 methods)."""

import json
import os
import numpy as np
from scipy import stats

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    event_accumulator = None

EVAL_TAGS = ['Reward/eval', 'eval/reward', 'eval/final_reward', 'Eval/Reward']


def collect_method(pattern, method_name, num_seeds=10):
    """Collect results for one method."""
    rewards = []
    seeds_found = []

    print(f"\n{method_name}:")

    for seed in range(num_seeds):
        run_name = pattern.format(seed=seed)
        run_path = os.path.join("runs", run_name)

        if not os.path.exists(run_path):
            print(f"  ⚠️  Seed {seed}: Not found")
            continue

        try:
            ea = event_accumulator.EventAccumulator(run_path)
            ea.Reload()
            scalars = ea.Tags().get('scalars', [])

            events = None
            for tag in EVAL_TAGS:
                if tag in scalars:
                    events = ea.Scalars(tag)
                    break

            if events and len(events) > 0:
                final = events[-1].value
                rewards.append(final)
                seeds_found.append(seed)
                print(f"  ✓ Seed {seed}: {final:.2f}")
            else:
                print(f"  ✗ Seed {seed}: No eval data")

        except Exception as e:
            print(f"  ✗ Seed {seed}: {str(e)[:50]}")

    return {
        'method': method_name,
        'seeds': seeds_found,
        'rewards': rewards,
        'mean': float(np.mean(rewards)) if rewards else 0,
        'std': float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0,
        'median': float(np.median(rewards)) if rewards else 0,
        'min': float(np.min(rewards)) if rewards else 0,
        'max': float(np.max(rewards)) if rewards else 0,
        'n': len(rewards),
        'success_rate': sum(1 for r in rewards if r > -110) / len(rewards) if rewards else 0,
    }


def main():
    if event_accumulator is None:
        print("Install tensorboard: pip install tensorboard")
        return

    patterns = {
        'Uniform': 'baseline_MountainCar-v0_seed{seed}',
        'PER': 'per_MountainCar-v0_seed{seed}',
        'TD-only': 'rmer_MountainCar-v0_seed{seed}_noLFIW_noTCE',
        'Full-RMER': 'rmer_MountainCar-v0_seed{seed}',
    }

    print("=" * 80)
    print("COLLECTING MOUNTAINCAR-V0 RESULTS (10 seeds)")
    print("=" * 80)

    all_results = {}
    for name, pattern in patterns.items():
        results = collect_method(pattern, name)
        all_results[name] = results

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY (MountainCar-v0, 200k steps)")
    print("=" * 80)
    print(f"{'Method':<12} {'Mean':>8} {'Std':>8} {'Median':>8} {'Success%':>9} {'n':>4}")
    print("-" * 80)

    for name in ['Uniform', 'PER', 'TD-only', 'Full-RMER']:
        r = all_results[name]
        success_pct = r['success_rate'] * 100
        print(f"{name:<12} {r['mean']:>8.1f} {r['std']:>8.1f} {r['median']:>8.1f} "
              f"{success_pct:>8.0f}% {r['n']:>4}")

    print("\nNote: Success = reward > -110 (solved MountainCar)")

    # Pairwise statistical comparisons
    print("\n" + "=" * 80)
    print("PAIRWISE STATISTICAL COMPARISONS")
    print("=" * 80)

    comparisons = [
        ('Uniform', 'PER'),
        ('PER', 'TD-only'),
        ('TD-only', 'Full-RMER'),
        ('Uniform', 'Full-RMER'),
    ]

    for m1, m2 in comparisons:
        r1 = all_results[m1]['rewards']
        r2 = all_results[m2]['rewards']

        if len(r1) < 2 or len(r2) < 2:
            print(f"\n{m1} vs {m2}: Insufficient data")
            continue

        t_stat, p_val = stats.ttest_ind(r1, r2)
        diff = np.mean(r2) - np.mean(r1)
        pooled_std = np.sqrt((np.std(r1, ddof=1) ** 2 + np.std(r2, ddof=1) ** 2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        winner = m2 if diff > 0 else m1

        print(f"\n{m1} vs {m2}:")
        print(f"  Difference: {diff:+.2f} points ({winner} better)")
        print(f"  p-value: {p_val:.4f}")
        print(f"  Cohen's d: {cohens_d:.3f}")

        if p_val < 0.001:
            sig = "✅ HIGHLY significant (p < 0.001)"
        elif p_val < 0.01:
            sig = "✅ VERY significant (p < 0.01)"
        elif p_val < 0.05:
            sig = "✅ Significant (p < 0.05)"
        elif p_val < 0.10:
            sig = "⚠️  Borderline (p < 0.10)"
        else:
            sig = "❌ Not significant"
        print(f"  {sig}")

    # Hierarchy analysis
    print("\n" + "=" * 80)
    print("HIERARCHY ANALYSIS")
    print("=" * 80)

    means = {name: all_results[name]['mean'] for name in ['Uniform', 'PER', 'TD-only', 'Full-RMER']}
    sorted_methods = sorted(means.items(), key=lambda x: x[1], reverse=True)

    print("Performance ranking (best to worst, higher = better):")
    for i, (method, mean) in enumerate(sorted_methods, 1):
        print(f"  {i}. {method:<12} {mean:>8.1f}")

    expected = ['Full-RMER', 'TD-only', 'PER', 'Uniform']
    actual = [m for m, _ in sorted_methods]
    if actual == expected:
        print("\n✅ Matches expected hierarchy: Full-RMER > TD-only > PER > Uniform")
    else:
        print("\n⚠️  Differs from expected hierarchy")
        print(f"   Expected: {' > '.join(expected)}")
        print(f"   Actual:   {' > '.join(actual)}")

    os.makedirs('results', exist_ok=True)
    out_path = 'results/mountaincar_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ Results saved: {out_path}")
    print("=" * 80)
    print("\nNext: python create_mountaincar_plots.py")


if __name__ == "__main__":
    main()
