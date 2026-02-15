"""
Collect results from CartPole-v1 experiments (4 methods x 10 seeds).
"""

import json
import os
import numpy as np
from scipy import stats

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    event_accumulator = None


def collect_method_results(pattern, method_name, env_name="CartPole-v1"):
    """Collect final eval rewards from TensorBoard logs."""
    runs_dir = "runs"
    rewards = []
    seeds = []

    for seed in range(10):
        run_name = pattern.format(env=env_name, seed=seed)
        run_path = os.path.join(runs_dir, run_name)

        if not os.path.exists(run_path):
            print(f"  Warning: {run_path} not found")
            continue

        try:
            ea = event_accumulator.EventAccumulator(run_path)
            ea.Reload()

            tag_names = ['Reward/eval', 'eval/reward', 'eval/final_reward', 'Eval/Reward']
            events = None
            for tag in tag_names:
                if tag in ea.Tags().get('scalars', []):
                    events = ea.Scalars(tag)
                    break

            if events and len(events) > 0:
                final_reward = events[-1].value
                rewards.append(final_reward)
                seeds.append(seed)
                print(f"  {method_name} seed {seed}: {final_reward:.2f}")
            else:
                print(f"  {method_name} seed {seed}: No eval data found")
        except Exception as e:
            print(f"  {method_name} seed {seed}: Error - {e}")

    return {
        'method': method_name,
        'seeds': seeds,
        'rewards': rewards,
        'mean': float(np.mean(rewards)) if rewards else 0,
        'std': float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0,
        'median': float(np.median(rewards)) if rewards else 0,
        'n': len(rewards),
    }


def main():
    env_name = "CartPole-v1"
    patterns = {
        'Uniform': 'baseline_{env}_seed{seed}',
        'PER': 'per_{env}_seed{seed}',
        'TD-only': 'rmer_{env}_seed{seed}_noLFIW_noTCE',
        'Full-RMER': 'rmer_{env}_seed{seed}',
    }

    if event_accumulator is None:
        print("Install tensorboard to collect from events: pip install tensorboard")
        return

    print("=" * 80)
    print("COLLECTING CARTPOLE-V1 RESULTS")
    print("=" * 80)
    print("")

    all_results = {}
    for method_name, pattern in patterns.items():
        print(f"{method_name}:")
        results = collect_method_results(pattern, method_name, env_name)
        all_results[method_name] = results
        print("")

    print("=" * 80)
    print("SUMMARY (CartPole-v1, 100k steps)")
    print("=" * 80)
    print(f"{'Method':<15} {'Mean':>10} {'Std':>10} {'Median':>10} {'n':>5}")
    print("-" * 80)
    for method_name, results in all_results.items():
        print(f"{method_name:<15} {results['mean']:>10.2f} {results['std']:>10.2f} "
              f"{results['median']:>10.2f} {results['n']:>5}")
    print("")

    print("=" * 80)
    print("PAIRWISE COMPARISONS")
    print("=" * 80)
    comparisons = [
        ('Uniform', 'PER'),
        ('PER', 'TD-only'),
        ('TD-only', 'Full-RMER'),
        ('Uniform', 'Full-RMER'),
    ]
    for method1, method2 in comparisons:
        r1 = all_results[method1]['rewards']
        r2 = all_results[method2]['rewards']
        if len(r1) < 2 or len(r2) < 2:
            print(f"\n{method1} vs {method2}: Insufficient data")
            continue
        t_stat, p_value = stats.ttest_ind(r1, r2)
        diff = np.mean(r2) - np.mean(r1)
        pooled_std = np.sqrt((np.std(r1, ddof=1)**2 + np.std(r2, ddof=1)**2) / 2)
        d = diff / pooled_std if pooled_std > 0 else 0
        print(f"\n{method1} vs {method2}:")
        print(f"  Difference: {diff:+.2f} points ({method2 if diff > 0 else method1} better)")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Cohen's d: {d:.3f}")
        print(f"  Significant (alpha=0.05): {'YES' if p_value < 0.05 else 'NO'}")

    os.makedirs('results', exist_ok=True)
    output_file = 'results/cartpole_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print("")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
