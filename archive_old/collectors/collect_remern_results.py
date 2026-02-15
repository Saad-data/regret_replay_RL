"""
Collect ReMERN training and test results from TensorBoard runs.
"""

import json
import os
import numpy as np
from scipy import stats

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    event_accumulator = None

EVAL_TAGS = ['Reward/eval', 'eval/reward', 'eval/final_reward', 'Eval/Reward']


def collect_remern_env(pattern, env_name, num_seeds=10):
    """Collect ReMERN results for one environment."""
    rewards = []
    seeds_found = []

    for seed in range(num_seeds):
        run_name = pattern.format(env=env_name, seed=seed)
        run_path = os.path.join("runs", run_name)

        if not os.path.exists(run_path):
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
        except Exception:
            pass

    return {
        'env': env_name,
        'method': 'ReMERN',
        'seeds': seeds_found,
        'rewards': rewards,
        'mean': float(np.mean(rewards)) if rewards else None,
        'std': float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0,
        'median': float(np.median(rewards)) if rewards else None,
        'min': float(np.min(rewards)) if rewards else None,
        'max': float(np.max(rewards)) if rewards else None,
        'n': len(rewards),
    }


def main():
    if event_accumulator is None:
        print("Install tensorboard: pip install tensorboard")
        return

    pattern = 'remern_{env}_seed{seed}'

    print("=" * 70)
    print("ReMERN TRAINING & TEST RESULTS")
    print("=" * 70)

    all_results = {}

    # CartPole-v1
    print("\nCartPole-v1:")
    cp = collect_remern_env(pattern, "CartPole-v1")
    all_results['CartPole-v1'] = cp
    if cp['rewards']:
        for seed, r in zip(cp['seeds'], cp['rewards']):
            print(f"  Seed {seed}: {r:.2f}")
        print(f"  Mean: {cp['mean']:.2f}  Std: {cp['std']:.2f}  n={cp['n']}")
    else:
        print("  No runs found.")

    # MountainCar-v0
    print("\nMountainCar-v0:")
    mc = collect_remern_env(pattern, "MountainCar-v0")
    all_results['MountainCar-v0'] = mc
    if mc['rewards']:
        for seed, r in zip(mc['seeds'], mc['rewards']):
            print(f"  Seed {seed}: {r:.2f}")
        print(f"  Mean: {mc['mean']:.2f}  Std: {mc['std']:.2f}  n={mc['n']}")
        if mc.get('mean') is not None and mc['mean'] > -200:
            success = sum(1 for r in mc['rewards'] if r > -110)
            print(f"  Success (reward > -110): {success}/{mc['n']}")
    else:
        print("  No runs found.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Environment':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'n':>5}")
    print("-" * 70)
    for env_name, res in all_results.items():
        if res['n'] > 0:
            print(f"{env_name:<25} {res['mean']:>10.2f} {res['std']:>10.2f} "
                  f"{res['min']:>10.2f} {res['max']:>10.2f} {res['n']:>5}")
        else:
            print(f"{env_name:<25} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {res['n']:>5}")
    print("=" * 70)

    os.makedirs('results', exist_ok=True)
    out_path = 'results/remern_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}\n")


if __name__ == "__main__":
    main()
