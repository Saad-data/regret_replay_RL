"""
Collect full_rmer final eval rewards from TensorBoard runs (seeds 0-9).
full_rmer = all components (no --no-lfiw, no --no-tce); run dir: rmer_LunarLander-v2_seed{N}
Writes full_rmer_results.json for final_comparison.py.
"""

from tensorboard.backend.event_processing import event_accumulator
import glob
import json
import os
import statistics


def main():
    print("Collecting full_rmer results from all seeds...")
    print("=" * 60)

    full_rmer_results = []
    runs_dir = "runs"

    for seed in range(10):  # Seeds 0-9
        # Full RMER: rmer_LunarLander-v2_seed{N} with no _noLFIW or _noTCE
        pattern = os.path.join(runs_dir, f"rmer_LunarLander-v2_seed{seed}")
        dirs = [
            d
            for d in glob.glob(pattern + "*")
            if "_no" not in d and os.path.isdir(d)
        ]

        if not dirs:
            print(f"  Seed {seed}: Not found")
            continue

        try:
            run_path = dirs[0]
            ea = event_accumulator.EventAccumulator(run_path)
            ea.Reload()

            if "Reward/eval" not in ea.Tags().get("scalars", []):
                print(f"  Seed {seed}: No eval data")
                continue

            eval_rewards = ea.Scalars("Reward/eval")
            if not eval_rewards:
                print(f"  Seed {seed}: No eval data")
                continue

            final_eval = eval_rewards[-1].value
            full_rmer_results.append((seed, final_eval))
            print(f"  Seed {seed}: {final_eval:.2f}")

        except Exception as e:
            print(f"  Seed {seed}: Error - {e}")

    print("=" * 60)
    print(f"\nResults collected: {len(full_rmer_results)}/10 seeds\n")

    if len(full_rmer_results) >= 1:
        rewards = [r for _, r in full_rmer_results]
        seeds = [s for s, _ in full_rmer_results]
        mean = statistics.mean(rewards)
        std = statistics.stdev(rewards) if len(rewards) >= 2 else 0.0
        median = statistics.median(rewards)

        print(f"full_rmer (n={len(full_rmer_results)} seeds):")
        print(f"  Mean:   {mean:.2f}")
        print(f"  Std:    {std:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Min:    {min(rewards):.2f}")
        print(f"  Max:    {max(rewards):.2f}")
        print()
        print(f"Seeds: {seeds}")
        print(f"Values: {[f'{x:.2f}' for x in rewards]}")

        # Write for final_comparison.py (sorted by seed)
        out = {"seeds": seeds, "rewards": rewards}
        with open("full_rmer_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote full_rmer_results.json ({len(rewards)} seeds)")
    else:
        print("  Not enough seeds for statistics.")


if __name__ == "__main__":
    main()
