"""
Collect td_only final eval rewards from TensorBoard event files (seeds 0-9).
td_only runs are: rmer_LunarLander-v2_seed{N}_noLFIW_noTCE
Writes td_only_results.json for final_comparison.py.
"""

from tensorboard.backend.event_processing import event_accumulator
import json
import os
import statistics

def main():
    print("Collecting td_only results from all seeds...")
    print("=" * 60)

    td_only_results = []
    runs_dir = "runs"

    for seed in range(10):  # Seeds 0-9
        # td_only = no LFIW, no TCE (naming from train_rmer.py run_name)
        run_path = os.path.join(runs_dir, f"rmer_LunarLander-v2_seed{seed}_noLFIW_noTCE")

        if not os.path.isdir(run_path):
            print(f"  Seed {seed}: Not found")
            continue

        try:
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
            td_only_results.append((seed, final_eval))
            print(f"  Seed {seed}: {final_eval:.2f}")

        except Exception as e:
            print(f"  Seed {seed}: Error - {e}")

    print("=" * 60)
    print(f"\nResults collected: {len(td_only_results)}/10 seeds\n")

    if len(td_only_results) >= 1:
        rewards = [r for _, r in td_only_results]
        seeds = [s for s, _ in td_only_results]
        mean = statistics.mean(rewards)
        std = statistics.stdev(rewards) if len(rewards) >= 2 else 0.0
        median = statistics.median(rewards)

        print(f"td_only (n={len(td_only_results)} seeds):")
        print(f"  Mean:   {mean:.2f}")
        print(f"  Std:    {std:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Min:    {min(rewards):.2f}")
        print(f"  Max:    {max(rewards):.2f}")
        print(f"\nSeeds: {seeds}")
        print(f"Values: {[f'{x:.2f}' for x in rewards]}")

        out = {"seeds": seeds, "rewards": rewards}
        with open("td_only_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote td_only_results.json ({len(rewards)} seeds)")
    else:
        print("  Not enough seeds collected for statistics.")


if __name__ == "__main__":
    main()
