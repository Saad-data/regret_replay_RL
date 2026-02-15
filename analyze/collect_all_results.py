# scrape tb logs -> results/complete_ALL_results.json

import json
import os
import numpy as np
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    event_accumulator = None

EVAL_TAGS = ["Reward/eval", "eval/reward", "eval/final_reward", "Eval/Reward"]


def collect_method(pattern, method_name, num_seeds=10, success_threshold=None):
    rewards = []
    seeds_found = []

    print(f"\n{method_name}:")

    for seed in range(num_seeds):
        run_name = pattern.format(seed=seed)
        run_path = os.path.join("runs", run_name)

        if not os.path.exists(run_path):
            print(f"  [skip] Seed {seed}: Not found")
            continue

        try:
            ea = event_accumulator.EventAccumulator(run_path)
            ea.Reload()
            scalars = ea.Tags().get("scalars", [])

            events = None
            for tag in EVAL_TAGS:
                if tag in scalars:
                    events = ea.Scalars(tag)
                    break

            if events and len(events) > 0:
                final = events[-1].value
                rewards.append(final)
                seeds_found.append(seed)
                print(f"  ok Seed {seed}: {final:.2f}")
            else:
                print(f"  -- Seed {seed}: No eval data")

        except Exception as e:
            print(f"  err Seed {seed}: Error - {str(e)[:40]}")

    success_rate = None
    if rewards and success_threshold is not None:
        success_rate = sum(1 for r in rewards if r > success_threshold) / len(rewards)

    return {
        "method": method_name,
        "seeds": seeds_found,
        "rewards": rewards,
        "mean": float(np.mean(rewards)) if rewards else 0,
        "std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0,
        "median": float(np.median(rewards)) if rewards else 0,
        "min": float(np.min(rewards)) if rewards else 0,
        "max": float(np.max(rewards)) if rewards else 0,
        "n": len(rewards),
        "success_rate": success_rate,
    }


def main():
    if event_accumulator is None:
        print("Install tensorboard: pip install tensorboard")
        return

    print("=" * 80)
    print("COLLECTING ALL EXPERIMENTAL RESULTS")
    print("=" * 80)

    # cartpole success > 450, mountaincar > -110 (we never hit -110 anyway)
    MAIN_METHODS = {
        "CartPole-v1": {
            "Uniform": ("baseline_CartPole-v1_seed{seed}", 450),
            "PER": ("per_CartPole-v1_seed{seed}", 450),
            "TD-only": ("rmer_CartPole-v1_seed{seed}_noLFIW_noTCE", 450),
            "Full-RMER (ReMERT)": ("rmer_CartPole-v1_seed{seed}", 450),
            "Full-RMER (ReMERN)": ("remern_CartPole-v1_seed{seed}", 450),
        },
        "MountainCar-v0": {
            "Uniform": ("baseline_MountainCar-v0_seed{seed}", -110),
            "PER": ("per_MountainCar-v0_seed{seed}", -110),
            "TD-only": ("rmer_MountainCar-v0_seed{seed}_noLFIW_noTCE", -110),
            "Full-RMER (ReMERT)": ("rmer_MountainCar-v0_seed{seed}", -110),
            "Full-RMER (ReMERN)": ("remern_MountainCar-v0_seed{seed}", -110),
        },
    }

    # Ablation methods (CartPole: success > 450, MountainCar: success > -110)
    ABLATION_METHODS_CP = {
        "LFIW-only": ("ablation_lfiw_only_CartPole-v1_seed{seed}", 450),
        "TCE-only": ("ablation_tce_only_CartPole-v1_seed{seed}", 450),
        "HTD+LFIW": ("ablation_htd_lfiw_CartPole-v1_seed{seed}", 450),
        "HTD+TCE": ("ablation_htd_tce_CartPole-v1_seed{seed}", 450),
        "LFIW+TCE": ("ablation_lfiw_tce_CartPole-v1_seed{seed}", 450),
    }
    ABLATION_METHODS_MC = {
        "LFIW-only": ("ablation_lfiw_only_MountainCar-v0_seed{seed}", -110),
        "TCE-only": ("ablation_tce_only_MountainCar-v0_seed{seed}", -110),
        "HTD+LFIW": ("ablation_htd_lfiw_MountainCar-v0_seed{seed}", -110),
        "HTD+TCE": ("ablation_htd_tce_MountainCar-v0_seed{seed}", -110),
        "LFIW+TCE": ("ablation_lfiw_tce_MountainCar-v0_seed{seed}", -110),
    }

    all_results = {}

    # Collect main methods
    for env_name, methods in MAIN_METHODS.items():
        print(f"\n{'=' * 80}")
        print(f"{env_name} - MAIN METHODS")
        print(f"{'=' * 80}")

        env_results = {}
        for method_name, (pattern, threshold) in methods.items():
            results = collect_method(pattern, method_name, success_threshold=threshold)
            env_results[method_name] = results

        all_results[env_name] = env_results

    # Collect ablations - CartPole
    print(f"\n{'=' * 80}")
    print("CartPole-v1 - ABLATION STUDIES")
    print(f"{'=' * 80}")

    ablation_cp = {}
    for method_name, (pattern, threshold) in ABLATION_METHODS_CP.items():
        results = collect_method(pattern, method_name, success_threshold=threshold)
        ablation_cp[method_name] = results
    all_results["Ablations_CartPole"] = ablation_cp

    # Collect ablations - MountainCar
    print(f"\n{'=' * 80}")
    print("MountainCar-v0 - ABLATION STUDIES")
    print(f"{'=' * 80}")

    ablation_mc = {}
    for method_name, (pattern, threshold) in ABLATION_METHODS_MC.items():
        results = collect_method(pattern, method_name, success_threshold=threshold)
        ablation_mc[method_name] = results
    all_results["Ablations_MountainCar"] = ablation_mc

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/complete_ALL_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("Complete results saved: results/complete_ALL_results.json")
    print("=" * 80)

    # Print summary tables
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    for env_name in ["CartPole-v1", "MountainCar-v0"]:
        print(f"\n{env_name}:")
        print(f"{'Method':<25} {'Mean':>10} {'Std':>10} {'Success':>10} {'n':>5}")
        print("-" * 70)

        for method, res in all_results[env_name].items():
            if res.get("success_rate") is not None:
                success = f"{res['success_rate'] * 100:.0f}%"
            else:
                success = "N/A"
            print(
                f"{method:<25} {res['mean']:>10.1f} {res['std']:>10.1f} {success:>10} {res['n']:>5}"
            )

    for ablabel, abkey in [("Ablations (CartPole-v1)", "Ablations_CartPole"), ("Ablations (MountainCar-v0)", "Ablations_MountainCar")]:
        print(f"\n{ablabel}:")
        print(f"{'Config':<25} {'Mean':>10} {'Std':>10} {'Success':>10} {'n':>5}")
        print("-" * 65)
        for method, res in all_results.get(abkey, {}).items():
            if res.get("success_rate") is not None:
                success = f"{res['success_rate'] * 100:.0f}%"
            else:
                success = "N/A"
            print(
                f"{method:<25} {res['mean']:>10.1f} {res['std']:>10.1f} {success:>10} {res['n']:>5}"
            )


if __name__ == "__main__":
    main()
