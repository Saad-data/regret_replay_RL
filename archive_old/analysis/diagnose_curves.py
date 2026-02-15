"""
Diagnostic: check learning curves for failed/poor seeds.
Reads TensorBoard events and reports:
- Did training diverge? (loss exploding)
- Did epsilon reach 0.05? (exploration ended)
- Did reward ever improve? (any learning)
- Final eval reward
"""

import os
from tensorboard.backend.event_processing import event_accumulator

RUNS_DIR = "runs"

# Which runs to diagnose
TD_ONLY_PATTERN = "rmer_LunarLander-v2_seed{}_noLFIW_noTCE"
FULL_RMER_PATTERN = "rmer_LunarLander-v2_seed{}"


def get_scalars(run_path, tag):
    """Get scalar values for a tag from a run. Returns list of (step, value)."""
    if not os.path.isdir(run_path):
        return []
    ea = event_accumulator.EventAccumulator(run_path)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def diagnose_run(run_path, run_name):
    """Run diagnostics on one run. Returns dict of checks."""
    out = {
        "name": run_name,
        "found": False,
        "loss_exploded": None,
        "loss_max": None,
        "loss_final": None,
        "epsilon_final": None,
        "epsilon_reached_005": None,
        "train_reward_max": None,
        "train_reward_ever_positive": None,
        "eval_final": None,
        "eval_count": 0,
        "steps": 0,
    }
    if not os.path.isdir(run_path):
        return out

    out["found"] = True
    loss_vals = get_scalars(run_path, "Loss/train")
    eps_vals = get_scalars(run_path, "Epsilon")
    train_reward_vals = get_scalars(run_path, "Reward/train")
    eval_vals = get_scalars(run_path, "Reward/eval")

    if loss_vals:
        steps_loss = [v[1] for v in loss_vals]
        out["loss_max"] = max(steps_loss)
        out["loss_final"] = steps_loss[-1]
        out["loss_exploded"] = out["loss_max"] > 100 or (out["loss_final"] != out["loss_final"])  # NaN
        out["steps"] = max(s[0] for s in loss_vals)

    if eps_vals:
        out["epsilon_final"] = eps_vals[-1][1]
        out["epsilon_reached_005"] = min(v[1] for v in eps_vals) <= 0.05

    if train_reward_vals:
        rewards = [v[1] for v in train_reward_vals]
        out["train_reward_max"] = max(rewards)
        out["train_reward_ever_positive"] = max(rewards) > 0

    if eval_vals:
        out["eval_final"] = eval_vals[-1][1]
        out["eval_count"] = len(eval_vals)

    return out


def main():
    print("=" * 80)
    print("DIAGNOSTIC: Learning curves (failed / poor seeds)")
    print("=" * 80)
    print("\nChecks:")
    print("  - Loss exploded: max(Loss/train) > 100 or NaN")
    print("  - Epsilon reached 0.05: exploration schedule finished")
    print("  - Train reward ever positive: any episode reward > 0")
    print("  - Eval final: last Reward/eval")
    print()

    for run_type, pattern, label in [
        ("td_only", TD_ONLY_PATTERN, "td_only (no LFIW, no TCE)"),
        ("full_rmer", FULL_RMER_PATTERN, "full_rmer"),
    ]:
        print("-" * 80)
        print(f"  {label}")
        print("-" * 80)
        for seed in range(10):
            run_dir = pattern.format(seed)
            run_path = os.path.join(RUNS_DIR, run_dir)
            d = diagnose_run(run_path, f"seed{seed}")
            if not d["found"]:
                print(f"  Seed {seed}: NOT FOUND")
                continue
            loss_ok = "EXPLODED" if d["loss_exploded"] else "ok"
            eps_ok = "yes" if d["epsilon_reached_005"] else "no"
            reward_ok = "yes" if d["train_reward_ever_positive"] else "no"
            loss_s = f"{d['loss_max']:.2f}" if d["loss_max"] is not None else "?"
            eps_s = f"{d['epsilon_final']:.3f}" if d["epsilon_final"] is not None else "?"
            train_s = f"{d['train_reward_max']:.1f}" if d["train_reward_max"] is not None else "?"
            eval_s = f"{d['eval_final']:.2f}" if d["eval_final"] is not None else "?"
            print(f"  Seed {seed}: steps={d['steps']} | loss_max={loss_s} ({loss_ok}) | "
                  f"eps_final={eps_s} (reached_0.05={eps_ok}) | "
                  f"train_reward_max={train_s} (ever_positive={reward_ok}) | "
                  f"eval_final={eval_s}")
        print()

    print("=" * 80)
    print("Open TensorBoard for full curves: tensorboard --logdir runs/")
    print("=" * 80)


if __name__ == "__main__":
    main()
