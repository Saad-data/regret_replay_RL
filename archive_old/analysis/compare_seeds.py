"""
Compare good vs bad seeds: what differs?
Good: 0, 1, 8. Bad: 6, 7, 9.
Uses TensorBoard events. Q-values are NOT logged (add to train_rmer.py to check).
"""

import os
from tensorboard.backend.event_processing import event_accumulator

RUNS_DIR = "runs"
GOOD_SEEDS = [0, 1, 8]
BAD_SEEDS = [6, 7, 9]
TD_ONLY_PATTERN = "rmer_LunarLander-v2_seed{}_noLFIW_noTCE"
FULL_RMER_PATTERN = "rmer_LunarLander-v2_seed{}"
EARLY_STEP_MAX = 30_000   # "early" = steps before this
LATE_STEP_MIN = 170_000   # "late" = steps after this


def get_scalars(run_path, tag):
    if not os.path.isdir(run_path):
        return []
    ea = event_accumulator.EventAccumulator(run_path)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def series_in_window(points, step_min, step_max):
    """Points (step, value) where step_min <= step <= step_max."""
    return [v for s, v in points if step_min <= s <= step_max]


def run_metrics(run_path):
    """Compute metrics for one run."""
    loss = get_scalars(run_path, "Loss/train")
    length = get_scalars(run_path, "Length/train")
    reward_train = get_scalars(run_path, "Reward/train")
    eval_pts = get_scalars(run_path, "Reward/eval")

    early_loss = series_in_window(loss, 1000, EARLY_STEP_MAX)
    late_loss = series_in_window(loss, LATE_STEP_MIN, 300_000)
    early_length = series_in_window(length, 0, EARLY_STEP_MAX)
    early_reward = series_in_window(reward_train, 0, EARLY_STEP_MAX)

    return {
        "loss_early_mean": sum(early_loss) / len(early_loss) if early_loss else None,
        "loss_late_mean": sum(late_loss) / len(late_loss) if late_loss else None,
        "loss_early_max": max(early_loss) if early_loss else None,
        "length_early_mean": sum(early_length) / len(early_length) if early_length else None,
        "length_early_min": min(early_length) if early_length else None,
        "reward_early_mean": sum(early_reward) / len(early_reward) if early_reward else None,
        "eval_curve": eval_pts,  # list of (step, eval_reward)
        "n_early_episodes": len(early_length),
    }


def main():
    print("=" * 80)
    print("GOOD vs BAD SEEDS COMPARISON")
    print("Good seeds:", GOOD_SEEDS, "| Bad seeds:", BAD_SEEDS)
    print("=" * 80)
    print("\nNote: Initial Q-values are NOT logged. Add Q-value logging to train_rmer.py")
    print("      to compare initial Q-values (e.g. mean max Q per batch).\n")

    for run_type, pattern, label in [
        ("td_only", TD_ONLY_PATTERN, "td_only"),
        ("full_rmer", FULL_RMER_PATTERN, "full_rmer"),
    ]:
        print("-" * 80)
        print(f"  {label}")
        print("-" * 80)

        good_metrics = []
        bad_metrics = []
        for seed in GOOD_SEEDS:
            path = os.path.join(RUNS_DIR, pattern.format(seed))
            m = run_metrics(path)
            m["seed"] = seed
            good_metrics.append(m)
        for seed in BAD_SEEDS:
            path = os.path.join(RUNS_DIR, pattern.format(seed))
            m = run_metrics(path)
            m["seed"] = seed
            bad_metrics.append(m)

        def avg(metrics, key):
            vals = [m[key] for m in metrics if m.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        def avg_curve(metrics, key="eval_curve"):
            """Average eval reward at each step (step -> mean reward)."""
            by_step = {}
            for m in metrics:
                for step, val in m.get(key, []):
                    by_step.setdefault(step, []).append(val)
            return {s: sum(vals) / len(vals) for s, vals in by_step.items()}

        # Early episode length
        good_len = avg(good_metrics, "length_early_mean")
        bad_len = avg(bad_metrics, "length_early_mean")
        print(f"\n  Early episode length (mean, steps < {EARLY_STEP_MAX}):")
        print(f"    Good: {good_len:.1f}" if good_len else "    Good: (no data)")
        print(f"    Bad:  {bad_len:.1f}" if bad_len else "    Bad:  (no data)")
        if good_len and bad_len:
            print(f"    → Bad seeds have {'longer' if bad_len > good_len else 'shorter'} early episodes.")

        # Early vs late loss
        print(f"\n  Training loss (mean):")
        good_early = avg(good_metrics, "loss_early_mean")
        good_late = avg(good_metrics, "loss_late_mean")
        bad_early = avg(bad_metrics, "loss_early_mean")
        bad_late = avg(bad_metrics, "loss_late_mean")
        print(f"    Good  early (1k–30k steps): {good_early:.2f}" if good_early else "    Good  early: (no data)")
        print(f"    Good  late  (170k+ steps):  {good_late:.2f}" if good_late else "    Good  late: (no data)")
        print(f"    Bad   early (1k–30k steps): {bad_early:.2f}" if bad_early else "    Bad   early: (no data)")
        print(f"    Bad   late  (170k+ steps):  {bad_late:.2f}" if bad_late else "    Bad   late: (no data)")
        if good_early and bad_early:
            print(f"    → Early loss: good vs bad diff = {good_early - bad_early:+.2f}")
        if good_late and bad_late:
            print(f"    → Late loss:  good vs bad diff = {good_late - bad_late:+.2f}")

        # When did eval performance diverge?
        good_curve = avg_curve(good_metrics)
        bad_curve = avg_curve(bad_metrics)
        steps = sorted(set(good_curve) | set(bad_curve))
        if steps:
            print(f"\n  Eval reward over time (when did good vs bad diverge?):")
            print(f"    {'Step':>8} {'Good(avg)':>10} {'Bad(avg)':>10} {'Diff':>8}")
            for step in steps[:25]:  # first 25 eval points
                g = good_curve.get(step)
                b = bad_curve.get(step)
                if g is not None and b is not None:
                    diff = g - b
                    print(f"    {step:>8} {g:>10.2f} {b:>10.2f} {diff:>+8.2f}")
            # First step where good clearly beats bad (good - bad > 30)
            for step in steps:
                g, b = good_curve.get(step), bad_curve.get(step)
                if g is not None and b is not None and (g - b) > 30:
                    print(f"\n    → First step where GOOD clearly ahead (good−bad > 30): step {step}, good={g:.2f}, bad={b:.2f}, diff={g-b:+.2f}")
                    break
            else:
                # First step where bad clearly beats good (bad - good > 30)
                for step in steps:
                    g, b = good_curve.get(step), bad_curve.get(step)
                    if g is not None and b is not None and (b - g) > 30:
                        print(f"\n    → Early on BAD was ahead at step {step}; good overtook later (see curve).")
                        break
        print()

    print("=" * 80)
    print("In TensorBoard (tensorboard --logdir runs/): compare manually:")
    print("  - Loss/train: good vs bad seeds side-by-side")
    print("  - Reward/eval: when curves separate")
    print("  - Length/train: early episode lengths")
    print("  - Add Q-value logging in train_rmer.py to check initial Q-values")
    print("=" * 80)


if __name__ == "__main__":
    main()
