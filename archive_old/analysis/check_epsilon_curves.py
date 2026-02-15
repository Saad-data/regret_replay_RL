"""
Check Epsilon curves from TensorBoard to see how seeds 0-4 vs 5-9 decayed.
Answers: Did seed 0 decay over ~10k, ~100k, or something else?
"""

import os
from tensorboard.backend.event_processing import event_accumulator

RUNS_DIR = "runs"


def get_scalars(run_path, tag):
    if not os.path.isdir(run_path):
        return []
    ea = event_accumulator.EventAccumulator(run_path)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def analyze_epsilon_curve(points, name):
    """Report when epsilon reached 0.1, 0.05, and over how many steps it decayed."""
    if not points or len(points) < 2:
        print(f"  {name}: No data")
        return
    steps = [p[0] for p in points]
    vals = [p[1] for p in points]
    # When did epsilon first go <= 0.1 and <= 0.06 (effectively 0.05)?
    step_01 = None
    step_05 = None
    for s, v in points:
        if step_01 is None and v <= 0.1:
            step_01 = s
        if step_05 is None and v <= 0.06:
            step_05 = s
    # Approximate "decay duration": step where epsilon went from ~1 to ~0.05
    first_high = next((s for s, v in points if v > 0.9), None)
    last_low = step_05
    decay_span = (last_low - first_high) if (first_high is not None and last_low is not None) else None

    print(f"  {name}:")
    print(f"    Steps logged: {min(steps)} - {max(steps)} (n={len(points)})")
    print(f"    Epsilon at start: {vals[0]:.3f}, at end: {vals[-1]:.3f}")
    print(f"    First step epsilon <= 0.10: {step_01}")
    print(f"    First step epsilon <= 0.06: {step_05}")
    print(f"    Decay span (first high to 0.05): ~{decay_span} steps" if decay_span else "    Decay span: N/A")
    if step_05 is not None:
        if step_05 <= 15000:
            print(f"    → Decay over ~10k steps (old/fast config)")
        elif step_05 <= 55000:
            print(f"    → Decay over ~20-50k steps (mid config)")
        else:
            print(f"    → Decay over 100k+ steps OR never reached 0.05 (update-based 100k)")
    print()


def main():
    print("=" * 70)
    print("EPSILON CURVE CHECK: How did seed 0 vs seed 5 actually decay?")
    print("=" * 70)

    for run_type, pattern, label in [
        ("td_only", "rmer_LunarLander-v2_seed{}_noLFIW_noTCE", "td_only"),
        ("full_rmer", "rmer_LunarLander-v2_seed{}", "full_rmer"),
    ]:
        print("-" * 70)
        print(label)
        print("-" * 70)
        for seed in [0, 5]:
            path = os.path.join(RUNS_DIR, pattern.format(seed))
            pts = get_scalars(path, "Epsilon")
            analyze_epsilon_curve(pts, f"Seed {seed}")

    print("=" * 70)
    print("INTERPRETATION:")
    print("  - Seed 0 reached 0.05 by ~41k steps → decay was over ~40k ENV steps (env-step decay).")
    print("  - Seed 5 stuck at 0.53 → decay was by updates with decay_steps=100k (50k updates at 200k).")
    print("  - So seeds 0-4 and 5-9 used DIFFERENT code/config (env-step vs update-step decay).")
    print("  - Recommendation: Re-run ALL seeds 0-9 with fixed code (env-step decay, 20k) for consistency.")
    print("=" * 70)


if __name__ == "__main__":
    main()
