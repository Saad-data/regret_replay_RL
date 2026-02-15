"""Quick analysis of pilot results (3 seeds × 500k steps)."""

from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np

EVAL_TAGS = ['Reward/eval', 'eval/reward', 'eval/final_reward', 'Eval/Reward']


def analyze_run(run_name):
    """Extract key metrics from a run."""
    run_path = os.path.join("runs", run_name)
    if not os.path.exists(run_path):
        return None

    try:
        ea = event_accumulator.EventAccumulator(run_path)
        ea.Reload()

        events = None
        for tag in EVAL_TAGS:
            if tag in ea.Tags().get('scalars', []):
                events = ea.Scalars(tag)
                break
        if not events:
            return None

        steps = [r.step for r in events]
        values = [r.value for r in events]

        # Find when reaches 450 (good performance)
        plateau_step = next((s for s, v in zip(steps, values) if v >= 450), None)

        # Final performance
        final_reward = values[-1] if values else 0

        # Check if still improving in last 10%
        last_10_pct = values[-max(1, len(values) // 10) :]
        if len(last_10_pct) > 1:
            still_improving = max(last_10_pct) - min(last_10_pct) > 10
        else:
            still_improving = False

        return {
            'plateau_step': plateau_step,
            'final_reward': final_reward,
            'still_improving': still_improving,
            'steps': steps,
            'values': values,
        }
    except Exception as e:
        print(f"  Error analyzing {run_name}: {e}")
        return None


def main():
    # Pilot run name patterns (CartPole-v1, seeds 0, 1, 2)
    methods = {
        'Uniform': 'baseline_CartPole-v1_seed{}',
        'PER': 'per_CartPole-v1_seed{}',
        'TD-only': 'rmer_CartPole-v1_seed{}_noLFIW_noTCE',
        'Full-RMER': 'rmer_CartPole-v1_seed{}',
    }
    pilot_seeds = [0, 1, 2]

    print("=" * 80)
    print("PILOT STUDY ANALYSIS (Seeds 0, 1, 2)")
    print("=" * 80)
    print()

    all_plateau_steps = []

    for method_name, pattern in methods.items():
        print(f"{method_name}:")

        plateau_steps = []
        final_rewards = []
        improving_count = 0

        for seed in pilot_seeds:
            run_name = pattern.format(seed)
            result = analyze_run(run_name)

            if result:
                plateau = result['plateau_step']
                final = result['final_reward']
                improving = result['still_improving']

                if plateau is not None:
                    plateau_steps.append(plateau)
                final_rewards.append(final)
                if improving:
                    improving_count += 1

                plateau_str = f"{plateau:>8}" if plateau is not None else "   never"
                status = "📈 improving" if improving else "📊 plateaued"
                print(f"  Seed {seed}: Plateau at {plateau_str}, Final: {final:>6.1f}, {status}")
            else:
                print(f"  Seed {seed}: (run not found or no eval data)")

        if plateau_steps:
            avg_plateau = np.mean(plateau_steps)
            all_plateau_steps.extend(plateau_steps)
            print(f"  → Avg plateau: {avg_plateau:>8.0f} steps")
        else:
            print(f"  → Never reached 450 reward")

        if final_rewards:
            print(f"  → Final reward: {np.mean(final_rewards):>6.1f} ± {np.std(final_rewards):>5.1f}")

        print(f"  → Still improving: {improving_count}/3 seeds")
        print()

    # Overall recommendation
    print("=" * 80)
    print("RECOMMENDATION FOR FULL RUN:")
    print("=" * 80)

    if all_plateau_steps:
        overall_avg = np.mean(all_plateau_steps)

        if overall_avg < 50000:
            print("✅ All methods plateau before 50k steps")
            print("   → RECOMMEND: Use 50,000 steps for full run")
            print("   → This will save ~7 hours!")
            print()
            print("   To use 50k steps, in Phase 4 change:")
            print("   STEPS=50000  # instead of 500000")

        elif overall_avg < 100000:
            print("✅ Most methods plateau before 100k steps")
            print("   → RECOMMEND: Use 100,000 steps for full run")
            print("   → This will save ~5 hours!")
            print()
            print("   To use 100k steps, in Phase 4 change:")
            print("   STEPS=100000  # instead of 500000")

        elif overall_avg < 200000:
            print("⚠️  Methods plateau around 100-200k steps")
            print("   → RECOMMEND: Use 200,000 steps for full run")
            print("   → This will save ~3 hours")
            print()
            print("   To use 200k steps, in Phase 4 change:")
            print("   STEPS=200000  # instead of 500000")

        else:
            print("⚠️  Methods take >200k steps to plateau")
            print("   → RECOMMEND: Continue with 500,000 steps")
            print("   → Full training duration needed")
    else:
        print("⚠️  No methods reached 450 reward in pilot")
        print("   → RECOMMEND: Continue with 500,000 steps")
        print("   → May need full duration to solve task")

    print()
    print("=" * 80)
    print("Next: Review learning curves in TensorBoard")
    print("  tensorboard --logdir runs/ --port 6006")
    print("=" * 80)


if __name__ == "__main__":
    main()
