"""
Run ablation study to evaluate individual component contributions.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime

from phase3_rmer.train_rmer import RMERTrainer
from utils.evaluation import evaluate_agent, compare_agents_statistical
from utils.visualization import plot_ablation_results, plot_learning_curves
from config import RMER_CONFIG, ABLATION_CONFIGS


def run_ablation_study(env_name='LunarLander-v2',
                      num_seeds=5,
                      total_steps=100000,
                      device='cpu'):
    """
    Run complete ablation study.

    Tests the following configurations:
    1. Uniform - no prioritization
    2. TD only - hindsight TD error only
    3. LFIW only - on-policy weights only
    4. TCE only - Q-accuracy only
    5. TD + LFIW
    6. TD + TCE
    7. LFIW + TCE
    8. Full RMER - all three components

    Args:
        env_name: Environment name
        num_seeds: Number of seeds per configuration
        total_steps: Training steps per run
        device: Device to run on
    """
    print("=" * 80)
    print("RMER ABLATION STUDY")
    print("=" * 80)
    print(f"Environment: {env_name}")
    print(f"Seeds: {num_seeds}")
    print(f"Steps per run: {total_steps}")
    print(f"Device: {device}")
    print("=" * 80)

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/ablation_{env_name}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)

    all_results = {}

    # Run each ablation configuration
    for config_name, config_mods in ABLATION_CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"Running configuration: {config_name}")
        print(f"{'=' * 80}")

        config_results = {
            'seeds': [],
            'eval_results': [],
            'training_data': [],
        }

        for seed_idx, seed in enumerate(range(num_seeds)):
            print(f"\nSeed {seed_idx + 1}/{num_seeds} (seed={seed})")

            # Create config for this run
            config = RMER_CONFIG.copy()
            config.update(config_mods)
            config['total_steps'] = total_steps

            # Create and train agent
            trainer = RMERTrainer(
                env_name=env_name,
                config=config,
                seed=seed,
                device=device
            )

            trainer.train(total_steps=total_steps)

            # Evaluate
            eval_results = evaluate_agent(
                trainer.agent,
                env_name,
                num_episodes=100,
                seed=seed
            )

            print(f"  Eval reward: {eval_results['mean_reward']:.2f} "
                  f"± {eval_results['std_reward']:.2f}")

            # Store results
            config_results['seeds'].append(seed)
            config_results['eval_results'].append(eval_results)
            config_results['training_data'].append({
                'episode_rewards': trainer.episode_rewards,
                'eval_rewards': trainer.eval_rewards,
                'losses': trainer.losses,
            })

        # Aggregate results across seeds
        all_rewards = [r['rewards'] for r in config_results['eval_results']]

        aggregated = {
            'mean_reward': np.mean([r['mean_reward'] for r in config_results['eval_results']]),
            'std_reward': np.std([r['mean_reward'] for r in config_results['eval_results']]),
            'rewards': np.concatenate(all_rewards),
            'all_seeds_results': config_results,
        }

        all_results[config_name] = aggregated

        print(f"\n{config_name} summary:")
        print(f"  Mean reward: {aggregated['mean_reward']:.2f} "
              f"± {aggregated['std_reward']:.2f}")

    # Statistical comparisons
    print(f"\n{'=' * 80}")
    print("STATISTICAL COMPARISONS")
    print("=" * 80)

    # compare_agents_statistical expects dict of name -> {rewards, ...}
    comparisons = compare_agents_statistical(all_results)

    for comparison, stats in comparisons.items():
        print(f"\n{comparison}:")
        print(f"  Mean difference: {stats['mean_diff']:.2f}")
        print(f"  p-value: {stats['p_value']:.4f}")
        print(f"  Significant: {stats['significant']}")
        print(f"  Cohen's d: {stats['cohens_d']:.3f}")

    # Save results
    results_file = f'{results_dir}/ablation_results.json'
    with open(results_file, 'w') as f:
        json_results = {}
        for name, data in all_results.items():
            json_results[name] = {
                'mean_reward': float(data['mean_reward']),
                'std_reward': float(data['std_reward']),
                'rewards': data['rewards'].tolist(),
            }
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Ablation bar plot
    plot_ablation_results(
        all_results,
        save_path=f'{results_dir}/plots/ablation_study.png'
    )

    # Learning curves for each configuration
    learning_curves_data = {}
    for config_name, data in all_results.items():
        training_data = data['all_seeds_results']['training_data']
        all_episode_rewards = [sd['episode_rewards'] for sd in training_data]
        all_eval_rewards = [sd['eval_rewards'] for sd in training_data if sd.get('eval_rewards')]

        learning_curves_data[config_name] = {
            'episode_rewards': all_episode_rewards[0] if all_episode_rewards else [],
            'eval_rewards': all_eval_rewards[0] if all_eval_rewards else [],
        }

    plot_learning_curves(
        learning_curves_data,
        env_name,
        save_path=f'{results_dir}/plots/learning_curves.png'
    )

    # Summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Mean Reward':<15} {'Std':<10}")
    print("-" * 80)

    sorted_configs = sorted(
        all_results.items(),
        key=lambda x: x[1]['mean_reward'],
        reverse=True
    )

    for config_name, data in sorted_configs:
        print(f"{config_name:<20} {data['mean_reward']:>10.2f} ± "
              f"{data['std_reward']:>6.2f}")

    print("=" * 80)
    print(f"\nAll results saved to: {results_dir}")

    return all_results


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Run RMER ablation study')
    parser.add_argument('--env', type=str, default='LunarLander-v2',
                        help='Environment name')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds per configuration')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Training steps per run')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')

    args = parser.parse_args()

    results = run_ablation_study(
        env_name=args.env,
        num_seeds=args.seeds,
        total_steps=args.steps,
        device=args.device
    )

    print("\nAblation study complete!")


if __name__ == '__main__':
    main()
