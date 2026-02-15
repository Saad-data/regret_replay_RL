"""
Compare RMER against baseline methods (Uniform replay, PER).
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime

from phase1_baseline.train_baseline import BaselineTrainer
from phase3_rmer.train_rmer import RMERTrainer
from utils.evaluation import evaluate_agent, compare_agents_statistical, compute_sample_efficiency
from utils.visualization import plot_learning_curves, plot_comparison_table
from config import BASELINE_CONFIG, RMER_CONFIG, ENV_CONFIG


def run_comparison_experiment(env_name='LunarLander-v2',
                              num_seeds=5,
                              total_steps=100000,
                              device='cpu'):
    """
    Compare RMER against baseline methods.

    Methods compared:
    1. Uniform Replay (baseline DQN)
    2. RMER (full)

    Args:
        env_name: Environment name
        num_seeds: Number of seeds per method
        total_steps: Training steps per run
        device: Device to run on
    """
    print("=" * 80)
    print("RMER vs BASELINE COMPARISON")
    print("=" * 80)
    print(f"Environment: {env_name}")
    print(f"Seeds: {num_seeds}")
    print(f"Steps per run: {total_steps}")
    print(f"Device: {device}")
    print("=" * 80)

    # Target reward for sample efficiency (from env config if available)
    target_reward = ENV_CONFIG.get('lunar_lander', {}).get('target_reward', 200)
    if env_name == 'CartPole-v1':
        target_reward = ENV_CONFIG.get('cartpole', {}).get('target_reward', 475)
    elif env_name == 'MountainCar-v0':
        target_reward = ENV_CONFIG.get('mountaincar', {}).get('target_reward', -110)

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/comparison_{env_name}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)

    methods = {
        'Uniform Replay': 'baseline',
        'RMER': 'rmer',
    }

    all_results = {}

    for method_name, method_type in methods.items():
        print(f"\n{'=' * 80}")
        print(f"Running: {method_name}")
        print(f"{'=' * 80}")

        method_results = {
            'seeds': [],
            'eval_results': [],
            'training_data': [],
        }

        for seed_idx, seed in enumerate(range(num_seeds)):
            print(f"\nSeed {seed_idx + 1}/{num_seeds} (seed={seed})")

            if method_type == 'baseline':
                config = BASELINE_CONFIG.copy()
                config['total_steps'] = total_steps

                trainer = BaselineTrainer(
                    env_name=env_name,
                    config=config,
                    seed=seed,
                    device=device
                )

            elif method_type == 'rmer':
                config = RMER_CONFIG.copy()
                config['total_steps'] = total_steps

                trainer = RMERTrainer(
                    env_name=env_name,
                    config=config,
                    seed=seed,
                    device=device
                )

            trainer.train(total_steps=total_steps)

            eval_results = evaluate_agent(
                trainer.agent,
                env_name,
                num_episodes=100,
                seed=seed
            )

            print(f"  Eval reward: {eval_results['mean_reward']:.2f} "
                  f"± {eval_results['std_reward']:.2f}")

            method_results['seeds'].append(seed)
            method_results['eval_results'].append(eval_results)
            method_results['training_data'].append({
                'episode_rewards': trainer.episode_rewards,
                'eval_rewards': trainer.eval_rewards,
                'losses': trainer.losses,
            })

        # Aggregate results
        all_rewards = [r['rewards'] for r in method_results['eval_results']]

        aggregated = {
            'mean_reward': np.mean([r['mean_reward'] for r in method_results['eval_results']]),
            'std_reward': np.std([r['mean_reward'] for r in method_results['eval_results']]),
            'mean_length': np.mean([r['mean_length'] for r in method_results['eval_results']]),
            'rewards': np.concatenate(all_rewards),
            'all_seeds_results': method_results,
        }

        # Sample efficiency
        avg_eval_rewards = []
        for seed_data in method_results['training_data']:
            if seed_data.get('eval_rewards'):
                avg_eval_rewards.append(seed_data['eval_rewards'])

        if avg_eval_rewards and aggregated['mean_length'] > 0:
            efficiency = compute_sample_efficiency(
                avg_eval_rewards[0],
                target_reward,
                aggregated['mean_length']
            )
            aggregated.update(efficiency)

        all_results[method_name] = aggregated

        print(f"\n{method_name} summary:")
        print(f"  Mean reward: {aggregated['mean_reward']:.2f} "
              f"± {aggregated['std_reward']:.2f}")
        if 'steps_to_target' in aggregated:
            print(f"  Steps to target: {aggregated['steps_to_target']:.0f}")

    # Statistical comparison
    print(f"\n{'=' * 80}")
    print("STATISTICAL COMPARISON")
    print("=" * 80)

    comparisons = compare_agents_statistical(all_results)

    for comparison, stats in comparisons.items():
        print(f"\n{comparison}:")
        print(f"  Mean difference: {stats['mean_diff']:.2f}")
        print(f"  p-value: {stats['p_value']:.4f}")
        print(f"  Significant: {'YES' if stats['significant'] else 'NO'}")
        print(f"  Cohen's d: {stats['cohens_d']:.3f}")

        d = abs(stats['cohens_d'])
        if d < 0.2:
            effect = "negligible"
        elif d < 0.5:
            effect = "small"
        elif d < 0.8:
            effect = "medium"
        else:
            effect = "large"
        print(f"  Effect size: {effect}")

    # Save results
    results_file = f'{results_dir}/comparison_results.json'
    with open(results_file, 'w') as f:
        json_results = {}
        for name, data in all_results.items():
            json_results[name] = {
                'mean_reward': float(data['mean_reward']),
                'std_reward': float(data['std_reward']),
                'mean_length': float(data['mean_length']),
                'rewards': data['rewards'].tolist(),
            }
            if 'steps_to_target' in data:
                json_results[name]['steps_to_target'] = float(data['steps_to_target'])
                json_results[name]['episodes_to_target'] = float(data['episodes_to_target'])
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Visualizations
    print("\nGenerating visualizations...")

    learning_curves_data = {}
    for method_name, data in all_results.items():
        seed_data = data['all_seeds_results']['training_data'][0]
        learning_curves_data[method_name] = {
            'episode_rewards': seed_data['episode_rewards'],
            'eval_rewards': seed_data.get('eval_rewards', []),
        }

    plot_learning_curves(
        learning_curves_data,
        env_name,
        save_path=f'{results_dir}/plots/comparison_learning_curves.png'
    )

    metrics = ['mean_reward', 'std_reward', 'mean_length']
    if 'steps_to_target' in all_results.get(list(all_results.keys())[0], {}):
        metrics.extend(['steps_to_target', 'episodes_to_target'])

    plot_comparison_table(
        all_results,
        metrics,
        save_path=f'{results_dir}/plots/comparison_table.png'
    )

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Mean Reward':<20} {'Sample Efficiency':<20}")
    print("-" * 80)

    sorted_methods = sorted(
        all_results.items(),
        key=lambda x: x[1]['mean_reward'],
        reverse=True
    )

    for method_name, data in sorted_methods:
        reward_str = f"{data['mean_reward']:>10.2f} ± {data['std_reward']:>6.2f}"
        if 'steps_to_target' in data:
            efficiency_str = f"{data['steps_to_target']:>8.0f} steps"
        else:
            efficiency_str = "N/A"
        print(f"{method_name:<20} {reward_str}   {efficiency_str}")

    print("=" * 80)

    if 'RMER' in all_results and 'Uniform Replay' in all_results:
        rmer_reward = all_results['RMER']['mean_reward']
        baseline_reward = all_results['Uniform Replay']['mean_reward']
        denom = abs(baseline_reward) if baseline_reward != 0 else 1
        improvement = ((rmer_reward - baseline_reward) / denom) * 100
        print(f"\nRMER improvement over baseline: {improvement:+.2f}%")

    print(f"\nAll results saved to: {results_dir}")

    return all_results


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare RMER with baselines')
    parser.add_argument('--env', type=str, default='LunarLander-v2',
                        help='Environment name')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds per method')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Training steps per run')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')

    args = parser.parse_args()

    results = run_comparison_experiment(
        env_name=args.env,
        num_seeds=args.seeds,
        total_steps=args.steps,
        device=args.device
    )

    print("\nComparison complete!")


if __name__ == '__main__':
    main()
