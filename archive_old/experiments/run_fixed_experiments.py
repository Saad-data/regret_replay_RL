"""
Run full experiments with fixed logic and save all results under fixed/.

- 200k steps per run
- Seeds 0-9 (10 seeds)
- Configs: td_only, full_rmer
- Output: fixed/runs/, fixed/checkpoints/, fixed/plots/, fixed/results/
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


OUTPUT_DIR = 'fixed'
TOTAL_STEPS = 200_000
SEEDS = list(range(10))  # 0-9
CONFIGS = ['td_only', 'full_rmer']


def run_fixed_experiments(env_name='LunarLander-v2', device='cpu'):
    """Run td_only and full_rmer with 200k steps, seeds 0-9; save under fixed/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
    results_dir = os.path.join(OUTPUT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    print("=" * 80)
    print("FIXED LOGIC EXPERIMENTS")
    print("=" * 80)
    print(f"Output dir: {OUTPUT_DIR}/")
    print(f"Environment: {env_name}")
    print(f"Steps: {TOTAL_STEPS}")
    print(f"Seeds: {SEEDS}")
    print(f"Configs: {CONFIGS}")
    print("=" * 80)

    all_results = {}

    for config_name in CONFIGS:
        config_mods = ABLATION_CONFIGS[config_name]
        print(f"\n{'=' * 80}")
        print(f"Config: {config_name}")
        print(f"{'=' * 80}")

        config_results = {
            'seeds': [],
            'eval_results': [],
            'training_data': [],
        }

        for seed_idx, seed in enumerate(SEEDS):
            print(f"\nSeed {seed_idx + 1}/{len(SEEDS)} (seed={seed})")

            config = RMER_CONFIG.copy()
            config.update(config_mods)
            config['total_steps'] = TOTAL_STEPS

            trainer = RMERTrainer(
                env_name=env_name,
                config=config,
                seed=seed,
                device=device,
                output_dir=OUTPUT_DIR,
            )

            trainer.train(total_steps=TOTAL_STEPS)

            eval_results = evaluate_agent(
                trainer.agent,
                env_name,
                num_episodes=100,
                seed=seed,
            )

            print(f"  Eval reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

            config_results['seeds'].append(seed)
            config_results['eval_results'].append(eval_results)
            config_results['training_data'].append({
                'episode_rewards': trainer.episode_rewards,
                'eval_rewards': trainer.eval_rewards,
                'losses': trainer.losses,
            })

        all_rewards = [r['rewards'] for r in config_results['eval_results']]
        aggregated = {
            'mean_reward': np.mean([r['mean_reward'] for r in config_results['eval_results']]),
            'std_reward': np.std([r['mean_reward'] for r in config_results['eval_results']]),
            'rewards': np.concatenate(all_rewards),
            'all_seeds_results': config_results,
        }
        all_results[config_name] = aggregated

        print(f"\n{config_name} summary: mean {aggregated['mean_reward']:.2f} ± {aggregated['std_reward']:.2f}")

    # Statistical comparison
    print(f"\n{'=' * 80}")
    print("STATISTICAL COMPARISON (full_rmer vs td_only)")
    print("=" * 80)
    comparisons = compare_agents_statistical(all_results)
    for comparison, stats in comparisons.items():
        print(f"  {comparison}: mean_diff={stats['mean_diff']:.2f}, p={stats['p_value']:.4f}, significant={stats['significant']}")

    # Save JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, f'fixed_results_{timestamp}.json')
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

    # Plots under fixed/results/plots/
    plot_ablation_results(
        all_results,
        save_path=os.path.join(results_dir, 'plots', 'ablation_study.png'),
    )
    learning_curves_data = {}
    for config_name, data in all_results.items():
        training_data = data['all_seeds_results']['training_data']
        all_episode_rewards = [sd['episode_rewards'] for sd in training_data]
        all_eval_rewards = [sd.get('eval_rewards', []) for sd in training_data]
        learning_curves_data[config_name] = {
            'episode_rewards': all_episode_rewards[0] if all_episode_rewards else [],
            'eval_rewards': all_eval_rewards[0] if all_eval_rewards else [],
        }
    plot_learning_curves(
        learning_curves_data,
        env_name,
        save_path=os.path.join(results_dir, 'plots', 'learning_curves.png'),
    )

    print("=" * 80)
    print(f"All outputs saved under: {OUTPUT_DIR}/")
    print("=" * 80)
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run fixed experiments into fixed/')
    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run_fixed_experiments(env_name=args.env, device=args.device)


if __name__ == '__main__':
    main()
