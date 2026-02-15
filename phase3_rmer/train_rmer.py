# train rmer (remer buffer with htd, lfiw, tce)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from phase3_rmer.rmer_agent import RMERAgent
from config import RMER_CONFIG, ENV_CONFIG


class RMERTrainer:
    # run_suffix for sensitivity runs (sensitivity_xxx_env_seedN)

    def __init__(self, env_name='LunarLander-v2', config=None, seed=42, device='cpu', output_dir=None, run_suffix=None):
        self.env_name = env_name
        self.config = config if config is not None else RMER_CONFIG.copy()
        self.seed = seed
        self.device = device
        self.output_dir = output_dir or ''
        self.run_suffix = run_suffix
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env = gym.make(env_name, render_mode=None)
        self.env.action_space.seed(seed)

        # Get dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Create RMER agent
        self.agent = RMERAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            total_steps=self.config['total_steps'],
            device=device,
            seed=seed
        )

        # Logging
        if run_suffix:
            run_name = f"sensitivity_{run_suffix}_{env_name}_seed{seed}"
        else:
            run_name = f"rmer_{env_name}_seed{seed}"
            if not self.config.get('use_hindsight_td', True):
                run_name += "_noHTD"
            if not self.config.get('use_lfiw', True):
                run_name += "_noLFIW"
            if not self.config.get('use_tce', True):
                run_name += "_noTCE"

        log_dir = os.path.join(self.output_dir, 'runs', run_name) if self.output_dir else f'runs/{run_name}'
        os.makedirs(os.path.dirname(log_dir) if self.output_dir else 'runs', exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.losses = []
        self.priority_history = []

    def train(self, total_steps=None):
        """
        Train the agent.

        Args:
            total_steps: Number of environment steps to train for
        """
        if total_steps is None:
            total_steps = self.config['total_steps']
        self.config['total_steps'] = total_steps
        self.agent.set_total_steps(total_steps)

        state, _ = self.env.reset(seed=self.seed)
        self.agent.start_episode()

        episode_reward = 0
        episode_length = 0
        episode_num = 0

        pbar = tqdm(total=total_steps, desc='Training RMER')

        for step in range(total_steps):
            # Decay epsilon by env step (so epsilon_decay_steps is in env steps)
            self.agent.dqn_agent.decay_epsilon_by_env_step(step)
            # Select action
            action = self.agent.select_action(state)

            # Take step in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.agent.push(state, action, reward, next_state, done)

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1

            # Train agent
            if (step >= self.config['learning_starts'] and
                step % self.config['train_freq'] == 0):

                stats = self.agent.update(env_step=step)

                if stats is not None:
                    self.losses.append(stats['loss'])

                    if step % self.config['log_freq'] == 0:
                        self.writer.add_scalar('Loss/train', stats['loss'], step)
                        self.writer.add_scalar('Epsilon', stats['epsilon'], step)

                        if 'lfiw_loss' in stats:
                            self.writer.add_scalar('LFIW/loss', stats['lfiw_loss'], step)
                            self.writer.add_scalar('LFIW/accuracy', stats['lfiw_accuracy'], step)

            # Episode end
            if done:
                self.agent.end_episode()

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                self.writer.add_scalar('Reward/train', episode_reward, step)
                self.writer.add_scalar('Length/train', episode_length, step)

                # Reset
                state, _ = self.env.reset()
                self.agent.start_episode()
                episode_reward = 0
                episode_length = 0
                episode_num += 1

                pbar.set_postfix({
                    'episode': episode_num,
                    'reward': self.episode_rewards[-1],
                    'epsilon': f'{self.agent.dqn_agent.epsilon:.3f}'
                })

            # Log priority statistics
            if step % (self.config['log_freq'] * 5) == 0 and step > 0:
                priority_stats = self.agent.get_priority_stats()
                self.priority_history.append((step, priority_stats))

                self.writer.add_scalar('Priorities/mean', priority_stats['mean'], step)
                self.writer.add_scalar('Priorities/std', priority_stats['std'], step)
                self.writer.add_scalar('Priorities/max', priority_stats['max'], step)

                # Log component contributions
                for component, comp_stats in priority_stats['component_stats'].items():
                    self.writer.add_scalar(
                        f'Components/{component}_mean',
                        comp_stats['mean'],
                        step
                    )

            # Evaluation
            if step % self.config['eval_freq'] == 0 and step > 0:
                eval_reward = self.evaluate(self.config['eval_episodes'])
                self.eval_rewards.append((step, eval_reward))
                self.writer.add_scalar('Reward/eval', eval_reward, step)

                print(f"\nStep {step}: Eval reward = {eval_reward:.2f}")

            # Save checkpoint
            if step % self.config['save_freq'] == 0 and step > 0:
                ckpt_path = os.path.join(self.output_dir, 'checkpoints', f'rmer_step{step}.pt') if self.output_dir else f'checkpoints/rmer_step{step}.pt'
                self.save_checkpoint(ckpt_path)

            pbar.update(1)

        pbar.close()
        self.env.close()
        self.writer.close()

    def evaluate(self, num_episodes=100):
        """
        Evaluate agent performance.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Average episode reward
        """
        eval_env = gym.make(self.env_name, render_mode=None)
        episode_rewards = []

        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state, eval_mode=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)

        eval_env.close()
        return np.mean(episode_rewards)

    def save_checkpoint(self, filepath):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.agent.save(filepath)

    def plot_results(self, save_path=None):
        """Plot training results with priority analysis."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'plots', 'rmer_results.png') if self.output_dir else 'plots/rmer_results.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        # Training rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) > 100:
            smoothed = np.convolve(
                self.episode_rewards,
                np.ones(100) / 100,
                mode='valid'
            )
            axes[0, 0].plot(smoothed, linewidth=2, label='Smoothed')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Evaluation rewards
        if self.eval_rewards:
            steps, rewards = zip(*self.eval_rewards)
            axes[0, 1].plot(steps, rewards, marker='o', linewidth=2)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].set_title('Evaluation Performance')
            axes[0, 1].grid(True)

        # Training loss
        if self.losses:
            axes[1, 0].plot(self.losses, alpha=0.3)
            if len(self.losses) > 100:
                smoothed = np.convolve(
                    self.losses,
                    np.ones(100) / 100,
                    mode='valid'
                )
                axes[1, 0].plot(smoothed, linewidth=2)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].grid(True)

        # Priority statistics
        if self.priority_history:
            steps, stats_list = zip(*self.priority_history)
            means = [s['mean'] for s in stats_list]
            stds = [s['std'] for s in stats_list]

            axes[1, 1].plot(steps, means, label='Mean', linewidth=2)
            axes[1, 1].fill_between(
                steps,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.3
            )
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Priority')
            axes[1, 1].set_title('Priority Distribution Over Training')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        # Component contributions
        if self.priority_history:
            components = ['hindsight_td', 'lfiw_weights', 'tce_scores']
            for comp in components:
                values = [s['component_stats'].get(comp, {}).get('mean', 0) for s in stats_list]
                axes[2, 0].plot(steps, values, label=comp.replace('_', ' ').title(), linewidth=2)

            axes[2, 0].set_xlabel('Step')
            axes[2, 0].set_ylabel('Component Value')
            axes[2, 0].set_title('RMER Component Contributions')
            axes[2, 0].legend()
            axes[2, 0].grid(True)

        # Episode lengths
        axes[2, 1].plot(self.episode_lengths, alpha=0.3)
        if len(self.episode_lengths) > 100:
            smoothed = np.convolve(
                self.episode_lengths,
                np.ones(100) / 100,
                mode='valid'
            )
            axes[2, 1].plot(smoothed, linewidth=2)
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Length')
        axes[2, 1].set_title('Episode Lengths')
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Results saved to {save_path}")


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train RMER agent')
    parser.add_argument('--env', type=str, default='LunarLander-v2',
                        help='Environment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Total training steps')
    parser.add_argument('--no-htd', action='store_true',
                        help='Disable hindsight TD')
    parser.add_argument('--no-lfiw', action='store_true',
                        help='Disable LFIW')
    parser.add_argument('--no-tce', action='store_true',
                        help='Disable TCE')

    args = parser.parse_args()

    # Modify config based on args
    config = RMER_CONFIG.copy()
    if args.no_htd:
        config['use_hindsight_td'] = False
    if args.no_lfiw:
        config['use_lfiw'] = False
    if args.no_tce:
        config['use_tce'] = False

    # Create trainer
    trainer = RMERTrainer(
        env_name=args.env,
        config=config,
        seed=args.seed,
        device=args.device
    )

    # Train
    trainer.train(total_steps=args.steps)

    # Plot results
    trainer.plot_results()

    # Final evaluation
    final_reward = trainer.evaluate(num_episodes=100)
    print(f"\nFinal evaluation reward: {final_reward:.2f}")


if __name__ == '__main__':
    main()
