# baseline dqn trainer - uniform replay

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from phase1_baseline.dqn import DQNAgent
from phase1_baseline.replay_buffer import ReplayBuffer
from config import BASELINE_CONFIG, ENV_CONFIG


class BaselineTrainer:
    # uniform replay only

    def __init__(self, env_name='CartPole-v1', config=None, seed=42, device='cpu', render=False):
        self.env_name = env_name
        self.config = config if config is not None else BASELINE_CONFIG.copy()
        self.seed = seed
        self.device = device
        self.render = render
        np.random.seed(seed)
        torch.manual_seed(seed)
        render_mode = 'human' if render else None
        self.env = gym.make(env_name, render_mode=render_mode)
        self.env.action_space.seed(seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.agent = DQNAgent(
            self.state_dim,
            self.action_dim,
            self.config,
            device=device
        )
        self.replay_buffer = ReplayBuffer(
            capacity=self.config['buffer_size'],
            seed=seed
        )
        self.writer = SummaryWriter(f'runs/baseline_{env_name}_seed{seed}')
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.losses = []

    def train(self, total_steps=None):
        if total_steps is None:
            total_steps = self.config['total_steps']

        state, _ = self.env.reset(seed=self.seed)
        episode_reward = 0
        episode_length = 0
        episode_num = 0

        pbar = tqdm(total=total_steps, desc='Training Baseline DQN')

        for step in range(total_steps):
            self.agent.decay_epsilon_by_env_step(step)
            action = self.agent.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_length += 1
            if (step >= self.config['learning_starts'] and
                step % self.config['train_freq'] == 0 and
                len(self.replay_buffer) >= self.config['batch_size']):

                batch = self.replay_buffer.sample(self.config['batch_size'])
                loss, _ = self.agent.update(batch)
                self.losses.append(loss)
                # print("step", step, "loss", loss)  # debug
                if step % self.config['log_freq'] == 0:
                    self.writer.add_scalar('Loss/train', loss, step)
                    self.writer.add_scalar('Epsilon', self.agent.epsilon, step)
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                self.writer.add_scalar('Reward/train', episode_reward, step)
                self.writer.add_scalar('Length/train', episode_length, step)
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_num += 1

                pbar.set_postfix({
                    'episode': episode_num,
                    'reward': self.episode_rewards[-1],
                    'epsilon': f'{self.agent.epsilon:.3f}'
                })
            if step % self.config['eval_freq'] == 0 and step > 0:
                eval_reward = self.evaluate(self.config['eval_episodes'])
                self.eval_rewards.append((step, eval_reward))
                self.writer.add_scalar('Reward/eval', eval_reward, step)

                print(f"\nStep {step}: Eval reward = {eval_reward:.2f}")
            if step % self.config['save_freq'] == 0 and step > 0:
                self.save_checkpoint(f'checkpoints/baseline_step{step}.pt')

            pbar.update(1)

        pbar.close()
        self.env.close()
        self.writer.close()

    def evaluate(self, num_episodes=100):
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.agent.save(filepath)

    def plot_results(self, save_path='plots/baseline_results.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(self.episode_rewards, alpha=0.3)
        if len(self.episode_rewards) > 100:
            smoothed = np.convolve(
                self.episode_rewards,
                np.ones(100) / 100,
                mode='valid'
            )
            axes[0, 0].plot(smoothed, linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].grid(True)
        if self.eval_rewards:
            steps, rewards = zip(*self.eval_rewards)
            axes[0, 1].plot(steps, rewards, marker='o')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].set_title('Evaluation Performance')
            axes[0, 1].grid(True)
        axes[1, 0].plot(self.episode_lengths, alpha=0.3)
        if len(self.episode_lengths) > 100:
            smoothed = np.convolve(
                self.episode_lengths,
                np.ones(100) / 100,
                mode='valid'
            )
            axes[1, 0].plot(smoothed, linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Length')
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].grid(True)
        if self.losses:
            axes[1, 1].plot(self.losses, alpha=0.3)
            if len(self.losses) > 100:
                smoothed = np.convolve(
                    self.losses,
                    np.ones(100) / 100,
                    mode='valid'
                )
                axes[1, 1].plot(smoothed, linewidth=2)
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Results saved to {save_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train baseline DQN')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Environment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Total training steps')
    parser.add_argument('--render', action='store_true',
                        help='Show environment window during training')

    args = parser.parse_args()
    trainer = BaselineTrainer(
        env_name=args.env,
        seed=args.seed,
        device=args.device,
        render=args.render
    )
    trainer.train(total_steps=args.steps)
    trainer.plot_results()
    final_reward = trainer.evaluate(num_episodes=100)
    print(f"\nFinal evaluation reward: {final_reward:.2f}")


if __name__ == '__main__':
    main()
