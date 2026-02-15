# dqn + PER (for sensitivity we override per_alpha etc via config)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from phase1_baseline.per_buffer import PERBuffer
from phase1_baseline.dqn import DQNAgent
from config import BASELINE_CONFIG, PER_CONFIG


class PERTrainer:
    # merged config so sensitivity sweeps can override per_alpha, per_beta

    def __init__(self, env_name='CartPole-v1', config=None, seed=0, device='cpu', run_suffix=None):
        self.env_name = env_name
        self.run_suffix = run_suffix
        self.config = {**BASELINE_CONFIG, **PER_CONFIG, **(config or {})}
        self.seed = seed
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env = gym.make(env_name, render_mode=None)
        self.env.action_space.seed(seed)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.agent = DQNAgent(state_dim, action_dim, self.config, device=device)
        self.replay_buffer = PERBuffer(
            capacity=self.config['buffer_size'],
            alpha=self.config.get('per_alpha', 0.6),
            beta_start=self.config.get('per_beta_start', 0.4),
            beta_frames=self.config['total_steps']
        )

        run_name = f"per_{env_name}_seed{seed}"
        if run_suffix:
            run_name = f"sensitivity_{run_suffix}_{env_name}_seed{seed}"
        os.makedirs('runs', exist_ok=True)
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.episode_rewards = []
        self.eval_rewards = []

        print(f"Initialized PER Trainer: {run_name}")

    def train(self, total_steps=None):
        if total_steps is None:
            total_steps = self.config['total_steps']

        state, _ = self.env.reset(seed=self.seed)
        episode_reward = 0
        episode_count = 0

        pbar = tqdm(total=total_steps, desc='Training PER')

        for step in range(total_steps):
            self.agent.decay_epsilon_by_env_step(step)
            action = self.agent.select_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward

            self.replay_buffer.add(state, action, reward, next_state, float(done))

            if (step >= self.config['learning_starts'] and
                step % self.config['train_freq'] == 0 and
                len(self.replay_buffer) >= self.config['batch_size']):

                batch = self.replay_buffer.sample(self.config['batch_size'])
                loss, td_errors = self.agent.update(batch)
                states, actions, rewards, next_states, dones, indices, weights = batch
                self.replay_buffer.update_priorities(indices, td_errors)
                self.replay_buffer.update_beta()

                if step % self.config['log_freq'] == 0:
                    self.writer.add_scalar('Loss/train', loss, step)
                    self.writer.add_scalar('Epsilon', self.agent.epsilon, step)
                    self.writer.add_scalar('PER/beta', self.replay_buffer.beta, step)
                # print("beta", self.replay_buffer.beta)  # was checking anneal

            if step % self.config['eval_freq'] == 0 and step > 0:
                eval_reward = self.evaluate(num_episodes=self.config.get('eval_episodes', 10))
                self.eval_rewards.append((step, eval_reward))
                self.writer.add_scalar('Reward/eval', eval_reward, step)
                pbar.set_postfix({'eval': f'{eval_reward:.1f}', 'eps': f'{self.agent.epsilon:.3f}'})

            if done:
                self.episode_rewards.append(episode_reward)
                episode_count += 1
                self.writer.add_scalar('Reward/train', episode_reward, step)
                state, _ = self.env.reset()
                episode_reward = 0
            else:
                state = next_state

            pbar.update(1)

        pbar.close()
        self.env.close()

        final_reward = self.evaluate(num_episodes=10)
        self.writer.add_scalar('Reward/eval_final', final_reward, total_steps)
        self.writer.close()

        print(f"Training complete! Final eval reward: {final_reward:.2f}")
        return final_reward

    def evaluate(self, num_episodes=10):
        eval_env = gym.make(self.env_name, render_mode=None)
        rewards = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.agent.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
            rewards.append(episode_reward)
        eval_env.close()
        return np.mean(rewards)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    config = BASELINE_CONFIG.copy()
    config['total_steps'] = args.steps

    trainer = PERTrainer(args.env, config, seed=args.seed, device=args.device)
    trainer.train(total_steps=args.steps)
