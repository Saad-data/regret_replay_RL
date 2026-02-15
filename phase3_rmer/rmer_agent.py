# rmer agent = dqn + rmer buffer

import torch
import numpy as np

from phase1_baseline.dqn import DQNAgent
from phase3_rmer.rmer_buffer import RMERBuffer


class RMERAgent:
    # just dqn + rmer buffer, same api as baseline

    def __init__(self, state_dim, action_dim, config, total_steps, device='cpu', seed=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.seed = seed
        self.dqn_agent = DQNAgent(
            state_dim,
            action_dim,
            config,
            device=device
        )
        self.replay_buffer = RMERBuffer(
            capacity=config['buffer_size'],
            agent=self.dqn_agent,
            config=config,
            total_steps=total_steps,
            device=device,
            seed=seed
        )
        self.batch_size = config['batch_size']
        self.train_step = 0

    def set_total_steps(self, total_steps):
        self.replay_buffer.total_steps = total_steps
        self.replay_buffer.tce_computer.total_steps = total_steps

    def select_action(self, state, eval_mode=False):
        return self.dqn_agent.select_action(state, eval_mode=eval_mode)

    def push(self, state, action, reward, next_state, done):
        """
        Store transition in RMER buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def start_episode(self):
        """Start a new episode (for distance tracking)."""
        self.replay_buffer.start_episode()

    def end_episode(self):
        """End current episode (compute distances)."""
        self.replay_buffer.end_episode()

    def update(self, env_step=None):
        """
        Perform one training update with RMER prioritization.

        Args:
            env_step: Current env step (for TCE progress; use so progress = env_step / total_steps).

        Returns:
            Dictionary with training statistics
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch according to RMER priorities
        states, actions, rewards, next_states, dones, indices = \
            self.replay_buffer.sample(self.batch_size)

        # Perform DQN update
        loss, _ = self.dqn_agent.update((states, actions, rewards, next_states, dones))

        # Update priorities based on new Q-values
        self.replay_buffer.update_priorities(
            indices, states, actions, rewards, next_states, dones
        )

        # Update LFIW classifier
        lfiw_loss, lfiw_acc = self.replay_buffer.update_lfiw()

        # Step time-adaptive components (pass env_step so TCE uses env steps)
        self.replay_buffer.step(env_step)

        self.train_step += 1

        # Collect statistics
        stats = {
            'loss': loss,
            'epsilon': self.dqn_agent.epsilon,
            'train_step': self.train_step,
        }

        if lfiw_loss is not None:
            stats['lfiw_loss'] = lfiw_loss
            stats['lfiw_accuracy'] = lfiw_acc

        return stats

    def get_priority_stats(self):
        """Get statistics about priorities."""
        return self.replay_buffer.get_priority_stats()

    def save(self, filepath):
        """
        Save agent state.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'dqn_state': {
                'q_network': self.dqn_agent.q_network.state_dict(),
                'target_network': self.dqn_agent.target_network.state_dict(),
                'optimizer': self.dqn_agent.optimizer.state_dict(),
                'train_step': self.dqn_agent.train_step,
                'epsilon': self.dqn_agent.epsilon,
            },
            'lfiw_state': {
                'classifier': self.replay_buffer.lfiw_estimator.classifier.state_dict(),
                'optimizer': self.replay_buffer.lfiw_estimator.optimizer.state_dict(),
            },
            'train_step': self.train_step,
        }

        torch.save(checkpoint, filepath)

    def load(self, filepath):
        """
        Load agent state.

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load DQN state
        self.dqn_agent.q_network.load_state_dict(checkpoint['dqn_state']['q_network'])
        self.dqn_agent.target_network.load_state_dict(checkpoint['dqn_state']['target_network'])
        self.dqn_agent.optimizer.load_state_dict(checkpoint['dqn_state']['optimizer'])
        self.dqn_agent.train_step = checkpoint['dqn_state']['train_step']
        self.dqn_agent.epsilon = checkpoint['dqn_state']['epsilon']

        # Load LFIW state
        self.replay_buffer.lfiw_estimator.classifier.load_state_dict(
            checkpoint['lfiw_state']['classifier']
        )
        self.replay_buffer.lfiw_estimator.optimizer.load_state_dict(
            checkpoint['lfiw_state']['optimizer']
        )

        self.train_step = checkpoint['train_step']
