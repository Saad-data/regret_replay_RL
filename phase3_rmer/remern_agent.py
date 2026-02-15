# remern agent = dqn + remern buffer (error net)

import torch
import numpy as np

from phase1_baseline.dqn import DQNAgent
from phase3_rmer.remern_buffer import ReMERNBuffer


class ReMERNAgent:
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
            device=device,
        )

        self.replay_buffer = ReMERNBuffer(
            capacity=config['buffer_size'],
            agent=self.dqn_agent,
            config=config,
            total_steps=total_steps,
            device=device,
            seed=seed,
        )

        self.batch_size = config['batch_size']
        self.train_step = 0

    @property
    def q_network(self):
        return self.dqn_agent.q_network

    @property
    def epsilon(self):
        return self.dqn_agent.epsilon

    def decay_epsilon_by_env_step(self, env_step):
        self.dqn_agent.decay_epsilon_by_env_step(env_step)

    def select_action(self, state, eval_mode=False):
        return self.dqn_agent.select_action(state, eval_mode=eval_mode)

    def push(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def start_episode(self):
        self.replay_buffer.start_episode()

    def end_episode(self):
        self.replay_buffer.end_episode()

    def update_batch(self, states, actions, rewards, next_states, dones, indices, weights):
        """
        One training step with given batch (used by trainer-driven loop).
        Returns (loss, htd_errors_numpy) for priority updates.
        """
        loss, td_errors = self.dqn_agent.update(
            (states, actions, rewards, next_states, dones, indices, weights)
        )

        htd = self.replay_buffer.hindsight_td.compute_hindsight_td_error(
            states, actions, rewards, next_states, dones
        )
        htd_np = htd if isinstance(htd, np.ndarray) else htd.detach().cpu().numpy()

        self.train_step += 1
        return loss, htd_np
