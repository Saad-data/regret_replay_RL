# remern buffer: same as rmer but error net instead of tce for accuracy

import numpy as np
import torch

from phase3_rmer.rmer_buffer import RMERBuffer
from phase2_components.error_network import ErrorNetwork


class ReMERNBuffer(RMERBuffer):
    # priority = htd * lfiw * error_accuracy (no tce)

    def __init__(self, capacity, agent, config, total_steps, device='cpu', seed=None):
        config_remern = {**config, 'use_tce': False}
        super().__init__(capacity, agent, config_remern, total_steps, device, seed)

        self.error_network = ErrorNetwork(
            state_dim=agent.state_dim,
            action_dim=agent.action_dim,
            hidden_dims=[64, 64],
            lr=config.get('error_network_lr', 1e-3),
        ).to(device)

        self.error_update_freq = config.get('error_update_freq', 100)

        print("Initialized ReMERN Buffer with Error Network")
        print(f"  Error network update freq: {self.error_update_freq}")

    def update_error_network(self, states, actions, q_values):
        return self.error_network.update(states, actions, q_values)

    def update_priorities(self, indices, hindsight_td_errors, states, actions, q_values):
        """
        Update priorities using ReMERN components.

        Args:
            indices: Buffer indices of sampled transitions
            hindsight_td_errors: HTD errors (numpy or tensor)
            states: States tensor (batch, state_dim)
            actions: Actions tensor (batch,)
            q_values: Q-values from main network (batch, action_dim)
        """
        batch_size = len(indices)

        # 1. Hindsight TD magnitude
        if torch.is_tensor(hindsight_td_errors):
            htd = np.abs(hindsight_td_errors.detach().cpu().numpy())
        else:
            htd = np.abs(hindsight_td_errors)

        # 2. LFIW weights (if enabled)
        if self.use_lfiw:
            lfiw_weights = self.lfiw_estimator.compute_on_policy_weights(states, actions)
        else:
            lfiw_weights = np.ones(batch_size, dtype=np.float32)

        # 3. Error Network accuracy (ReMERN component)
        error_accuracy = self.error_network.get_accuracy(states, actions)

        # Combined priority
        priorities = htd * lfiw_weights * error_accuracy

        # Temperature scaling
        priorities = np.power(priorities + 1e-8, 1.0 / self.priority_temperature)

        # Clipping
        priorities = np.clip(priorities, self.priority_epsilon, self.priority_clip_max)

        # Update buffer priorities
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = float(p)

        return {
            'mean_priority': float(np.mean(priorities)),
            'mean_htd': float(np.mean(htd)),
            'mean_lfiw': float(np.mean(lfiw_weights)),
            'mean_accuracy': float(np.mean(error_accuracy)),
        }
