# PER buffer (Schaul et al) - used by train_per

import numpy as np
import torch


class PERBuffer:
    # P(i) ~ priority^alpha, importance weights (N*P)^(-beta)

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self._size = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self._size > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        if self._size < batch_size:
            raise ValueError(f"Not enough samples: {self._size} < {batch_size}")
        priorities = self.priorities[:self._size]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(self._size, batch_size, replace=False, p=probs)
        weights = (self._size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)),
            indices,
            torch.FloatTensor(weights)
        )

    def update_priorities(self, indices, td_errors):
        priorities = np.abs(td_errors) + 1e-6  # TODO: epsilon could come from config
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def update_beta(self):
        self.frame += 1
        fraction = min(self.frame / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    def __len__(self):
        return self._size
