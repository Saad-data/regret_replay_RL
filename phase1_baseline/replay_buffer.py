# uniform replay buffer (phase1)

import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    # just stores transitions, sample uniform

    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

        if seed is not None:
            np.random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.position = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    # PER for comparison - td error as priority (not used in main pipeline, phase1 has PERBuffer)

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, seed=None):
        super().__init__(capacity, seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        # Use sum tree for efficient sampling (simplified version)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        idx = len(self.buffer)
        super().push(state, action, reward, next_state, done)
        if idx < len(self.priorities):
            self.priorities[idx] = self.max_priority

    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        priorities = self.priorities[:buffer_len]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (buffer_len * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        weights = torch.FloatTensor(weights)

        self.frame += 1

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
