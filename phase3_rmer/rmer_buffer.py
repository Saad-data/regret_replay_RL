# rmer buffer: priority = htd * lfiw * tce. circular buffer so indices stay put

import numpy as np
import torch

from config import HINDSIGHT_TD_CONFIG, LFIW_CONFIG, TCE_CONFIG


def _lfiw_config_with_override(config):
    # sensitivity sweep can override lfiw temp
    out = {**LFIW_CONFIG}
    if "lfiw_temperature" in config:
        out["temperature"] = config["lfiw_temperature"]
    return out

from phase2_components.hindsight_td import HindsightTDComputer
from phase2_components.lfiw import LFIWEstimator
from phase2_components.tce import TCEComputer, DistanceTracker


class RMERBuffer:
    # priority = lfiw * tce_acc * |htd|, unused bits = 1.0

    def __init__(self, capacity, agent, config, total_steps, device='cpu', seed=None):
        self.capacity = capacity
        self.agent = agent
        self.config = config
        self.device = device
        self.total_steps = total_steps
        self.use_htd = config.get('use_hindsight_td', True)
        self.use_lfiw = config.get('use_lfiw', True)
        self.use_tce = config.get('use_tce', True)
        self.priority_epsilon = config.get('priority_epsilon', 1e-6)
        self.priority_clip_max = config.get('priority_clip_max', 5.0)
        self.priority_temperature = config.get('priority_temperature', 1.0)

        if seed is not None:
            np.random.seed(seed)
        self.buffer = [None] * capacity
        self.trans_ids = np.full(capacity, -1, dtype=np.int64)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self._write_pos = 0
        self._size = 0
        self._next_trans_id = 0

        state_dim = agent.state_dim
        action_dim = agent.action_dim

        self.hindsight_td = HindsightTDComputer(agent, HINDSIGHT_TD_CONFIG)
        lfiw_cfg = _lfiw_config_with_override(config)
        self.lfiw_estimator = LFIWEstimator(
            state_dim, action_dim, lfiw_cfg, device=device
        )
        self.tce_computer = TCEComputer(TCE_CONFIG, total_steps=total_steps)
        self.distance_tracker = DistanceTracker()

    def push(self, state, action, reward, next_state, done):
        trans_id = self._next_trans_id
        self._next_trans_id += 1
        self.buffer[self._write_pos] = (state, action, reward, next_state, done)
        self.trans_ids[self._write_pos] = trans_id
        self.priorities[self._write_pos] = 1.0
        self.distance_tracker.add_transition(trans_id)
        self._write_pos = (self._write_pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def start_episode(self):
        self.distance_tracker.start_episode()

    def end_episode(self):
        self.distance_tracker.end_episode()

    def sample(self, batch_size):
        n = self._size
        if n < batch_size:
            raise ValueError("Buffer too small for batch")

        prio = self.priorities[:n].copy()
        prio = np.maximum(prio, self.priority_epsilon)
        t = self.priority_temperature
        probs = np.power(prio, 1.0 / t)
        probs /= probs.sum()

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        return states, actions, rewards, next_states, dones, indices

    def update_priorities(self, indices, states, actions, rewards, next_states, dones):
        if self.use_htd:
            htd = self.hindsight_td.compute_hindsight_td_error(
                states, actions, rewards, next_states, dones
            )
        else:
            htd = np.ones(len(indices), dtype=np.float32)

        # LFIW weights (or 1.0)
        if self.use_lfiw:
            lfiw_w = self.lfiw_estimator.compute_on_policy_weights(states, actions)
        else:
            lfiw_w = np.ones(len(indices), dtype=np.float32)

        # TCE accuracy (or 1.0) — use transition IDs for distance lookup, not buffer indices
        if self.use_tce:
            trans_ids = self.trans_ids[indices]
            distances = self.distance_tracker.get_distances(trans_ids)
            tce_acc = self.tce_computer.compute_accuracy_scores(distances)
            self.tce_computer.update_bellman_error(htd)
        else:
            tce_acc = np.ones(len(indices), dtype=np.float32)

        raw = lfiw_w * tce_acc * (np.abs(htd) + self.priority_epsilon)

        # Percentile clip and cap
        p95 = np.percentile(raw, 95)
        clip_upper = min(p95 * 2.0, self.priority_clip_max)
        raw = np.clip(raw, self.priority_epsilon, clip_upper)

        for idx, p in zip(indices, raw):
            self.priorities[idx] = float(p)

    def update_lfiw(self):
        """One LFIW update step; returns (loss, accuracy) or (None, None)."""
        if not self.use_lfiw:
            return None, None
        return self.lfiw_estimator.update_step(self)

    def step(self, env_step=None):
        """Advance TCE (and other time-based components). Pass env_step so TCE progress uses env steps."""
        self.tce_computer.step(env_step)

    def get_priority_stats(self):
        n = self._size
        if n == 0:
            return {'mean': 0.0, 'max': 0.0, 'std': 0.0, 'component_stats': {}}
        p = self.priorities[:n]
        return {
            'mean': float(np.mean(p)),
            'max': float(np.max(p)),
            'std': float(np.std(p)),
            'component_stats': {},
        }

    def __len__(self):
        return self._size

    def get_fast_slow_indices(self, fast_ratio=0.2):
        """
        Return (fast_physical_indices, slow_physical_indices) by age:
        fast = newest fast_ratio, slow = oldest (1 - fast_ratio).
        Uses circular buffer order when full so LFIW gets correct age-based split.
        """
        n = self._size
        split = int(n * (1 - fast_ratio))
        if split <= 0 or split >= n:
            return list(range(n)), []
        if n < self.capacity:
            return list(range(split, n)), list(range(0, split))
        # Full: oldest at _write_pos, newest at (_write_pos - 1) % capacity
        fast_phys = [(self._write_pos + i) % self.capacity for i in range(split, n)]
        slow_phys = [(self._write_pos + i) % self.capacity for i in range(0, split)]
        return fast_phys, slow_phys
