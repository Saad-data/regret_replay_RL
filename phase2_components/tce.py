# tce = q-accuracy from distance to terminal. accuracy = exp(-tce), clip with b1/b2 over time

import numpy as np
from collections import deque


class TCEComputer:
    # tce = f(h)*L + c + gamma^(h+1)*c, clip b1..b2

    def __init__(self, config, total_steps):
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self.gamma = config['gamma']
        self.c = config['c_max_suboptimality']
        self.b1_start = config['b1_start']
        self.b1_end = config['b1_end']
        self.b2_start = config['b2_start']
        self.b2_end = config['b2_end']
        self.bellman_error_window = config['moving_avg_window']
        self.bellman_error_momentum = config['moving_avg_momentum']
        self.bellman_errors = deque(maxlen=self.bellman_error_window)
        self.moving_avg_bellman_error = 0.0

    def update_bellman_error(self, td_errors):
        mean_td = np.mean(td_errors)
        if len(self.bellman_errors) == 0:
            self.moving_avg_bellman_error = mean_td
        else:
            self.moving_avg_bellman_error = (
                self.bellman_error_momentum * self.moving_avg_bellman_error +
                (1 - self.bellman_error_momentum) * mean_td
            )

        self.bellman_errors.append(mean_td)

    def get_adaptive_bounds(self):
        progress = min(1.0, self.current_step / self.total_steps)
        b1 = self.b1_start + (self.b1_end - self.b1_start) * progress
        b2 = self.b2_start - (self.b2_start - self.b2_end) * progress

        return b1, b2

    def compute_discounted_distance(self, h):
        if self.gamma == 1.0:
            return h

        f_h = (self.gamma - self.gamma ** (h + 1)) / (1 - self.gamma)
        return f_h

    def compute_tce(self, distances_to_terminal):
        h = np.array(distances_to_terminal)
        f_h = np.array([self.compute_discounted_distance(hi) for hi in h])
        L = self.moving_avg_bellman_error
        tce = f_h * L + self.c + (self.gamma ** (h + 1)) * self.c
        b1, b2 = self.get_adaptive_bounds()
        tce = np.clip(tce, b1, b2)

        return tce

    def compute_accuracy_scores(self, distances_to_terminal):
        tce = self.compute_tce(distances_to_terminal)
        accuracy = np.exp(-tce)
        return accuracy

    def step(self, env_step=None):
        # use env_step from trainer so progress is in env steps
        if env_step is not None:
            self.current_step = env_step
        else:
            self.current_step += 1


class DistanceTracker:
    # trans_id -> steps to end of episode (for tce)

    def __init__(self):
        self.episode_transitions = []
        self.distances = {}
        self.transition_counter = 0

    def start_episode(self):
        self.episode_transitions = []

    def add_transition(self, transition_id):
        self.episode_transitions.append(transition_id)

    def end_episode(self):
        episode_length = len(self.episode_transitions)

        for i, trans_id in enumerate(self.episode_transitions):
            distance = episode_length - i - 1
            self.distances[trans_id] = distance

        self.episode_transitions = []

    def get_distance(self, transition_id):
        return self.distances.get(transition_id, 0)

    def get_distances(self, transition_ids):
        return np.array([self.get_distance(tid) for tid in transition_ids])


def test_tce():
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import TCE_CONFIG

    print("Testing TCE Computer...")

    # Create TCE computer
    tce = TCEComputer(TCE_CONFIG, total_steps=100000)

    # Update Bellman error
    td_errors = np.random.uniform(0, 1, 32)
    tce.update_bellman_error(td_errors)
    print(f"Moving avg Bellman error: {tce.moving_avg_bellman_error:.4f}")

    # Compute TCE for different distances
    distances = np.array([0, 1, 5, 10, 50, 100])
    tce_values = tce.compute_tce(distances)
    accuracy_scores = tce.compute_accuracy_scores(distances)

    print("\nDistance to terminal | TCE | Accuracy")
    print("-" * 45)
    for d, t, a in zip(distances, tce_values, accuracy_scores):
        print(f"{d:18d} | {t:.4f} | {a:.4f}")

    # Test adaptive bounds
    print("\nAdaptive bounds over training:")
    print("Progress | b1 | b2")
    print("-" * 30)
    for step in [0, 25000, 50000, 75000, 100000]:
        tce.current_step = step
        b1, b2 = tce.get_adaptive_bounds()
        progress = step / 100000
        print(f"{progress:8.2f} | {b1:.2f} | {b2:.2f}")

    # Test distance tracker
    print("\nTesting Distance Tracker...")
    tracker = DistanceTracker()

    # Simulate episode
    tracker.start_episode()
    episode_ids = []
    for i in range(10):
        trans_id = tracker.transition_counter
        tracker.add_transition(trans_id)
        episode_ids.append(trans_id)
        tracker.transition_counter += 1

    tracker.end_episode()

    # Get distances
    distances = tracker.get_distances(episode_ids)
    print(f"Episode length: {len(episode_ids)}")
    print(f"Distances: {distances}")
    print(f"Expected: {list(range(len(episode_ids)-1, -1, -1))}")

    print("\nTCE test passed!")


if __name__ == '__main__':
    test_tce()
