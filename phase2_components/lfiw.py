# lfiw = on-policy weight. classifier: fast buffer (recent) vs slow (old), temp scaling

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class OnPolicyClassifier(nn.Module):
    # s,a -> prob on-policy (recent)

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(OnPolicyClassifier, self).__init__()
        input_dim = state_dim + action_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.action_dim = action_dim
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, states, actions):
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        x = torch.cat([states, actions_onehot], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x).squeeze(1)

        return logits


class LFIWEstimator:
    # fast vs slow buffer, train classifier, then k^(1/T) / mean for weights

    def __init__(self, state_dim, action_dim, config, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.classifier = OnPolicyClassifier(
            state_dim,
            action_dim,
            hidden_dim=config['classifier_hidden_dim']
        ).to(device)

        self.optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=config['classifier_lr']
        )
        self.temperature = config['temperature']
        self.fast_ratio = config['fast_buffer_ratio']
        self.train_step = 0
        self.update_freq = config['classifier_update_freq']
        self.min_samples = config['min_samples_for_training']
        self.classifier_losses = []
        self.classifier_accuracy = []

    def split_buffer_indices(self, buffer_size):
        split_point = int(buffer_size * (1 - self.fast_ratio))
        fast_indices = list(range(split_point, buffer_size))
        slow_indices = list(range(0, split_point))

        return fast_indices, slow_indices

    def train_classifier(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < self.min_samples:
            return 0.0, 0.5
        if hasattr(replay_buffer, 'get_fast_slow_indices'):
            fast_indices, slow_indices = replay_buffer.get_fast_slow_indices(self.fast_ratio)
        else:
            fast_indices, slow_indices = self.split_buffer_indices(len(replay_buffer))

        if len(fast_indices) < batch_size // 2 or len(slow_indices) < batch_size // 2:
            return 0.0, 0.5
        fast_sample_indices = np.random.choice(fast_indices, batch_size // 2, replace=False)
        slow_sample_indices = np.random.choice(slow_indices, batch_size // 2, replace=False)
        fast_states, fast_actions = [], []
        slow_states, slow_actions = [], []

        for idx in fast_sample_indices:
            item = replay_buffer.buffer[idx]
            s, a = item[0], item[1]
            fast_states.append(s)
            fast_actions.append(a)

        for idx in slow_sample_indices:
            item = replay_buffer.buffer[idx]
            s, a = item[0], item[1]
            slow_states.append(s)
            slow_actions.append(a)
        fast_states = torch.FloatTensor(np.array(fast_states)).to(self.device)
        fast_actions = torch.LongTensor(np.array(fast_actions)).to(self.device)
        slow_states = torch.FloatTensor(np.array(slow_states)).to(self.device)
        slow_actions = torch.LongTensor(np.array(slow_actions)).to(self.device)
        fast_labels = torch.ones(batch_size // 2).to(self.device)
        slow_labels = torch.zeros(batch_size // 2).to(self.device)
        fast_logits = self.classifier(fast_states, fast_actions)
        slow_logits = self.classifier(slow_states, slow_actions)
        loss = F.binary_cross_entropy_with_logits(
            fast_logits, fast_labels
        ) + F.binary_cross_entropy_with_logits(
            slow_logits, slow_labels
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            fast_preds = (torch.sigmoid(fast_logits) > 0.5).float()
            slow_preds = (torch.sigmoid(slow_logits) < 0.5).float()
            accuracy = (fast_preds.sum() + slow_preds.sum()) / batch_size

        self.classifier_losses.append(loss.item())
        self.classifier_accuracy.append(accuracy.item())

        return loss.item(), accuracy.item()

    def compute_on_policy_weights(self, states, actions):
        # w = k^(1/T) / mean, k = sigmoid(classifier)
        states = states.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            logits = self.classifier(states, actions)
            probs = torch.sigmoid(logits)
            scaled_probs = torch.pow(probs, 1.0 / self.temperature)
            weights = scaled_probs / (scaled_probs.mean() + 1e-8)

        return weights.cpu().numpy()

    def update_step(self, replay_buffer):
        self.train_step += 1

        if self.train_step % self.update_freq == 0:
            return self.train_classifier(replay_buffer)

        return None, None


def test_lfiw():
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from phase1_baseline.replay_buffer import ReplayBuffer
    from config import LFIW_CONFIG

    print("Testing LFIW Estimator...")

    # Create LFIW estimator
    lfiw = LFIWEstimator(
        state_dim=4,
        action_dim=2,
        config=LFIW_CONFIG,
        device='cpu'
    )

    # Create replay buffer with dummy data
    buffer = ReplayBuffer(capacity=10000, seed=42)

    for i in range(5000):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = 0
        buffer.push(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")

    # Train classifier
    loss, acc = lfiw.train_classifier(buffer, batch_size=64)
    print(f"Initial classifier loss: {loss:.4f}, accuracy: {acc:.4f}")

    # Train for a few iterations
    for _ in range(10):
        loss, acc = lfiw.train_classifier(buffer, batch_size=64)

    print(f"After training loss: {loss:.4f}, accuracy: {acc:.4f}")

    # Compute weights for a batch
    states, actions, _, _, _ = buffer.sample(32)
    weights = lfiw.compute_on_policy_weights(states, actions)

    print(f"Weights shape: {weights.shape}")
    print(f"Weights mean: {weights.mean():.4f}")
    print(f"Weights std: {weights.std():.4f}")
    print(f"Weights min/max: {weights.min():.4f} / {weights.max():.4f}")

    print("LFIW test passed!")


if __name__ == '__main__':
    test_lfiw()
