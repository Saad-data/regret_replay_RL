# quick tests for htd, lfiw, tce (run from project root or add path)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt

from phase1_baseline.dqn import DQNAgent
from phase1_baseline.replay_buffer import ReplayBuffer
from phase2_components.hindsight_td import HindsightTDComputer
from phase2_components.lfiw import LFIWEstimator
from phase2_components.tce import TCEComputer, DistanceTracker
from config import BASELINE_CONFIG, HINDSIGHT_TD_CONFIG, LFIW_CONFIG, TCE_CONFIG


def _sample_batch_with_indices(buffer, batch_size):
    indices = np.random.choice(len(buffer), batch_size, replace=False)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for idx in indices:
        s, a, r, ns, d = buffer.buffer[idx]
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
    return (states, actions, rewards, next_states, dones), indices


def test_all_components():
    """Test all Phase 2 components together."""

    print("=" * 60)
    print("Testing all RMER components")
    print("=" * 60)

    # Setup
    state_dim = 4
    action_dim = 2
    batch_size = 32

    # Create agent
    print("\n1. Creating DQN agent...")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=BASELINE_CONFIG,
        device='cpu'
    )
    print("Agent created")

    # Create components
    print("\n2. Creating RMER components...")

    htd = HindsightTDComputer(agent, HINDSIGHT_TD_CONFIG)
    print("Hindsight TD computer created")

    lfiw = LFIWEstimator(state_dim, action_dim, LFIW_CONFIG, device='cpu')
    print("LFIW estimator created")

    tce = TCEComputer(TCE_CONFIG, total_steps=100000)
    distance_tracker = DistanceTracker()
    print("TCE computer and distance tracker created")

    # Create replay buffer with data
    print("\n3. Populating replay buffer...")
    buffer = ReplayBuffer(capacity=10000, seed=42)

    # Simulate episodes and track distances
    num_episodes = 20

    for ep in range(num_episodes):
        distance_tracker.start_episode()
        episode_length = np.random.randint(10, 50)

        for step in range(episode_length):
            state = np.random.randn(state_dim)
            action = np.random.randint(0, action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = 1 if step == episode_length - 1 else 0

            buffer.push(state, action, reward, next_state, done)
            trans_id = len(buffer) - 1
            distance_tracker.add_transition(trans_id)

        distance_tracker.end_episode()

    print(f"Buffer populated with {len(buffer)} transitions from {num_episodes} episodes")

    # Test components on a batch
    print("\n4. Testing components on a batch...")

    # Sample batch with indices so we can look up distances
    batch, batch_trans_ids = _sample_batch_with_indices(buffer, batch_size)
    states, actions, rewards, next_states, dones = batch

    print(f"Sampled batch of size {batch_size}")

    # Test Hindsight TD
    print("\n5. Computing Hindsight TD errors...")
    hindsight_td = htd.compute_hindsight_td_error(
        states, actions, rewards, next_states, dones
    )
    standard_td = htd.compute_standard_td_error(
        states, actions, rewards, next_states, dones
    )

    print(f"   Hindsight TD - mean: {hindsight_td.mean():.4f}, std: {hindsight_td.std():.4f}")
    print(f"   Standard TD - mean: {standard_td.mean():.4f}, std: {standard_td.std():.4f}")
    print(f"   Difference: {np.abs(hindsight_td - standard_td).mean():.4f}")
    print("Hindsight TD computed successfully")

    # Test LFIW
    print("\n6. Training LFIW classifier...")
    for i in range(20):
        loss, acc = lfiw.train_classifier(buffer, batch_size=64)
        if i % 5 == 0:
            print(f"   Iteration {i}: loss={loss:.4f}, accuracy={acc:.4f}")

    on_policy_weights = lfiw.compute_on_policy_weights(states, actions)
    print(f"   On-policy weights - mean: {on_policy_weights.mean():.4f}, std: {on_policy_weights.std():.4f}")
    print(f"   Range: [{on_policy_weights.min():.4f}, {on_policy_weights.max():.4f}]")
    print("LFIW trained and weights computed")

    # Test TCE
    print("\n7. Computing TCE and Q-accuracy...")

    # Update Bellman error
    tce.update_bellman_error(standard_td)

    # Get distances for batch (same indices as our batch)
    distances = distance_tracker.get_distances(batch_trans_ids)

    # Compute TCE and accuracy
    tce_values = tce.compute_tce(distances)
    accuracy_scores = tce.compute_accuracy_scores(distances)

    print(f"   Distances - mean: {distances.mean():.2f}, range: [{distances.min()}, {distances.max()}]")
    print(f"   TCE - mean: {tce_values.mean():.4f}, std: {tce_values.std():.4f}")
    print(f"   Accuracy scores - mean: {accuracy_scores.mean():.4f}, std: {accuracy_scores.std():.4f}")
    print("TCE and accuracy computed successfully")

    # Combine priorities
    print("\n8. Computing combined RMER priorities...")

    # Normalize components
    td_normalized = hindsight_td / (hindsight_td.sum() + 1e-8)
    lfiw_normalized = on_policy_weights / (on_policy_weights.sum() + 1e-8)
    tce_normalized = accuracy_scores / (accuracy_scores.sum() + 1e-8)

    # Combine: priority = on_policy × accuracy × td_error
    priorities = on_policy_weights * accuracy_scores * hindsight_td
    priorities = priorities / (priorities.sum() + 1e-8)

    print(f"   Combined priorities - mean: {priorities.mean():.4f}, std: {priorities.std():.4f}")
    print(f"   Range: [{priorities.min():.6f}, {priorities.max():.6f}]")
    print("Priorities computed successfully")

    # Visualize
    print("\n9. Creating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Hindsight TD
    axes[0, 0].hist(hindsight_td, bins=20, alpha=0.7, label='Hindsight')
    axes[0, 0].hist(standard_td, bins=20, alpha=0.7, label='Standard')
    axes[0, 0].set_xlabel('TD Error')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Hindsight vs Standard TD Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # On-policy weights
    axes[0, 1].hist(on_policy_weights, bins=20, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Weight')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('LFIW On-Policy Weights')
    axes[0, 1].grid(True, alpha=0.3)

    # Q-accuracy scores
    axes[0, 2].scatter(distances, accuracy_scores, alpha=0.6)
    axes[0, 2].set_xlabel('Distance to Terminal')
    axes[0, 2].set_ylabel('Accuracy Score')
    axes[0, 2].set_title('TCE Accuracy vs Distance')
    axes[0, 2].grid(True, alpha=0.3)

    # Component contributions
    axes[1, 0].bar(['Hindsight TD', 'LFIW', 'TCE'],
                   [td_normalized.mean(), lfiw_normalized.mean(), tce_normalized.mean()])
    axes[1, 0].set_ylabel('Normalized Mean')
    axes[1, 0].set_title('Component Contributions')
    axes[1, 0].grid(True, alpha=0.3)

    # Combined priorities
    axes[1, 1].hist(priorities, bins=20, color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Priority')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Combined RMER Priorities')
    axes[1, 1].grid(True, alpha=0.3)

    # Priority composition
    top_k = 10
    top_indices = np.argsort(priorities)[-top_k:]

    axes[1, 2].bar(range(top_k), hindsight_td[top_indices] / (hindsight_td[top_indices].max() + 1e-8),
                   alpha=0.5, label='TD')
    axes[1, 2].bar(range(top_k), on_policy_weights[top_indices] / (on_policy_weights[top_indices].max() + 1e-8),
                   alpha=0.5, label='LFIW')
    axes[1, 2].bar(range(top_k), accuracy_scores[top_indices] / (accuracy_scores[top_indices].max() + 1e-8),
                   alpha=0.5, label='TCE')
    axes[1, 2].set_xlabel('Top Priority Transitions')
    axes[1, 2].set_ylabel('Normalized Component Value')
    axes[1, 2].set_title(f'Top {top_k} Priority Composition')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/component_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualization saved to plots/component_test_results.png")

    print("\n" + "=" * 60)
    print("All component tests passed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    test_all_components()
