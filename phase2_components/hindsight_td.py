# hindsight td = td error after update (Q_k not Q_{k-1}). preview step then rollback

import torch
import torch.nn.functional as F
import numpy as np


class HindsightTDComputer:
    # |Q_k(s,a) - target|, Q_k = after one grad step. we simulate step then restore params

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.clip_value = config.get('clip_td_error', 100.0)

    def compute_hindsight_td_error(self, states, actions, rewards, next_states, dones):
        device = self.agent.device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        with torch.no_grad():
            next_q_values = self.agent.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.agent.gamma * next_q_values * (1 - dones)
        current_q_values = self.agent.q_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        loss = F.mse_loss(current_q_values, target_q_values)
        original_params = [p.clone().detach() for p in self.agent.q_network.parameters()]
        self.agent.optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in self.agent.q_network.parameters():
                if param.grad is not None:
                    param.data.sub_(self.agent.config['learning_rate'] * param.grad)
            q_k = self.agent.q_network(states).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

            hindsight_td = torch.abs(q_k - target_q_values)
            hindsight_td = torch.clamp(hindsight_td, 0, self.clip_value)
        with torch.no_grad():
            for param, original in zip(self.agent.q_network.parameters(), original_params):
                param.data.copy_(original)

        return hindsight_td.cpu().numpy()

    def compute_standard_td_error(self, states, actions, rewards, next_states, dones):
        # before-update td (for per comparison)
        device = self.agent.device

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        with torch.no_grad():
            next_q_values = self.agent.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.agent.gamma * next_q_values * (1 - dones)

            current_q_values = self.agent.q_network(states).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

            td_error = torch.abs(current_q_values - target_q_values)
            td_error = torch.clamp(td_error, 0, self.clip_value)

        return td_error.cpu().numpy()


class HindsightTDAgentWrapper:
    # wrapper to get hindsight td during train (not used in current pipeline i think)

    def __init__(self, agent, config):
        self.agent = agent
        self.td_computer = HindsightTDComputer(agent, config)
        self.use_hindsight = config.get('use_preview_update', True)

    def update_with_hindsight_td(self, batch):
        states, actions, rewards, next_states, dones = batch

        if self.use_hindsight:
            hindsight_td = self.td_computer.compute_hindsight_td_error(
                states, actions, rewards, next_states, dones
            )
        else:
            hindsight_td = self.td_computer.compute_standard_td_error(
                states, actions, rewards, next_states, dones
            )
        loss = self.agent.update(batch)

        return loss, hindsight_td


def test_hindsight_td():
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from phase1_baseline.dqn import DQNAgent
    from config import BASELINE_CONFIG, HINDSIGHT_TD_CONFIG

    print("Testing Hindsight TD Error computation...")

    # Create agent
    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        config=BASELINE_CONFIG,
        device='cpu'
    )

    # Create hindsight TD computer
    htd = HindsightTDComputer(agent, HINDSIGHT_TD_CONFIG)

    # Create dummy batch
    batch_size = 32
    states = torch.randn(batch_size, 4)
    actions = torch.randint(0, 2, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4)
    dones = torch.zeros(batch_size)

    # Compute hindsight TD
    hindsight_td = htd.compute_hindsight_td_error(
        states, actions, rewards, next_states, dones
    )

    # Compute standard TD
    standard_td = htd.compute_standard_td_error(
        states, actions, rewards, next_states, dones
    )

    print(f"Hindsight TD shape: {hindsight_td.shape}")
    print(f"Hindsight TD mean: {hindsight_td.mean():.4f}")
    print(f"Hindsight TD std: {hindsight_td.std():.4f}")
    print(f"Standard TD mean: {standard_td.mean():.4f}")
    print(f"Standard TD std: {standard_td.std():.4f}")
    print(f"Difference: {np.abs(hindsight_td - standard_td).mean():.4f}")

    print("Hindsight TD test passed!")


if __name__ == '__main__':
    test_hindsight_td()
