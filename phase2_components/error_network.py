# remern: E(s,a) = max_a' Q(s,a') - Q(s,a), learned

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


class ErrorNetwork(nn.Module):
    # s,a -> suboptimality (then exp(-e) for accuracy in priority)

    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], lr=1e-3):
        super(ErrorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.update_count = 0

    def forward(self, states, actions):
        batch_size = states.shape[0]
        actions = actions.long().clamp(0, self.action_dim - 1)
        actions_onehot = torch.zeros(batch_size, self.action_dim, device=states.device)
        actions_onehot.scatter_(1, actions.unsqueeze(1), 1.0)
        inputs = torch.cat([states, actions_onehot], dim=1)
        errors = torch.relu(self.network(inputs).squeeze(-1))

        return errors

    def compute_targets(self, q_values, actions):
        q_best = q_values.max(dim=1)[0]
        q_executed = q_values.gather(1, actions.unsqueeze(1).long()).squeeze(1)
        suboptimality = q_best - q_executed

        return suboptimality.detach()

    def update(self, states, actions, q_values):
        targets = self.compute_targets(q_values, actions)
        predictions = self.forward(states, actions)
        loss = nn.functional.mse_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        mae = torch.abs(predictions - targets).mean().item()

        return loss.item(), mae

    def get_accuracy(self, states, actions):
        with torch.no_grad():
            errors = self.forward(states, actions)
            accuracy = torch.exp(-errors)

        return accuracy.cpu().numpy()
