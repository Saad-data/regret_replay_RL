# dqn agent - target net + soft update (phase1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class QNetwork(nn.Module):
    # simple ff with configurable hidden dims

    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128], activation='relu'):
        super(QNetwork, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(input_dim, action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # xavier init
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        x = state
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        q_values = self.output_layer(x)
        return q_values


class DQNAgent:
    # target net + soft update, epsilon greedy

    def __init__(self, state_dim, action_dim, config, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device

        self.q_network = QNetwork(
            state_dim,
            action_dim,
            hidden_dims=config['hidden_dims'],
            activation=config['activation']
        ).to(device)

        self.target_network = QNetwork(
            state_dim,
            action_dim,
            hidden_dims=config['hidden_dims'],
            activation=config['activation']
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config['learning_rate']
        )
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.epsilon = config['epsilon_start']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay_steps = config['epsilon_decay_steps']
        self.train_step = 0  # TODO: could remove if we only use decay_epsilon_by_env_step

    def select_action(self, state, eval_mode=False):
        if eval_mode or np.random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                action = q_values.argmax(dim=1).item()
        else:
            action = np.random.randint(self.action_dim)
        return action

    def update_epsilon(self):
        # legacy - uses train_step. prefer decay_epsilon_by_env_step
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) *
            self.train_step / self.epsilon_decay_steps
        )

    def decay_epsilon_by_env_step(self, env_step):
        # decay by env step so schedule matches total_steps
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) *
            min(1.0, env_step / self.epsilon_decay_steps)
        )

    def update(self, batch):
        # batch can be (s,a,r,s',d) or (s,a,r,s',d, indices, weights) for PER
        if len(batch) == 7:
            states, actions, rewards, next_states, dones, indices, weights = batch
        else:
            states, actions, rewards, next_states, dones = batch
            weights = torch.ones(len(states))

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        td_errors = target_q_values - current_q_values
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        self.soft_update_target()
        self.train_step += 1
        # print("DEBUG loss", loss.item())  # left from debugging
        return loss.item(), td_errors.detach().cpu().numpy()

    def soft_update_target(self):
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'epsilon': self.epsilon,
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint['train_step']
        self.epsilon = checkpoint['epsilon']
