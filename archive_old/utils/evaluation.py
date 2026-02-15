"""
Utility functions for evaluation and analysis.
"""

import numpy as np
import gymnasium as gym
from scipy import stats


def evaluate_agent(agent, env_name, num_episodes=100, seed=None):
    """
    Evaluate an agent's performance.

    Args:
        agent: Agent to evaluate (must have select_action method)
        env_name: Name of environment
        num_episodes: Number of episodes to run
        seed: Random seed for evaluation

    Returns:
        Dictionary with evaluation statistics
    """
    env = gym.make(env_name, render_mode=None)
    if seed is not None:
        env.action_space.seed(seed)

    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=(seed + ep) if seed is not None else None)
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'rewards': episode_rewards,
        'lengths': episode_lengths,
    }


def compute_learning_curve_metrics(rewards, target_reward=None, window=100):
    """
    Compute metrics from learning curve.

    Args:
        rewards: List of episode rewards
        target_reward: Target reward to measure convergence
        window: Window size for smoothing

    Returns:
        Dictionary with learning curve metrics
    """
    rewards = np.array(rewards)

    # Smooth rewards
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    else:
        smoothed = rewards

    metrics = {
        'final_performance': np.mean(rewards[-window:]) if len(rewards) >= window else np.mean(rewards),
        'peak_performance': np.max(smoothed),
        'stability': np.std(rewards[-window:]) if len(rewards) >= window else np.std(rewards),
    }

    # Convergence metrics
    if target_reward is not None:
        # Episodes to reach target
        for i, r in enumerate(smoothed):
            if r >= target_reward:
                metrics['episodes_to_target'] = i
                break
        else:
            metrics['episodes_to_target'] = len(rewards)  # Never reached

        # Percentage of episodes above target
        metrics['percent_above_target'] = (rewards >= target_reward).mean() * 100

    return metrics


def compare_agents_statistical(results_dict, metric='mean_reward', alpha=0.05):
    """
    Perform statistical comparison between agents.

    Args:
        results_dict: Dictionary mapping agent_name -> evaluation results
        metric: Metric to compare
        alpha: Significance level

    Returns:
        Dictionary with comparison results
    """
    agents = list(results_dict.keys())
    n_agents = len(agents)

    # Pairwise t-tests
    comparisons = {}

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            agent1, agent2 = agents[i], agents[j]

            data1 = results_dict[agent1]['rewards']
            data2 = results_dict[agent2]['rewards']

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(data1, data2)

            # Compute effect size (Cohen's d)
            mean_diff = np.mean(data1) - np.mean(data2)
            pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            comparisons[f"{agent1}_vs_{agent2}"] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'cohens_d': cohens_d,
                'mean_diff': mean_diff,
            }

    return comparisons


def compute_sample_efficiency(training_data, target_reward, episode_length_avg):
    """
    Compute sample efficiency metrics.

    Args:
        training_data: List of (step, eval_reward) tuples
        target_reward: Target reward threshold
        episode_length_avg: Average episode length

    Returns:
        Dictionary with sample efficiency metrics
    """
    steps, rewards = zip(*training_data)
    steps = np.array(steps)
    rewards = np.array(rewards)

    # Find first time target is reached
    reached_indices = np.where(rewards >= target_reward)[0]

    if len(reached_indices) > 0:
        steps_to_target = steps[reached_indices[0]]
        episodes_to_target = steps_to_target / episode_length_avg
    else:
        steps_to_target = steps[-1]
        episodes_to_target = steps_to_target / episode_length_avg

    # Compute area under curve (AUC)
    auc = np.trapz(rewards, steps)
    max_r = float(np.max(rewards)) if len(rewards) > 0 else 1.0
    normalized_auc = auc / (steps[-1] * max_r) if steps[-1] * max_r > 0 else 0

    return {
        'steps_to_target': float(steps_to_target),
        'episodes_to_target': float(episodes_to_target),
        'auc': float(auc),
        'normalized_auc': float(normalized_auc),
        'final_reward': float(rewards[-1]) if len(rewards) > 0 else 0,
    }


def analyze_priority_distribution(priority_history):
    """
    Analyze priority distribution over training.

    Args:
        priority_history: List of (step, priority_stats) tuples

    Returns:
        Dictionary with priority analysis
    """
    if not priority_history:
        return {}

    steps, stats_list = zip(*priority_history)

    # Extract metrics over time
    means = [s['mean'] for s in stats_list]
    stds = [s['std'] for s in stats_list]
    maxs = [s['max'] for s in stats_list]

    # Component contributions
    components = ['hindsight_td', 'lfiw_weights', 'tce_scores']
    component_evolution = {
        comp: [s['component_stats'].get(comp, {}).get('mean', 0) for s in stats_list]
        for comp in components
    }

    # Compute trends (linear regression)
    trends = {}
    for comp, values in component_evolution.items():
        if len(values) > 1:
            slope, intercept, r_value, _, _ = stats.linregress(range(len(values)), values)
            trends[comp] = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'direction': 'increasing' if slope > 0 else 'decreasing',
            }

    return {
        'mean_priority_trajectory': means,
        'std_priority_trajectory': stds,
        'max_priority_trajectory': maxs,
        'component_evolution': component_evolution,
        'component_trends': trends,
        'final_component_balance': {
            comp: component_evolution[comp][-1]
            for comp in components
        }
    }


class PerformanceTracker:
    """Track and analyze performance metrics during training."""

    def __init__(self, window_size=100):
        """
        Initialize tracker.

        Args:
            window_size: Window size for computing statistics
        """
        self.window_size = window_size
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.steps = []

    def add_episode(self, reward, length, step):
        """Add episode data."""
        self.rewards.append(reward)
        self.lengths.append(length)
        self.steps.append(step)

    def add_loss(self, loss):
        """Add training loss."""
        self.losses.append(loss)

    def get_recent_stats(self):
        """Get statistics from recent window."""
        recent_rewards = self.rewards[-self.window_size:]
        recent_lengths = self.lengths[-self.window_size:]

        return {
            'mean_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'std_reward': np.std(recent_rewards) if recent_rewards else 0,
            'mean_length': np.mean(recent_lengths) if recent_lengths else 0,
            'num_episodes': len(self.rewards),
        }

    def check_convergence(self, target_reward, patience=5):
        """
        Check if training has converged.

        Args:
            target_reward: Target reward threshold
            patience: Number of evaluations above target needed

        Returns:
            True if converged, False otherwise
        """
        if len(self.rewards) < self.window_size * patience:
            return False

        recent_means = []
        for i in range(-patience * self.window_size, 0, self.window_size):
            chunk = self.rewards[i:i + self.window_size]
            if len(chunk) == self.window_size:
                recent_means.append(np.mean(chunk))

        return len(recent_means) >= patience and all(mean >= target_reward for mean in recent_means)
