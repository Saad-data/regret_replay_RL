# remern = rmer with error network instead of tce

import argparse
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from phase3_rmer.remern_agent import ReMERNAgent
from config import RMER_CONFIG


def evaluate(agent, env_name, seed, num_episodes, render=False):
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep + 1000)
        episode_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action = agent.select_action(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    env.close()
    return np.mean(rewards)


def train_remern(env_name, seed, total_steps, config, render=False):
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cpu")

    config = {**config, 'total_steps': total_steps}
    agent = ReMERNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        total_steps=total_steps,
        device=device,
        seed=seed,
    )

    run_name = f"remern_{env_name.replace('/', '_')}_seed{seed}"
    writer = SummaryWriter(f"runs/{run_name}")

    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    buffer = agent.replay_buffer

    pbar = tqdm(range(total_steps), desc="Training ReMERN")

    for step in pbar:
        agent.decay_epsilon_by_env_step(step)
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        agent.push(state, action, reward, next_state, done)

        if step >= config['learning_starts'] and step % config['train_freq'] == 0:
            if len(buffer) >= config['batch_size']:
                states, actions, rewards, next_states, dones, indices = buffer.sample(config['batch_size'])
                weights = torch.ones_like(rewards)

                states_t = states.to(device)
                actions_t = actions.to(device)
                rewards_t = rewards.to(device)
                next_states_t = next_states.to(device)
                dones_t = dones.to(device)
                weights_t = weights.to(device)

                with torch.no_grad():
                    q_values = agent.q_network(states_t)

                loss, htd_errors = agent.update_batch(
                    states_t, actions_t, rewards_t, next_states_t, dones_t, indices, weights_t
                )

                if step % buffer.error_update_freq == 0:
                    error_loss, error_mae = buffer.update_error_network(states_t, actions_t, q_values)
                    writer.add_scalar("train/error_network_loss", error_loss, step)
                    writer.add_scalar("train/error_network_mae", error_mae, step)

                priority_stats = buffer.update_priorities(
                    indices, htd_errors, states_t, actions_t, q_values
                )

                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/epsilon", agent.epsilon, step)
                writer.add_scalar("train/mean_priority", priority_stats["mean_priority"], step)
                writer.add_scalar("train/mean_htd", priority_stats["mean_htd"], step)
                writer.add_scalar("train/mean_lfiw", priority_stats["mean_lfiw"], step)
                writer.add_scalar("train/mean_accuracy", priority_stats["mean_accuracy"], step)

                if buffer.use_lfiw and step % buffer.lfiw_estimator.update_freq == 0:
                    lfiw_loss, lfiw_acc = buffer.update_lfiw()
                    if lfiw_loss is not None:
                        writer.add_scalar("train/lfiw_loss", lfiw_loss, step)
                        writer.add_scalar("train/lfiw_accuracy", lfiw_acc, step)

        if done:
            writer.add_scalar("train/episode_reward", episode_reward, step)
            episode_count += 1
            pbar.set_postfix(episode=episode_count, reward=f"{episode_reward:.1f}", epsilon=f"{agent.epsilon:.3f}")
            state, _ = env.reset()
            episode_reward = 0
        else:
            state = next_state

        if step % config.get("eval_freq", 20000) == 0 and step > 0:
            eval_reward = evaluate(agent, env_name, seed, config.get("eval_episodes", 10), render=render)
            writer.add_scalar("Reward/eval", eval_reward, step)
            print(f"\nStep {step}: Eval reward = {eval_reward:.2f}")

    writer.close()
    env.close()

    final_reward = evaluate(agent, env_name, seed, 10, render=render)
    print(f"\nTraining complete! Final eval reward: {final_reward:.2f}")
    return final_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--render", action="store_true", help="Render the environment and agent during training and evaluation")
    args = parser.parse_args()

    config = RMER_CONFIG.copy()
    train_remern(args.env, args.seed, args.steps, config, render=args.render)
