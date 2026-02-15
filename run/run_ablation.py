# one ablation run (htd_lfiw, tce_only, etc)

import argparse
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from phase3_rmer.rmer_agent import RMERAgent
from config.ablation_configs import get_config, get_run_name


def evaluate(agent, env_name, seed, num_episodes):
    env = gym.make(env_name)
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


def train_ablation(config_name, env_name, seed, total_steps):
    config = get_config(config_name)
    config["total_steps"] = total_steps

    env = gym.make(env_name)
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cpu")

    agent = RMERAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        total_steps=total_steps,
        device=device,
        seed=seed,
    )

    run_name = get_run_name(config_name, env_name.replace("/", "_"), seed)
    writer = SummaryWriter(f"runs/{run_name}")

    state, _ = env.reset()
    agent.start_episode()
    episode_reward = 0
    episode_count = 0

    pbar = tqdm(range(total_steps), desc=f"{config_name} s{seed}")

    for step in pbar:
        agent.dqn_agent.decay_epsilon_by_env_step(step)
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        agent.push(state, action, reward, next_state, done)

        if (
            step >= config["learning_starts"]
            and step % config["train_freq"] == 0
        ):
            stats = agent.update(env_step=step)
            if stats is not None:
                writer.add_scalar("train/loss", stats["loss"], step)
                writer.add_scalar("train/epsilon", stats["epsilon"], step)
                if "lfiw_loss" in stats and stats["lfiw_loss"] is not None:
                    writer.add_scalar("train/lfiw_loss", stats["lfiw_loss"], step)
                    writer.add_scalar(
                        "train/lfiw_accuracy", stats["lfiw_accuracy"], step
                    )

        if done:
            agent.end_episode()
            writer.add_scalar("train/episode_reward", episode_reward, step)
            episode_count += 1
            pbar.set_postfix(ep=episode_count, r=f"{episode_reward:.1f}")
            state, _ = env.reset()
            agent.start_episode()
            episode_reward = 0
        else:
            state = next_state

        if step % config.get("eval_freq", 20000) == 0 and step > 0:
            eval_reward = evaluate(
                agent, env_name, seed, config.get("eval_episodes", 10)
            )
            writer.add_scalar("Reward/eval", eval_reward, step)
            print(f"\nStep {step}: Eval reward = {eval_reward:.2f}")

    writer.close()
    env.close()

    final_reward = evaluate(agent, env_name, seed, 10)
    print(f"\n{config_name} complete! Final: {final_reward:.2f}")
    return final_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["lfiw_only", "tce_only", "htd_lfiw", "htd_tce", "lfiw_tce"],
    )
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200000)
    args = parser.parse_args()

    train_ablation(args.config, args.env, args.seed, args.steps)
