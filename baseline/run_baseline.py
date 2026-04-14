from __future__ import annotations

from app.env import EmailEnv
from agents.rule_based import rule_based_agent


def run_rule_baseline(episodes: int = 10):
    env = EmailEnv()
    episode_rewards = []

    for episode in range(episodes):
        observation = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = rule_based_agent(observation)
            observation, reward, done, info = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: reward={total_reward:.2f}")

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"\nAverage rule-based reward over {episodes} episodes: {avg_reward:.2f}")
    return episode_rewards


if __name__ == "__main__":
    run_rule_baseline(episodes=10)