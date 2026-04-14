from __future__ import annotations

import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_rewards(csv_path: str = "training/checkpoints/episode_rewards.csv") -> Tuple[List[int], List[float]]:
    episodes = []
    rewards = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))

    return episodes, rewards


def moving_average(values: List[float], window: int = 10) -> List[float]:
    if not values:
        return []

    ma = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        ma.append(sum(window_vals) / len(window_vals))
    return ma


def plot_rewards(csv_path: str = "training/checkpoints/episode_rewards.csv", output_path: str = "visualization/reward_curve.png"):
    episodes, rewards = load_rewards(csv_path)
    smoothed = moving_average(rewards, window=10)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, alpha=0.4, label="Episode Reward")
    plt.plot(episodes, smoothed, linewidth=2, label="Moving Average (10)")
    plt.title("PPO Learning Curve - Email Triage Environment")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    print(f"Saved reward plot to {output_path}")


if __name__ == "__main__":
    plot_rewards()