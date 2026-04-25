from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt


def plot_curriculum(
    log_path: str = "training/checkpoints/curriculum_log.txt",
    output_path: str = "visualization/curriculum_progression.png",
):
    episodes = []
    levels = []
    rewards = []

    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"{log_path} not found. Run python -m training.train_ppo first."
        )

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            episodes.append(int(row["episode"]))
            levels.append(int(row["level"]))
            rewards.append(float(row["reward"]))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, levels, label="Recommended Curriculum Level")
    plt.title("Self-Improvement Curriculum Progression")
    plt.xlabel("Episode")
    plt.ylabel("Difficulty Level")
    plt.yticks([1, 2, 3])
    plt.grid(True)
    plt.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    print(f"Saved curriculum plot to {output_path}")


if __name__ == "__main__":
    plot_curriculum()