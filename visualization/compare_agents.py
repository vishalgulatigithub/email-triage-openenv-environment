from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from training.evaluate import compare_agents


def extract_metric(comparison: Dict[str, Any], metric_name: str) -> tuple[list[str], list[float]]:
    agents = []
    values = []

    for agent_name, result in comparison.items():
        metric = result.get("aggregated", {}).get(metric_name, {})
        agents.append(agent_name)
        values.append(float(metric.get("mean", 0.0)))

    return agents, values


def plot_metric(comparison: Dict[str, Any], metric_name: str, output_path: str):
    agents, values = extract_metric(comparison, metric_name)

    plt.figure(figsize=(8, 5))
    plt.bar(agents, values)
    plt.title(metric_name.replace("_", " ").title())
    plt.xlabel("Agent")
    plt.ylabel("Mean Value")
    plt.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    print(f"Saved {metric_name} chart to {output_path}")


def run_all_charts(num_episodes: int = 20):
    comparison = compare_agents(num_episodes=num_episodes)

    metrics = [
        "total_reward",
        "urgent_handled",
        "urgent_missed",
        "sla_breaches",
    ]

    for metric in metrics:
        output_path = f"visualization/{metric}_comparison.png"
        plot_metric(comparison, metric, output_path)


def plot_combined_comparison(comparison):
    import matplotlib.pyplot as plt

    agents = []
    rewards = []
    safe_scores = []

    for agent, result in comparison.items():
        agg = result["aggregated"]
        agents.append(agent)
        rewards.append(agg["total_reward"]["mean"])
        safe_scores.append(agg["safe_policy_score"]["mean"])

    x = range(len(agents))

    plt.figure(figsize=(10, 6))
    plt.bar(x, rewards, width=0.4, label="Reward", align="center")
    plt.bar([i + 0.4 for i in x], safe_scores, width=0.4, label="Safety Score", align="center")

    plt.xticks([i + 0.2 for i in x], agents)
    plt.title("Reward vs Safety Trade-off")
    plt.legend()
    plt.grid(True)

    plt.savefig("visualization/reward_vs_safety.png")
    plt.show()


if __name__ == "__main__":
    run_all_charts(num_episodes=20)