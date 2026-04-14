from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch

from app.env import EmailEnv
from app.models import Action
from agents.rule_based import rule_based_agent
from agents.random_agent import random_agent
from training.state_encoder import encode_state, get_state_dim
from training.train_ppo import (
    CATEGORY_ACTIONS,
    PRIORITY_ACTIONS,
    SCHEDULE_ACTIONS,
    PPOPolicy,
    action_indices_to_env_action,
)


def run_agent_episode(env: EmailEnv, agent_type: str, policy: PPOPolicy | None = None) -> Dict[str, Any]:
    observation = env.reset()
    done = False

    total_reward = 0.0
    steps = 0
    final_info: Dict[str, Any] = {}
    actions_log: List[Dict[str, Any]] = []

    while not done:
        if agent_type == "random":
            action = random_agent(observation)

        elif agent_type == "rule":
            action = rule_based_agent(observation)

        elif agent_type == "ppo":
            if policy is None:
                raise ValueError("PPO policy is required for agent_type='ppo'")
            state = encode_state(observation)
            with torch.no_grad():
                category_idx, priority_idx, schedule_idx, _, _ = policy.act(state)
            action = action_indices_to_env_action(
                observation=observation,
                category_idx=category_idx,
                priority_idx=priority_idx,
                schedule_idx=schedule_idx,
            )
        else:
            raise ValueError(f"Unsupported agent_type: {agent_type}")

        next_observation, reward, done, info = env.step(action)

        actions_log.append(
            {
                "email_id": action.email_id,
                "predicted_category": action.classify_category,
                "predicted_priority": action.classify_priority,
                "schedule_action": action.schedule_action,
                "reward": float(reward),
            }
        )

        total_reward += float(reward)
        steps += 1
        final_info = info
        observation = next_observation

    metrics = final_info.get("metrics", {})
    return {
        "agent_type": agent_type,
        "total_reward": total_reward,
        "steps": steps,
        "metrics": metrics,
        "actions_log": actions_log,
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = defaultdict(list)

    for result in results:
        summary["total_reward"].append(result["total_reward"])
        summary["steps"].append(result["steps"])

        metrics = result.get("metrics", {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary[key].append(value)

    aggregated = {}
    for key, values in summary.items():
        if values:
            aggregated[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    urgent_handled_mean = aggregated.get("urgent_handled", {}).get("mean", 0.0)
    urgent_missed_mean = aggregated.get("urgent_missed", {}).get("mean", 0.0)
    sla_breaches_mean = aggregated.get("sla_breaches", {}).get("mean", 0.0)

    safe_policy_score = urgent_handled_mean - urgent_missed_mean - sla_breaches_mean

    aggregated["safe_policy_score"] = {
        "mean": safe_policy_score,
        "min": safe_policy_score,
        "max": safe_policy_score,
    }

    return aggregated


def load_ppo_policy(checkpoint_path: str = "training/checkpoints/ppo_email_triage.pt") -> PPOPolicy:
    input_dim = get_state_dim()
    policy = PPOPolicy(input_dim=input_dim)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def evaluate_agent(agent_type: str, num_episodes: int = 25, checkpoint_path: str | None = None) -> Dict[str, Any]:
    env = EmailEnv()
    policy = None

    if agent_type == "ppo":
        if checkpoint_path is None:
            checkpoint_path = "training/checkpoints/ppo_email_triage.pt"
        policy = load_ppo_policy(checkpoint_path)

    results = []
    for _ in range(num_episodes):
        result = run_agent_episode(env=env, agent_type=agent_type, policy=policy)
        results.append(result)

    aggregated = aggregate_results(results)
    return {
        "agent_type": agent_type,
        "num_episodes": num_episodes,
        "aggregated": aggregated,
        "raw_results": results,
    }


def compare_agents(num_episodes: int = 25, checkpoint_path: str = "training/checkpoints/ppo_email_triage.pt") -> Dict[str, Any]:
    comparison = {
        "random": evaluate_agent("random", num_episodes=num_episodes),
        "rule": evaluate_agent("rule", num_episodes=num_episodes),
        "ppo": evaluate_agent("ppo", num_episodes=num_episodes, checkpoint_path=checkpoint_path),
    }
    return comparison


def print_comparison_table(comparison: Dict[str, Any]) -> None:
    print("=" * 120)
    print(
        f"{'AGENT':<12} {'AVG_REWARD':<15} {'URGENT_HANDLED':<18} "
        f"{'URGENT_MISSED':<18} {'SLA_BREACHES':<15} {'SAFE_SCORE':<12}"
    )
    print("=" * 120)

    for agent_name, result in comparison.items():
        agg = result["aggregated"]
        avg_reward = agg.get("total_reward", {}).get("mean", 0.0)
        urgent_handled = agg.get("urgent_handled", {}).get("mean", 0.0)
        urgent_missed = agg.get("urgent_missed", {}).get("mean", 0.0)
        sla_breaches = agg.get("sla_breaches", {}).get("mean", 0.0)
        safe_score = agg.get("safe_policy_score", {}).get("mean", 0.0)

        print(
            f"{agent_name:<12} "
            f"{avg_reward:<15.2f} "
            f"{urgent_handled:<18.2f} "
            f"{urgent_missed:<18.2f} "
            f"{sla_breaches:<15.2f} "
            f"{safe_score:<12.2f}"
        )

    print("=" * 120)


if __name__ == "__main__":
    comparison = compare_agents(num_episodes=20)
    print_comparison_table(comparison)

    