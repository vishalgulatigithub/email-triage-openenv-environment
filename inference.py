from __future__ import annotations

import argparse
import json

import torch

from app.env import EmailEnv
from agents.rule_based import rule_based_agent
from agents.random_agent import random_agent
from training.state_encoder import encode_state, get_state_dim
from training.train_ppo import PPOPolicy, action_indices_to_env_action


def load_ppo_policy(checkpoint_path: str) -> PPOPolicy:
    policy = PPOPolicy(input_dim=get_state_dim())
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def choose_action(agent_name: str, observation, policy=None):
    if agent_name == "rule":
        return rule_based_agent(observation)

    if agent_name == "random":
        return random_agent(observation)

    if agent_name == "ppo":
        if policy is None:
            raise ValueError("PPO policy required for ppo inference")
        state = encode_state(observation)
        with torch.no_grad():
            category_idx, priority_idx, schedule_idx, _, _ = policy.act(state)
        return action_indices_to_env_action(
            observation=observation,
            category_idx=category_idx,
            priority_idx=priority_idx,
            schedule_idx=schedule_idx,
        )

    raise ValueError(f"Unsupported agent: {agent_name}")


def run(agent_name: str, checkpoint_path: str = "training/checkpoints/ppo_email_triage.pt"):
    env = EmailEnv()
    observation = env.reset()
    done = False
    total_reward = 0.0

    policy = None
    if agent_name == "ppo":
        policy = load_ppo_policy(checkpoint_path)

    trajectory = []

    while not done:
        action = choose_action(agent_name, observation, policy)
        next_observation, reward, done, info = env.step(action)

        trajectory.append(
            {
                "email_id": action.email_id,
                "classify_category": action.classify_category,
                "classify_priority": action.classify_priority,
                "schedule_action": action.schedule_action,
                "reward": float(reward),
                "grading": info.get("grading", {}),
            }
        )

        total_reward += float(reward)
        observation = next_observation

    result = {
        "agent": agent_name,
        "total_reward": total_reward,
        "steps": len(trajectory),
        "trajectory": trajectory,
    }

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["random", "rule", "ppo"], default="rule")
    parser.add_argument("--checkpoint", default="training/checkpoints/ppo_email_triage.pt")
    args = parser.parse_args()

    run(agent_name=args.agent, checkpoint_path=args.checkpoint)