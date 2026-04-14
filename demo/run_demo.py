from __future__ import annotations

import argparse
import time

import torch

from app.env import EmailEnv
from agents.rule_based import rule_based_agent
from agents.random_agent import random_agent
from training.state_encoder import get_state_dim, encode_state
from training.train_ppo import PPOPolicy, action_indices_to_env_action


def load_ppo_policy(checkpoint_path: str) -> PPOPolicy:
    policy = PPOPolicy(input_dim=get_state_dim())
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def get_agent_action(agent_name: str, observation, policy=None):
    if agent_name == "random":
        return random_agent(observation)

    if agent_name == "rule":
        return rule_based_agent(observation)

    if agent_name == "ppo":
        if policy is None:
            raise ValueError("PPO policy is required")
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


def print_step(observation, action, reward, info):
    current_email = observation.get("current_email") or {}
    extracted = observation.get("extracted") or {}

    print("\n" + "=" * 120)
    print(f"Email ID:        {current_email.get('id')}")
    print(f"Sender:          {current_email.get('sender')}")
    print(f"Subject:         {current_email.get('subject')}")
    print(f"Category truth:  {current_email.get('category')}")
    print(f"Priority truth:  {current_email.get('priority')}")
    print(f"Urgency score:   {current_email.get('urgency_score')}")
    print(f"Adversarial:     {current_email.get('adversarial')}")
    print(f"Deadline steps:  {current_email.get('deadline_steps')}")
    print("-" * 120)
    print(f"Extracted:       {extracted}")
    print("-" * 120)
    print(f"Chosen category: {action.classify_category}")
    print(f"Chosen priority: {action.classify_priority}")
    print(f"Schedule action: {action.schedule_action}")
    print(f"Response text:   {action.response_text}")
    print(f"Reward:          {reward:.2f}")
    print(f"Grading:         {info.get('grading', {})}")
    print(f"Metrics:         {info.get('metrics', {})}")
    print("=" * 120)


def run_demo(agent_name: str, checkpoint_path: str = "training/checkpoints/ppo_email_triage.pt", sleep_seconds: float = 0.0):
    env = EmailEnv()
    observation = env.reset()
    done = False
    total_reward = 0.0
    policy = None

    if agent_name == "ppo":
        policy = load_ppo_policy(checkpoint_path)

    step_num = 0
    while not done:
        step_num += 1
        action = get_agent_action(agent_name, observation, policy=policy)
        next_observation, reward, done, info = env.step(action)

        print(f"\nSTEP {step_num}")
        print_step(next_observation if not done else observation, action, float(reward), info)

        total_reward += float(reward)
        observation = next_observation

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    print("\n" + "#" * 120)
    print(f"Demo complete for agent={agent_name}")
    print(f"Total reward: {total_reward:.2f}")
    print("#" * 120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="rule", choices=["random", "rule", "ppo"])
    parser.add_argument("--checkpoint", type=str, default="training/checkpoints/ppo_email_triage.pt")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    run_demo(agent_name=args.agent, checkpoint_path=args.checkpoint, sleep_seconds=args.sleep)