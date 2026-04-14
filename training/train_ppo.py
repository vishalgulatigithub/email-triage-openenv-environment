from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from app.env import EmailEnv
from app.models import Action
from training.state_encoder import encode_state, get_state_dim


CATEGORY_ACTIONS = [
    "spam",
    "complaint",
    "billing",
    "support",
    "escalation",
    "newsletter",
    "phishing",
    "general",
]

PRIORITY_ACTIONS = ["low", "medium", "high", "critical"]
SCHEDULE_ACTIONS = ["reply", "escalate", "defer", "ignore"]


@dataclass
class Transition:
    state: torch.Tensor
    category_action: int
    priority_action: int
    schedule_action: int
    logprob: torch.Tensor
    reward: float
    value: torch.Tensor
    done: bool


class PPOPolicy(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.category_head = nn.Linear(128, len(CATEGORY_ACTIONS))
        self.priority_head = nn.Linear(128, len(PRIORITY_ACTIONS))
        self.schedule_head = nn.Linear(128, len(SCHEDULE_ACTIONS))
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.shared(x)
        category_logits = self.category_head(features)
        priority_logits = self.priority_head(features)
        schedule_logits = self.schedule_head(features)
        value = self.value_head(features).squeeze(-1)

        return category_logits, priority_logits, schedule_logits, value

    def act(self, state: torch.Tensor) -> Tuple[int, int, int, torch.Tensor, torch.Tensor]:
        category_logits, priority_logits, schedule_logits, value = self.forward(state)

        category_dist = Categorical(logits=category_logits)
        priority_dist = Categorical(logits=priority_logits)
        schedule_dist = Categorical(logits=schedule_logits)

        category_action = category_dist.sample()
        priority_action = priority_dist.sample()
        schedule_action = schedule_dist.sample()

        logprob = (
            category_dist.log_prob(category_action)
            + priority_dist.log_prob(priority_action)
            + schedule_dist.log_prob(schedule_action)
        )

        return (
            int(category_action.item()),
            int(priority_action.item()),
            int(schedule_action.item()),
            logprob.squeeze(0),
            value.squeeze(0),
        )

    def evaluate_actions(
        self,
        states: torch.Tensor,
        category_actions: torch.Tensor,
        priority_actions: torch.Tensor,
        schedule_actions: torch.Tensor,
    ):
        category_logits, priority_logits, schedule_logits, values = self.forward(states)

        category_dist = Categorical(logits=category_logits)
        priority_dist = Categorical(logits=priority_logits)
        schedule_dist = Categorical(logits=schedule_logits)

        logprobs = (
            category_dist.log_prob(category_actions)
            + priority_dist.log_prob(priority_actions)
            + schedule_dist.log_prob(schedule_actions)
        )

        entropies = (
            category_dist.entropy()
            + priority_dist.entropy()
            + schedule_dist.entropy()
        )

        return logprobs, entropies, values


def action_indices_to_env_action(
    observation,
    category_idx: int,
    priority_idx: int,
    schedule_idx: int,
) -> Action:
    current_email = observation.get("current_email") or {}
    category = CATEGORY_ACTIONS[category_idx]
    priority = PRIORITY_ACTIONS[priority_idx]
    schedule_action = SCHEDULE_ACTIONS[schedule_idx]

    response_text = ""
    if schedule_action == "reply":
        response_text = build_response_text(category, priority)

    return Action(
        email_id=current_email.get("id"),
        classify_category=category,
        classify_priority=priority,
        schedule_action=schedule_action,
        response_text=response_text,
        rationale="PPO policy action",
    )


def build_response_text(category: str, priority: str) -> str:
    if category == "complaint":
        return "We are sorry for the issue and are reviewing the complaint right away."
    if category == "billing":
        return "We are reviewing the billing details and will share an update shortly."
    if category == "support":
        return "We are investigating the support issue and will follow up soon."
    if priority in {"high", "critical"}:
        return "This has been acknowledged and is being reviewed immediately."
    return "Thanks for your email. We will get back to you shortly."


def compute_returns_and_advantages(
    rewards: List[float],
    values: List[torch.Tensor],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    returns = []
    advantages = []

    next_value = 0.0
    gae = 0.0

    value_list = [float(v.item()) for v in values]

    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - value_list[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = value_list[t]
        returns.insert(0, gae + value_list[t])

    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)

    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = get_state_dim()

    policy = PPOPolicy(input_dim=input_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    env = EmailEnv()

    num_episodes = 400
    ppo_epochs = 6
    clip_eps = 0.15
    value_coef = 0.6
    entropy_coef = 0.005
    max_grad_norm = 0.5

    episode_rewards = []

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        transitions: List[Transition] = []
        total_reward = 0.0

        while not done:
            state = encode_state(observation).to(device)

            category_idx, priority_idx, schedule_idx, logprob, value = policy.act(state)
            env_action = action_indices_to_env_action(
                observation=observation,
                category_idx=category_idx,
                priority_idx=priority_idx,
                schedule_idx=schedule_idx,
            )

            next_observation, reward, done, info = env.step(env_action)

            transitions.append(
                Transition(
                    state=state.detach(),
                    category_action=category_idx,
                    priority_action=priority_idx,
                    schedule_action=schedule_idx,
                    logprob=logprob.detach(),
                    reward=float(reward),
                    value=value.detach(),
                    done=done,
                )
            )

            total_reward += float(reward)
            observation = next_observation

        episode_rewards.append(total_reward)

        states = torch.stack([t.state for t in transitions]).to(device)
        category_actions = torch.tensor([t.category_action for t in transitions], dtype=torch.long).to(device)
        priority_actions = torch.tensor([t.priority_action for t in transitions], dtype=torch.long).to(device)
        schedule_actions = torch.tensor([t.schedule_action for t in transitions], dtype=torch.long).to(device)
        old_logprobs = torch.stack([t.logprob for t in transitions]).to(device)
        rewards = [t.reward for t in transitions]
        values = [t.value for t in transitions]
        dones = [t.done for t in transitions]

        returns, advantages = compute_returns_and_advantages(rewards, values, dones)
        returns = returns.to(device)
        advantages = advantages.to(device)

        for _ in range(ppo_epochs):
            new_logprobs, entropies, new_values = policy.evaluate_actions(
                states=states,
                category_actions=category_actions,
                priority_actions=priority_actions,
                schedule_actions=schedule_actions,
            )

            ratios = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(new_values, returns)
            entropy_bonus = entropies.mean()

            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

        avg_last_10 = sum(episode_rewards[-10:]) / max(1, len(episode_rewards[-10:]))
        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"reward={total_reward:.2f} | "
            f"avg_last_10={avg_last_10:.2f}"
        )

    os.makedirs("training/checkpoints", exist_ok=True)
    torch.save(policy.state_dict(), "training/checkpoints/ppo_email_triage.pt")
    print("Saved model to training/checkpoints/ppo_email_triage.pt")

    try:
        save_rewards_csv(episode_rewards)
    except Exception as e:
        print(f"Could not save rewards CSV: {e}")

    return episode_rewards


def save_rewards_csv(rewards: List[float]) -> None:
    os.makedirs("training/checkpoints", exist_ok=True)
    path = "training/checkpoints/episode_rewards.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("episode,reward\n")
        for i, reward in enumerate(rewards, start=1):
            f.write(f"{i},{reward}\n")
    print(f"Saved rewards CSV to {path}")


if __name__ == "__main__":
    train()