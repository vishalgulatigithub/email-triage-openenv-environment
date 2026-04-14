from __future__ import annotations

from typing import Any, Dict

from app.models import Action


def compute_reward(
    grading: Dict[str, float],
    env_context: Dict[str, Any],
    action: Action,
    email_data: Dict[str, Any],
) -> float:
    classification_score = float(grading.get("classification_score", 0.0))
    scheduling_score = float(grading.get("scheduling_score", 0.0))
    sla_score = float(grading.get("sla_score", 0.0))
    safety_score = float(grading.get("safety_score", 0.0))
    efficiency_score = float(grading.get("efficiency_score", 0.0))
    backlog_penalty = float(grading.get("backlog_penalty", 0.0))
    escalation_penalty = float(grading.get("escalation_penalty", 0.0))

    reward = 0.0

    # Keep correctness important
    reward += 1.4 * classification_score
    reward += 1.8 * scheduling_score

    # Still reward safety/SLA heavily, but slightly less dominant
    reward += 1.4 * sla_score
    reward += 1.6 * safety_score

    # Boost efficiency so PPO learns cost-aware behavior
    reward += 1.6 * efficiency_score

    # Penalize backlog clearly
    reward -= 1.0 * backlog_penalty

    # Reduce escalation penalty a bit so PPO is not overly punished
    reward -= 0.5 * escalation_penalty

    # Response shaping
    reward += _response_quality_bonus(action=action, email_data=email_data)

    # Operational pressure shaping
    reward -= _operational_pressure_penalty(
        env_context=env_context,
        email_data=email_data,
        action=action,
    )

    return max(-5.0, min(5.0, reward))


def _response_quality_bonus(action: Action, email_data: Dict[str, Any]) -> float:
    schedule_action = (action.schedule_action or "").lower()
    response_text = (action.response_text or "").strip()
    category = (email_data.get("category") or "general").lower()

    if schedule_action != "reply":
        return 0.0

    if category in {"spam", "phishing"}:
        return -1.2

    if not response_text:
        return -0.4

    length = len(response_text.split())

    if length < 4:
        return -0.2
    if 8 <= length <= 35:
        return 0.5
    if 36 <= length <= 60:
        return 0.3
    if length > 80:
        return -0.2

    return 0.2


def _operational_pressure_penalty(
    env_context: Dict[str, Any],
    email_data: Dict[str, Any],
    action: Action,
) -> float:
    pending_urgent = int(env_context.get("pending_urgent", 0))
    inbox_size = int(env_context.get("inbox_size", 0))
    schedule_action = (action.schedule_action or "").lower()
    priority = (email_data.get("priority") or "medium").lower()

    penalty = 0.0

    if pending_urgent >= 4:
        penalty += 0.15 * pending_urgent

    # Discourage wasting effort on low-value replies during high load
    if inbox_size >= 8 and schedule_action == "reply" and priority == "low":
        penalty += 0.6

    # Discourage excessive escalation under load
    if inbox_size >= 10 and schedule_action == "escalate" and priority in {"low", "medium"}:
        penalty += 0.4

    return penalty