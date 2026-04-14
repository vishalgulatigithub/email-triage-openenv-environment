from __future__ import annotations

from typing import Any, Dict

from app.models import Action
from agents.classifier_agent import classify_email
from agents.scheduler_agent import schedule_email


def rule_based_agent(observation: Dict[str, Any]) -> Action:
    """
    Full two-agent rule-based controller:
      classifier -> scheduler -> merged Action
    """
    current_email = observation.get("current_email") or {}

    category, priority = classify_email(observation)
    schedule_action, response_text = schedule_email(
        observation=observation,
        classified_category=category,
        classified_priority=priority,
    )

    return Action(
        email_id=current_email.get("id"),
        classify_category=category,
        classify_priority=priority,
        schedule_action=schedule_action,
        response_text=response_text,
        rationale=(
            f"Rule-based pipeline chose category={category}, "
            f"priority={priority}, schedule_action={schedule_action}"
        ),
    )