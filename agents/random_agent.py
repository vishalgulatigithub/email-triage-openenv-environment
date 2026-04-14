from __future__ import annotations

import random
from typing import Dict

from app.models import Action

CATEGORIES = [
    "spam",
    "complaint",
    "billing",
    "support",
    "escalation",
    "newsletter",
    "phishing",
    "general",
]

PRIORITIES = ["low", "medium", "high", "critical"]
SCHEDULE_ACTIONS = ["reply", "escalate", "defer", "ignore"]


def random_agent(observation: Dict) -> Action:
    current_email = observation.get("current_email") or {}

    schedule_action = random.choice(SCHEDULE_ACTIONS)
    response_text = ""
    if schedule_action == "reply":
        response_text = random.choice(
            [
                "Thanks for reaching out. We will review this shortly.",
                "Acknowledged. Our team will get back soon.",
                "We have received your message and are checking it.",
            ]
        )

    return Action(
        email_id=current_email.get("id"),
        classify_category=random.choice(CATEGORIES),
        classify_priority=random.choice(PRIORITIES),
        schedule_action=schedule_action,
        response_text=response_text,
        rationale="Random baseline action",
    )