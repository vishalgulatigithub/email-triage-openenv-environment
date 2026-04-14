from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List

from app.generator import build_email_variants, generate_adversarial_email, generate_spike_batch


DATA_PATH = Path("data/emails.json")


def load_tasks() -> List[Dict[str, Any]]:
    """
    Load seed emails from data/emails.json and expand them into
    a richer scenario set with synthetic variants and adversarial cases.
    """
    if not DATA_PATH.exists():
        return _fallback_tasks()

    with DATA_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        return _fallback_tasks()

    expanded: List[Dict[str, Any]] = []

    for item in raw:
        normalized = normalize_seed_email(item)
        expanded.append(normalized)
        expanded.extend(build_email_variants(normalized, n_variants=40))

    expanded.extend([generate_adversarial_email() for _ in range(60)])
    expanded.extend(generate_spike_batch(count=120))

    random.shuffle(expanded)
    return expanded


def normalize_seed_email(item: Dict[str, Any]) -> Dict[str, Any]:
    category = (item.get("category") or "general").lower()
    priority = (item.get("priority") or "medium").lower()

    return {
        "id": item.get("id", "seed-email"),
        "sender": item.get("sender", "user@example.com"),
        "subject": item.get("subject", "No subject"),
        "body": item.get("body", ""),
        "category": category,
        "priority": priority,
        "urgency_score": float(item.get("urgency_score", _priority_to_urgency(priority))),
        "sender_importance": int(item.get("sender_importance", 1)),
        "deadline_steps": int(item.get("deadline_steps", _priority_to_deadline(priority))),
        "adversarial": bool(item.get("adversarial", category in {"phishing"})),
        "spam_score": float(item.get("spam_score", 0.8 if category == "spam" else 0.1)),
        "phishing_risk": float(item.get("phishing_risk", 0.95 if category == "phishing" else 0.1)),
        "requires_response": bool(item.get("requires_response", category not in {"spam", "newsletter"})),
        "estimated_response_cost": float(item.get("estimated_response_cost", 1.0)),
        "metadata": item.get("metadata", {}),
    }


def _priority_to_urgency(priority: str) -> float:
    return {
        "low": 0.2,
        "medium": 0.45,
        "high": 0.75,
        "critical": 0.95,
    }.get(priority, 0.3)


def _priority_to_deadline(priority: str) -> int:
    return {
        "low": 6,
        "medium": 4,
        "high": 2,
        "critical": 1,
    }.get(priority, 4)


def _fallback_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "id": "fallback-1",
            "sender": "customer@shopmail.com",
            "subject": "Damaged item received",
            "body": "I received a broken item and need a replacement urgently.",
            "category": "complaint",
            "priority": "high",
            "urgency_score": 0.8,
            "sender_importance": 1,
            "deadline_steps": 2,
            "adversarial": False,
            "spam_score": 0.1,
            "phishing_risk": 0.1,
            "requires_response": True,
            "estimated_response_cost": 1.5,
            "metadata": {},
        },
        {
            "id": "fallback-2",
            "sender": "promo@offers-mail.com",
            "subject": "Claim your urgent reward",
            "body": "Click now to claim your exclusive reward.",
            "category": "spam",
            "priority": "low",
            "urgency_score": 0.35,
            "sender_importance": 0,
            "deadline_steps": 6,
            "adversarial": True,
            "spam_score": 0.95,
            "phishing_risk": 0.35,
            "requires_response": False,
            "estimated_response_cost": 0.2,
            "metadata": {},
        },
    ]