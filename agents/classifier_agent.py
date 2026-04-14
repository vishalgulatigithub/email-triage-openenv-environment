from __future__ import annotations

from typing import Any, Dict, Tuple


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


def classify_email(observation: Dict[str, Any]) -> Tuple[str, str]:
    """
    Rule-based classifier agent.
    Output:
        (category, priority)
    """
    current_email = observation.get("current_email") or {}
    extracted = observation.get("extracted") or {}

    subject = str(current_email.get("subject", "")).lower()
    body = str(current_email.get("body", "")).lower()
    sender = str(current_email.get("sender", "")).lower()
    text = f"{subject} {body}"

    # Strong phishing/spam checks first
    if extracted.get("phishing_risk", 0.0) >= 0.7 or any(
        token in text for token in ["verify account", "password", "credentials", "secure link"]
    ):
        return "phishing", _priority_from_signals(observation, hard_min="high")

    if extracted.get("spam_score", 0.0) >= 0.7 or any(
        token in text for token in ["offer", "reward", "claim", "free", "limited time", "click now"]
    ):
        return "spam", "low"

    if any(token in text for token in ["damaged", "replacement", "refund", "bad experience", "complaint"]):
        return "complaint", _priority_from_signals(observation)

    if any(token in text for token in ["invoice", "billing", "charged", "payment", "refund amount"]):
        return "billing", _priority_from_signals(observation)

    if any(token in text for token in ["outage", "system down", "production", "error", "support", "issue"]):
        return "support", _priority_from_signals(observation, boost=True)

    if any(token in sender for token in ["ceo@", "cto@", "finance@", "ops@", "vip-client@"]):
        if any(token in text for token in ["urgent", "immediately", "major client", "critical"]):
            return "escalation", "critical"
        return "escalation", _priority_from_signals(observation, hard_min="high")

    if "newsletter" in sender or "noreply" in sender or "weekly update" in text:
        return "newsletter", "low"

    # fallback to extracted heuristics
    inferred_category = extracted.get("inferred_category", "general")
    inferred_priority = extracted.get("inferred_priority", "medium")

    if inferred_category not in CATEGORIES:
        inferred_category = "general"
    if inferred_priority not in PRIORITIES:
        inferred_priority = "medium"

    return inferred_category, inferred_priority


def _priority_from_signals(
    observation: Dict[str, Any],
    boost: bool = False,
    hard_min: str | None = None,
) -> str:
    extracted = observation.get("extracted") or {}
    current_email = observation.get("current_email") or {}

    urgency_score = float(extracted.get("urgency_score", current_email.get("urgency_score", 0.0)))
    sender_importance = int(extracted.get("sender_importance", current_email.get("sender_importance", 1)))
    deadline_steps = current_email.get("deadline_steps")

    if urgency_score >= 0.9 or deadline_steps == 1:
        priority = "critical"
    elif urgency_score >= 0.7 or sender_importance >= 3:
        priority = "high"
    elif urgency_score >= 0.4 or sender_importance >= 2:
        priority = "medium"
    else:
        priority = "low"

    if boost:
        priority = _bump_priority(priority)

    if hard_min:
        priority = _max_priority(priority, hard_min)

    return priority


def _bump_priority(priority: str) -> str:
    order = ["low", "medium", "high", "critical"]
    idx = min(order.index(priority) + 1, len(order) - 1)
    return order[idx]


def _max_priority(a: str, b: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return a if order.get(a, 1) >= order.get(b, 1) else b