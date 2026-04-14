from __future__ import annotations

from typing import Any, Dict, List

import torch


CATEGORY_TO_IDX = {
    "spam": 0,
    "complaint": 1,
    "billing": 2,
    "support": 3,
    "escalation": 4,
    "newsletter": 5,
    "phishing": 6,
    "general": 7,
}

PRIORITY_TO_IDX = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


def encode_state(observation: Dict[str, Any]) -> torch.Tensor:
    """
    Compact numeric state encoding for PPO.

    This avoids text embeddings initially so you can train quickly.
    Later you can replace this with sentence-transformer embeddings
    for a stronger wow factor.
    """
    current_email = observation.get("current_email") or {}
    extracted = observation.get("extracted") or {}
    metrics = observation.get("metrics") or {}
    history = observation.get("history") or []

    subject = str(current_email.get("subject", ""))
    body = str(current_email.get("body", ""))

    category = str(extracted.get("inferred_category", "general"))
    priority = str(extracted.get("inferred_priority", "medium"))

    subject_len = len(subject.split())
    body_len = len(body.split())

    features: List[float] = [
        float(subject_len),
        float(body_len),
        float(observation.get("time_step", 0)),
        float(observation.get("time_of_day", 0)),
        float(observation.get("inbox_size", 0)),
        float(observation.get("pending_urgent", 0)),
        float(1 if observation.get("spike_active", False) else 0),
        float(extracted.get("urgency_hits", 0)),
        float(extracted.get("complaint_hits", 0)),
        float(extracted.get("billing_hits", 0)),
        float(extracted.get("support_hits", 0)),
        float(extracted.get("spam_hits", 0)),
        float(extracted.get("phishing_hits", 0)),
        float(extracted.get("urgency_score", 0.0)),
        float(extracted.get("spam_score", 0.0)),
        float(extracted.get("phishing_risk", 0.0)),
        float(extracted.get("sender_importance", current_email.get("sender_importance", 1))),
        float(current_email.get("deadline_steps", 0) or 0),
        float(current_email.get("urgency_score", 0.0)),
        float(current_email.get("spam_score", 0.0)),
        float(current_email.get("phishing_risk", 0.0)),
        float(1 if current_email.get("adversarial", False) else 0),
        float(1 if current_email.get("requires_response", False) else 0),
        float(current_email.get("estimated_response_cost", 0.0)),
        float(CATEGORY_TO_IDX.get(category, CATEGORY_TO_IDX["general"])),
        float(PRIORITY_TO_IDX.get(priority, PRIORITY_TO_IDX["medium"])),
        float(metrics.get("emails_seen", 0)),
        float(metrics.get("emails_processed", 0)),
        float(metrics.get("urgent_handled", 0)),
        float(metrics.get("urgent_missed", 0)),
        float(metrics.get("spam_ignored", 0)),
        float(metrics.get("phishing_replied", 0)),
        float(metrics.get("unnecessary_escalations", 0)),
        float(metrics.get("sla_breaches", 0)),
        float(metrics.get("cumulative_reward", 0.0)),
        float(len(history)),
    ]

    return torch.tensor(features, dtype=torch.float32)


def get_state_dim() -> int:
    dummy = {
        "current_email": {},
        "extracted": {},
        "metrics": {},
        "history": [],
    }
    return int(encode_state(dummy).shape[0])