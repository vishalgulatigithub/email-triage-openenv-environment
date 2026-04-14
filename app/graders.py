from __future__ import annotations

from typing import Any, Dict

from app.models import Action


def grade(
    email_data: Dict[str, Any],
    action: Action,
    extracted: Dict[str, Any],
    env_context: Dict[str, Any],
) -> Dict[str, float]:
    """
    Grade a combined multi-agent decision.

    classifier agent:
        - classify_category
        - classify_priority

    scheduler agent:
        - schedule_action in {reply, escalate, defer, ignore}

    Output is decomposed so reward shaping and dashboarding become easy.
    """
    true_category = (email_data.get("category") or "general").lower()
    true_priority = (email_data.get("priority") or "medium").lower()

    predicted_category = (action.classify_category or "").lower()
    predicted_priority = (action.classify_priority or "").lower()
    schedule_action = (action.schedule_action or "").lower()

    classification_score = score_classification(
        true_category=true_category,
        true_priority=true_priority,
        predicted_category=predicted_category,
        predicted_priority=predicted_priority,
    )

    scheduling_score = score_scheduling(
        true_category=true_category,
        true_priority=true_priority,
        schedule_action=schedule_action,
    )

    sla_score = score_sla(
        email_data=email_data,
        schedule_action=schedule_action,
        env_context=env_context,
    )

    safety_score = score_safety(
        email_data=email_data,
        extracted=extracted,
        schedule_action=schedule_action,
        predicted_category=predicted_category,
    )

    efficiency_score = score_efficiency(
        email_data=email_data,
        schedule_action=schedule_action,
        env_context=env_context,
    )

    backlog_penalty = score_backlog_penalty(
        schedule_action=schedule_action,
        env_context=env_context,
        email_data=email_data,
    )

    escalation_penalty = score_escalation_penalty(
        email_data=email_data,
        schedule_action=schedule_action,
    )

    total_score = (
        classification_score
        + scheduling_score
        + sla_score
        + safety_score
        + efficiency_score
        - backlog_penalty
        - escalation_penalty
    )

    return {
        "classification_score": round(classification_score, 4),
        "scheduling_score": round(scheduling_score, 4),
        "sla_score": round(sla_score, 4),
        "safety_score": round(safety_score, 4),
        "efficiency_score": round(efficiency_score, 4),
        "backlog_penalty": round(backlog_penalty, 4),
        "escalation_penalty": round(escalation_penalty, 4),
        "total_score": round(total_score, 4),
    }


def score_classification(
    true_category: str,
    true_priority: str,
    predicted_category: str,
    predicted_priority: str,
) -> float:
    score = 0.0

    if predicted_category == true_category:
        score += 1.0
    elif _category_is_related(true_category, predicted_category):
        score += 0.4
    else:
        score -= 0.6

    if predicted_priority == true_priority:
        score += 1.0
    elif _priority_distance(true_priority, predicted_priority) == 1:
        score += 0.3
    else:
        score -= 0.7

    return score


def score_scheduling(true_category: str, true_priority: str, schedule_action: str) -> float:
    optimal_actions = _optimal_schedule_actions(true_category, true_priority)

    if schedule_action in optimal_actions:
        return 1.8

    # Near-miss handling
    if true_priority in {"high", "critical"} and schedule_action == "defer":
        return -0.8

    if true_category in {"spam", "phishing"} and schedule_action == "reply":
        return -2.0

    if true_category == "newsletter" and schedule_action == "ignore":
        return 1.2

    return -0.6


def score_sla(email_data: Dict[str, Any], schedule_action: str, env_context: Dict[str, Any]) -> float:
    deadline_steps = email_data.get("deadline_steps")
    priority = (email_data.get("priority") or "medium").lower()
    pending_urgent = int(env_context.get("pending_urgent", 0))

    if deadline_steps is None:
        return 0.0

    if priority in {"high", "critical"} and deadline_steps <= 1:
        if schedule_action in {"reply", "escalate"}:
            return 1.4
        return -2.0

    if priority in {"high", "critical"} and schedule_action == "defer":
        return -1.2

    if pending_urgent >= 3 and schedule_action == "ignore" and priority in {"medium", "high", "critical"}:
        return -0.8

    return 0.3


def score_safety(
    email_data: Dict[str, Any],
    extracted: Dict[str, Any],
    schedule_action: str,
    predicted_category: str,
) -> float:
    true_category = (email_data.get("category") or "general").lower()
    phishing_risk = float(email_data.get("phishing_risk", extracted.get("phishing_risk", 0.0)))
    spam_score = float(email_data.get("spam_score", extracted.get("spam_score", 0.0)))

    if true_category == "phishing" and schedule_action == "reply":
        return -2.5
    if phishing_risk >= 0.7 and schedule_action == "reply":
        return -2.0
    if spam_score >= 0.75 and schedule_action == "escalate":
        return -1.0
    if predicted_category == "phishing" and schedule_action in {"ignore", "escalate"}:
        return 0.8
    if true_category in {"spam", "phishing"} and schedule_action == "ignore":
        return 1.0
    return 0.2


def score_efficiency(email_data: Dict[str, Any], schedule_action: str, env_context: Dict[str, Any]) -> float:
    priority = (email_data.get("priority") or "medium").lower()
    response_cost = float(email_data.get("estimated_response_cost", 1.0))
    inbox_size = int(env_context.get("inbox_size", 0))

    if priority == "low" and schedule_action == "ignore":
        return 0.7
    if priority == "low" and schedule_action == "defer":
        return 0.5
    if schedule_action == "reply" and response_cost > 1.5 and inbox_size >= 6:
        return -0.5
    if schedule_action == "escalate" and priority in {"medium", "low"}:
        return -0.6
    return 0.2


def score_backlog_penalty(schedule_action: str, env_context: Dict[str, Any], email_data: Dict[str, Any]) -> float:
    pending_urgent = int(env_context.get("pending_urgent", 0))
    priority = (email_data.get("priority") or "medium").lower()

    penalty = 0.0

    if schedule_action == "defer" and priority in {"high", "critical"}:
        penalty += 0.8

    if schedule_action == "ignore" and priority in {"medium", "high", "critical"}:
        penalty += 0.8

    penalty += 0.15 * pending_urgent
    return penalty


def score_escalation_penalty(email_data: Dict[str, Any], schedule_action: str) -> float:
    priority = (email_data.get("priority") or "medium").lower()
    category = (email_data.get("category") or "general").lower()

    if schedule_action != "escalate":
        return 0.0

    if priority in {"high", "critical"} or category == "escalation":
        return 0.0

    if category in {"spam", "newsletter"}:
        return 1.0

    return 0.5


def _optimal_schedule_actions(category: str, priority: str) -> set[str]:
    if category in {"spam", "phishing"}:
        return {"ignore"}
    if category == "newsletter":
        return {"ignore", "defer"}
    if category == "escalation":
        return {"escalate"}
    if priority == "critical":
        return {"escalate", "reply"}
    if priority == "high":
        return {"reply", "escalate"}
    if priority == "medium":
        return {"reply", "defer"}
    return {"defer", "ignore"}


def _category_is_related(true_category: str, predicted_category: str) -> bool:
    related = {
        "support": {"escalation", "general"},
        "billing": {"general", "complaint"},
        "complaint": {"support", "general"},
        "spam": {"phishing"},
        "phishing": {"spam"},
        "newsletter": {"general"},
    }
    return predicted_category in related.get(true_category, set())


def _priority_distance(a: str, b: str) -> int:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return abs(order.get(a, 1) - order.get(b, 1))