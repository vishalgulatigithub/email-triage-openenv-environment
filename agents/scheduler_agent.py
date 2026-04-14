from __future__ import annotations

from typing import Any, Dict, Tuple


SCHEDULE_ACTIONS = ["reply", "escalate", "defer", "ignore"]


def schedule_email(
    observation: Dict[str, Any],
    classified_category: str,
    classified_priority: str,
) -> Tuple[str, str]:
    """
    Scheduler agent:
        input:
            - environment observation
            - classifier outputs
        output:
            - schedule_action
            - response_text
    """
    current_email = observation.get("current_email") or {}
    extracted = observation.get("extracted") or {}
    metrics = observation.get("metrics") or {}

    inbox_size = int(observation.get("inbox_size", 0))
    pending_urgent = int(observation.get("pending_urgent", 0))
    spike_active = bool(observation.get("spike_active", False))

    subject = str(current_email.get("subject", "")).lower()
    body = str(current_email.get("body", "")).lower()
    sender = str(current_email.get("sender", "")).lower()

    # Safety first
    if classified_category in {"spam", "phishing"}:
        return "ignore", ""

    # Leadership / critical escalation
    if classified_category == "escalation" or classified_priority == "critical":
        if "production" in body or "system down" in body or "major client" in body:
            return "escalate", _draft_escalation_reply(current_email)
        if "ceo@" in sender or "cto@" in sender:
            return "escalate", _draft_escalation_reply(current_email)

    # High priority support/complaint/billing
    if classified_priority == "high":
        if pending_urgent >= 3 or spike_active:
            return "escalate", _draft_escalation_reply(current_email)
        return "reply", _draft_reply(current_email, classified_category)

    # Medium cases
    if classified_priority == "medium":
        if classified_category in {"complaint", "billing", "support"}:
            if inbox_size >= 8 and pending_urgent >= 2:
                return "defer", ""
            return "reply", _draft_reply(current_email, classified_category)

        if classified_category == "newsletter":
            return "defer", ""

        return "defer", ""

    # Low priority
    if classified_category == "newsletter":
        return "ignore", ""

    if inbox_size >= 10 and extracted.get("requires_response", False) is False:
        return "ignore", ""

    # Default
    if any(token in subject + " " + body for token in ["follow up", "checking in", "whenever convenient"]):
        return "defer", ""

    return "ignore", ""


def _draft_reply(current_email: Dict[str, Any], category: str) -> str:
    sender = current_email.get("sender", "there")

    if category == "complaint":
        return (
            f"Hi, sorry about the issue you faced. We have reviewed your email and "
            f"will help resolve the damaged item case as quickly as possible."
        )

    if category == "billing":
        return (
            f"Hi, thanks for flagging the billing issue. We are reviewing the invoice "
            f"details and will follow up with clarification shortly."
        )

    if category == "support":
        return (
            f"Hi, thanks for reporting this support issue. Our team is checking the "
            f"problem and we will update you with the next steps soon."
        )

    return (
        f"Hi, thanks for your email. We have reviewed the request and will get back "
        f"to you shortly with an update."
    )


def _draft_escalation_reply(current_email: Dict[str, Any]) -> str:
    return (
        "Acknowledged. This issue has been escalated for immediate review and the "
        "relevant team is being notified now."
    )