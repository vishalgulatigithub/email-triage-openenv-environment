from __future__ import annotations

from typing import Any, Dict


def extract_features(email_data: Dict[str, Any] | None) -> Dict[str, Any]:
    if not email_data:
        return {}

    subject = str(email_data.get("subject", ""))
    body = str(email_data.get("body", ""))
    sender = str(email_data.get("sender", ""))
    text = f"{subject} {body}".lower()

    urgency_terms = [
        "urgent",
        "immediately",
        "asap",
        "end of day",
        "time-sensitive",
        "critical",
        "blocked",
        "system down",
        "production",
    ]
    complaint_terms = ["damaged", "refund", "replacement", "bad experience", "complaint"]
    billing_terms = ["invoice", "payment", "charged", "billing", "refund amount"]
    support_terms = ["issue", "bug", "error", "support", "failing", "outage"]
    spam_terms = ["offer", "reward", "claim", "free", "limited time", "click now"]
    phishing_terms = ["verify account", "password", "credentials", "secure link", "reset password"]

    urgency_hits = sum(1 for term in urgency_terms if term in text)
    complaint_hits = sum(1 for term in complaint_terms if term in text)
    billing_hits = sum(1 for term in billing_terms if term in text)
    support_hits = sum(1 for term in support_terms if term in text)
    spam_hits = sum(1 for term in spam_terms if term in text)
    phishing_hits = sum(1 for term in phishing_terms if term in text)

    sender_importance = int(email_data.get("sender_importance", _infer_sender_importance(sender)))

    inferred_category = infer_category(
        complaint_hits=complaint_hits,
        billing_hits=billing_hits,
        support_hits=support_hits,
        spam_hits=spam_hits,
        phishing_hits=phishing_hits,
    )

    inferred_priority = infer_priority(
        urgency_hits=urgency_hits,
        sender_importance=sender_importance,
        text=text,
    )

    spam_score = min(1.0, 0.2 * spam_hits + (0.15 if "promo" in sender.lower() else 0.0))
    phishing_risk = min(1.0, 0.25 * phishing_hits + (0.2 if "verify" in text else 0.0))
    urgency_score = min(
        1.0,
        0.15 * urgency_hits
        + 0.15 * sender_importance
        + (0.2 if "production" in text or "system down" in text else 0.0),
    )

    sentiment = infer_sentiment(text)
    contains_deadline = any(token in text for token in ["end of day", "today", "now", "immediately", "asap"])
    requires_response = inferred_category not in {"spam", "newsletter", "phishing"}

    return {
        "urgency_hits": urgency_hits,
        "complaint_hits": complaint_hits,
        "billing_hits": billing_hits,
        "support_hits": support_hits,
        "spam_hits": spam_hits,
        "phishing_hits": phishing_hits,
        "urgency_score": urgency_score,
        "spam_score": spam_score,
        "phishing_risk": phishing_risk,
        "sender_importance": sender_importance,
        "inferred_category": inferred_category,
        "inferred_priority": inferred_priority,
        "sentiment": sentiment,
        "contains_deadline": contains_deadline,
        "requires_response": requires_response,
    }


def infer_category(
    complaint_hits: int,
    billing_hits: int,
    support_hits: int,
    spam_hits: int,
    phishing_hits: int,
) -> str:
    if phishing_hits >= 1:
        return "phishing"
    if spam_hits >= 2:
        return "spam"
    if complaint_hits >= 1:
        return "complaint"
    if billing_hits >= 1:
        return "billing"
    if support_hits >= 1:
        return "support"
    return "general"


def infer_priority(urgency_hits: int, sender_importance: int, text: str) -> str:
    if "production" in text or "system down" in text or "major client" in text:
        return "critical"
    if urgency_hits >= 3 or sender_importance >= 3:
        return "high"
    if urgency_hits >= 1 or sender_importance >= 2:
        return "medium"
    return "low"


def infer_sentiment(text: str) -> str:
    negative_terms = ["damaged", "angry", "bad", "issue", "error", "blocked", "failed"]
    positive_terms = ["thanks", "great", "helpful", "appreciate"]

    neg = sum(1 for t in negative_terms if t in text)
    pos = sum(1 for t in positive_terms if t in text)

    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def _infer_sender_importance(sender: str) -> int:
    sender = sender.lower()
    if sender.startswith(("ceo@", "cto@", "finance@", "ops@", "vip-client@")):
        return 3
    if sender.endswith(("company.com", "enterprise.com", "client-enterprise.com")):
        return 2
    if "promo" in sender or "noreply" in sender:
        return 0
    return 1