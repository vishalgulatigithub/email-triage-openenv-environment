from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List

CATEGORY_POOL = [
    "spam",
    "complaint",
    "billing",
    "support",
    "escalation",
    "newsletter",
    "phishing",
    "general",
]

PRIORITY_POOL = ["low", "medium", "high", "critical"]


def build_email_variants(seed_email: Dict[str, Any], n_variants: int = 8) -> List[Dict[str, Any]]:
    """
    Generate realistic synthetic variants from a single seed email.
    Keeps the original semantic intent while adding phrasing variation,
    sender variation, urgency manipulation, and adversarial cases.
    """
    variants: List[Dict[str, Any]] = []

    base_sender = seed_email.get("sender", "user@example.com")
    base_subject = seed_email.get("subject", "")
    base_body = seed_email.get("body", "")
    base_category = (seed_email.get("category") or "general").lower()
    base_priority = (seed_email.get("priority") or "medium").lower()

    for i in range(n_variants):
        category = mutate_category(base_category)
        priority = mutate_priority(base_priority, category)
        sender = mutate_sender(base_sender, category)
        subject = mutate_subject(base_subject, category, priority)
        body = mutate_body(base_body, category, priority)

        adversarial = infer_adversarial(subject, body, category)
        phishing_risk = 0.9 if category == "phishing" else (0.75 if adversarial else 0.1)
        spam_score = 0.9 if category == "spam" else (0.7 if adversarial else 0.1)
        urgency_score = infer_urgency_score(priority, subject, body)
        deadline_steps = infer_deadline_steps(priority, urgency_score)

        variants.append(
            {
                "id": f"{seed_email.get('id', 'email')}-{uuid.uuid4().hex[:8]}",
                "sender": sender,
                "subject": subject,
                "body": body,
                "category": category,
                "priority": priority,
                "urgency_score": urgency_score,
                "sender_importance": infer_sender_importance(sender),
                "deadline_steps": deadline_steps,
                "adversarial": adversarial,
                "spam_score": spam_score,
                "phishing_risk": phishing_risk,
                "requires_response": category not in {"spam", "newsletter"},
                "estimated_response_cost": estimate_response_cost(category, priority),
                "metadata": {
                    "synthetic": True,
                    "seed_id": seed_email.get("id"),
                },
            }
        )

    return variants


def generate_adversarial_email() -> Dict[str, Any]:
    templates = [
        {
            "sender": "security-team@alerts-company.com",
            "subject": "URGENT: Verify account immediately",
            "body": "Your mailbox may be disabled unless you verify credentials now. Click the secure link immediately.",
            "category": "phishing",
            "priority": "high",
        },
        {
            "sender": "promo@fastrewards-mail.com",
            "subject": "Last chance: urgent reward claim",
            "body": "Immediate action required to claim your company benefit. Submit payment info now.",
            "category": "spam",
            "priority": "medium",
        },
        {
            "sender": "ceo-office@external-support.net",
            "subject": "Need this done now",
            "body": "Please send the payroll sheet immediately. I am traveling. Do not call.",
            "category": "phishing",
            "priority": "critical",
        },
    ]

    chosen = random.choice(templates)
    chosen["id"] = f"adv-{uuid.uuid4().hex[:8]}"
    chosen["urgency_score"] = infer_urgency_score(chosen["priority"], chosen["subject"], chosen["body"])
    chosen["sender_importance"] = infer_sender_importance(chosen["sender"])
    chosen["deadline_steps"] = infer_deadline_steps(chosen["priority"], chosen["urgency_score"])
    chosen["adversarial"] = True
    chosen["spam_score"] = 0.8 if chosen["category"] == "spam" else 0.4
    chosen["phishing_risk"] = 0.95 if chosen["category"] == "phishing" else 0.5
    chosen["requires_response"] = False
    chosen["estimated_response_cost"] = 1.5
    chosen["metadata"] = {"synthetic": True, "adversarial_template": True}
    return chosen


def generate_spike_batch(count: int = 6) -> List[Dict[str, Any]]:
    """
    Generate a peak-hour mixed workload.
    """
    batch: List[Dict[str, Any]] = []
    for _ in range(count):
        roll = random.random()
        if roll < 0.2:
            batch.append(generate_adversarial_email())
        elif roll < 0.45:
            batch.append(generate_template_email("complaint", "high"))
        elif roll < 0.7:
            batch.append(generate_template_email("support", "medium"))
        elif roll < 0.85:
            batch.append(generate_template_email("billing", "medium"))
        else:
            batch.append(generate_template_email("newsletter", "low"))
    return batch


def generate_template_email(category: str, priority: str) -> Dict[str, Any]:
    sender_map = {
        "complaint": "customer@shopmail.com",
        "billing": "finance@client-enterprise.com",
        "support": "ops@company.com",
        "escalation": "ceo@company.com",
        "newsletter": "noreply@updates.com",
        "spam": "promo@deals-mail.com",
        "phishing": "security@account-alerts.net",
        "general": "user@example.com",
    }

    subject_map = {
        "complaint": "Damaged item received - need help",
        "billing": "Invoice discrepancy for March",
        "support": "Production issue needs investigation",
        "escalation": "Critical issue requiring immediate visibility",
        "newsletter": "Weekly product updates",
        "spam": "Exclusive offer expires today",
        "phishing": "Verify your credentials now",
        "general": "Following up on previous message",
    }

    body_map = {
        "complaint": "I received a damaged product and need a replacement as soon as possible.",
        "billing": "We noticed a mismatch in the billed amount and need clarification.",
        "support": "A customer-facing workflow is failing intermittently in production.",
        "escalation": "This issue is blocking a major client and needs urgent escalation.",
        "newsletter": "Here are the latest updates from our team this week.",
        "spam": "Claim your reward immediately by clicking the link.",
        "phishing": "Your account will be disabled unless you verify your password immediately.",
        "general": "Wanted to follow up and see if you had an update.",
    }

    sender = sender_map.get(category, "user@example.com")
    subject = subject_map.get(category, "Update")
    body = body_map.get(category, "Please review.")

    urgency_score = infer_urgency_score(priority, subject, body)
    return {
        "id": f"tpl-{uuid.uuid4().hex[:8]}",
        "sender": sender,
        "subject": subject,
        "body": body,
        "category": category,
        "priority": priority,
        "urgency_score": urgency_score,
        "sender_importance": infer_sender_importance(sender),
        "deadline_steps": infer_deadline_steps(priority, urgency_score),
        "adversarial": infer_adversarial(subject, body, category),
        "spam_score": 0.85 if category == "spam" else 0.1,
        "phishing_risk": 0.95 if category == "phishing" else 0.1,
        "requires_response": category not in {"spam", "newsletter"},
        "estimated_response_cost": estimate_response_cost(category, priority),
        "metadata": {"synthetic": True, "template_generated": True},
    }


def mutate_category(base_category: str) -> str:
    roll = random.random()
    if roll < 0.75:
        return base_category
    if base_category == "general":
        return random.choice(["support", "billing", "newsletter"])
    return random.choice(CATEGORY_POOL)


def mutate_priority(base_priority: str, category: str) -> str:
    if category == "escalation":
        return random.choice(["high", "critical"])
    if category in {"spam", "newsletter"}:
        return random.choice(["low", "medium"])
    if category in {"phishing", "complaint"}:
        return random.choice(["medium", "high", "critical"])
    return base_priority if random.random() < 0.7 else random.choice(PRIORITY_POOL)


def mutate_sender(sender: str, category: str) -> str:
    if category == "newsletter":
        return random.choice(["noreply@updates.com", "newsletter@vendor.com"])
    if category == "spam":
        return random.choice(["promo@offers-mail.com", "rewards@deal-center.com"])
    if category == "phishing":
        return random.choice(["security@account-alerts.net", "ceo-office@external-support.net"])
    if category == "escalation":
        return random.choice(["ceo@company.com", "cto@company.com", "vip-client@enterprise.com"])
    return sender


def mutate_subject(subject: str, category: str, priority: str) -> str:
    urgent_prefixes = ["URGENT:", "Immediate action:", "ASAP:", "Important:"]
    base = subject.strip() or "Please review"

    if category == "spam":
        choices = [
            "Exclusive offer expires now",
            "Claim your benefit today",
            "Limited time urgent reward",
        ]
        return random.choice(choices)

    if category == "phishing":
        choices = [
            "URGENT: Verify account immediately",
            "Password reset required now",
            "Security alert: action needed",
        ]
        return random.choice(choices)

    if priority in {"high", "critical"} and random.random() < 0.5:
        return f"{random.choice(urgent_prefixes)} {base}"

    return base


def mutate_body(body: str, category: str, priority: str) -> str:
    additions = {
        "complaint": [
            "The item arrived damaged and I need this resolved quickly.",
            "This is affecting my experience and I need help urgently.",
        ],
        "billing": [
            "Please clarify the invoice mismatch before end of day.",
            "This discrepancy impacts our approval workflow.",
        ],
        "support": [
            "The issue appears customer-facing and intermittent.",
            "Several users are blocked right now.",
        ],
        "phishing": [
            "Do not delay. Verify your credentials through the secure link now.",
            "Immediate confirmation is required to keep access active.",
        ],
        "spam": [
            "Click now to unlock your reward.",
            "Act immediately to avoid losing this offer.",
        ],
        "newsletter": [
            "This week's updates are attached below.",
            "Here are our latest product highlights.",
        ],
        "general": [
            "Checking in for an update whenever you have time.",
            "Please let me know when convenient.",
        ],
        "escalation": [
            "A major client is impacted and leadership visibility is required.",
            "This needs immediate coordination across teams.",
        ],
    }

    extra = random.choice(additions.get(category, ["Please review this update."]))
    if priority in {"high", "critical"}:
        extra += " This is time-sensitive."
    return f"{body.strip()} {extra}".strip()


def infer_sender_importance(sender: str) -> int:
    sender = sender.lower()
    if sender.startswith(("ceo@", "cto@", "finance@", "ops@", "vip-client@")):
        return 3
    if sender.endswith(("company.com", "enterprise.com", "client-enterprise.com")):
        return 2
    if "noreply" in sender or "promo" in sender:
        return 0
    return 1


def infer_adversarial(subject: str, body: str, category: str) -> bool:
    text = f"{subject} {body}".lower()
    suspicious_terms = [
        "click now",
        "verify account",
        "password",
        "credential",
        "free money",
        "reward",
        "act immediately",
    ]
    return category in {"phishing"} or any(term in text for term in suspicious_terms)


def infer_urgency_score(priority: str, subject: str, body: str) -> float:
    base = {
        "low": 0.2,
        "medium": 0.45,
        "high": 0.75,
        "critical": 0.95,
    }.get(priority, 0.3)

    text = f"{subject} {body}".lower()
    if any(token in text for token in ["urgent", "immediately", "asap", "time-sensitive", "end of day"]):
        base += 0.1
    if any(token in text for token in ["production", "system down", "client impacted", "blocked"]):
        base += 0.1

    return min(base, 1.0)


def infer_deadline_steps(priority: str, urgency_score: float) -> int:
    if priority == "critical" or urgency_score >= 0.9:
        return 1
    if priority == "high" or urgency_score >= 0.7:
        return 2
    if priority == "medium":
        return 4
    return 6


def estimate_response_cost(category: str, priority: str) -> float:
    cost = 0.5
    if category in {"complaint", "billing", "support"}:
        cost += 0.5
    if category in {"escalation"} or priority in {"high", "critical"}:
        cost += 1.0
    return cost