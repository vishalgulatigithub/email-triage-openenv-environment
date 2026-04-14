from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Category(str, Enum):
    spam = "spam"
    complaint = "complaint"
    billing = "billing"
    support = "support"
    escalation = "escalation"
    newsletter = "newsletter"
    phishing = "phishing"
    general = "general"


class ScheduleAction(str, Enum):
    reply = "reply"
    escalate = "escalate"
    defer = "defer"
    ignore = "ignore"


class Stage(str, Enum):
    triage = "triage"
    done = "done"


class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str

    # Ground truth / generator-provided labels
    category: Optional[str] = None
    priority: Optional[str] = None
    urgency_score: float = 0.0
    sender_importance: int = 0
    deadline_steps: Optional[int] = None

    # Environment realism
    adversarial: bool = False
    spam_score: float = 0.0
    phishing_risk: float = 0.0
    requires_response: bool = True
    estimated_response_cost: float = 0.0

    # Free-form metadata for future extension
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """
    New RL-style decision:
    1) classify category
    2) classify priority
    3) choose operational scheduling action
    4) optionally draft a response
    """
    email_id: Optional[str] = None
    classify_category: Optional[str] = None
    classify_priority: Optional[str] = None
    schedule_action: Optional[str] = None
    response_text: Optional[str] = None

    # Optional explainability hook for demo/debug
    rationale: Optional[str] = None


class RewardBreakdown(BaseModel):
    classification_score: float = 0.0
    scheduling_score: float = 0.0
    sla_score: float = 0.0
    safety_score: float = 0.0
    efficiency_score: float = 0.0
    backlog_penalty: float = 0.0
    escalation_penalty: float = 0.0
    total_score: float = 0.0


class HistoryItem(BaseModel):
    step: int
    email_id: str
    predicted_category: Optional[str] = None
    predicted_priority: Optional[str] = None
    schedule_action: Optional[str] = None
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """
    Rich environment state for RL / evaluation.
    """
    inbox: List[Email]
    current_email: Optional[Email] = None
    stage: str = Stage.triage.value

    # Operational state
    time_step: int = 0
    time_of_day: int = 9
    inbox_size: int = 0
    pending_urgent: int = 0
    spike_active: bool = False

    # Derived / extracted signals
    extracted: Dict[str, Any] = Field(default_factory=dict)

    # Recent history for memory / decision context
    history: List[HistoryItem] = Field(default_factory=list)

    # Metrics snapshot
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # Horizon info
    remaining_steps: int = 0


class Reward(float):
    pass