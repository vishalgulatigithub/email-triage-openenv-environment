from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from app.features import extract_features
from app.graders import grade
from app.models import Action, Email, HistoryItem, Observation, Reward
from app.reward import compute_reward
from app.tasks import load_tasks


class EmailEnv:
    """
    Production-style email triage environment.

    Key upgrades over the original version:
    - Inbox queue instead of single email episode
    - Time progression and workload spikes
    - Pending urgent backlog
    - Adversarial / spam / phishing handling hooks
    - Multi-dimensional reward
    - Rich observation for RL training
    """

    def __init__(self) -> None:
        self.task_templates = load_tasks()

        # Episode controls
        self.max_steps = 30
        self.base_batch_size = 6
        self.spike_batch_size = (4, 8)
        self.spike_every_n_steps = 6
        self.max_history = 8

        self.reset()

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.steps = 0
        self.time_step = 0
        self.time_of_day = 9
        self.pending_urgent = 0
        self.spike_active = False

        self.history: List[HistoryItem] = []
        self.completed: List[Dict[str, Any]] = []

        self.metrics: Dict[str, Any] = {
            "emails_seen": 0,
            "emails_processed": 0,
            "urgent_handled": 0,
            "urgent_missed": 0,
            "spam_ignored": 0,
            "phishing_replied": 0,
            "unnecessary_escalations": 0,
            "sla_breaches": 0,
            "cumulative_reward": 0.0,
        }

        self.inbox: List[Dict[str, Any]] = self._generate_initial_batch()
        self.current_email: Dict[str, Any] | None = self._pop_next_email()

        return self._build_observation().dict()

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, action: Action) -> Tuple[Dict[str, Any], Reward, bool, Dict[str, Any]]:
        if self.current_email is None:
            observation = self._build_observation().dict()
            return observation, Reward(0.0), True, {"error": "No current email. Reset environment."}

        self.steps += 1
        self.metrics["emails_seen"] += 1

        email = Email(**self.current_email)
        extracted = extract_features(self.current_email)

        grading = grade(
            email_data=self.current_email,
            action=action,
            extracted=extracted,
            env_context=self._build_env_context(),
        )

        reward_value = compute_reward(
            grading=grading,
            env_context=self._build_env_context(),
            action=action,
            email_data=self.current_email,
        )

        reward_value = float(reward_value)
        self.metrics["cumulative_reward"] += reward_value

        self._apply_action_effects(email=email, action=action, grading=grading)
        self._record_history(email=email, action=action, reward_value=reward_value, grading=grading)

        self._advance_time()
        self._maybe_inject_spike()
        self._update_deadlines_and_backlog()

        self.current_email = self._pop_next_email()
        done = self._is_done()

        observation = self._build_observation()
        info = {
            "grading": grading,
            "metrics": self.metrics,
            "steps": self.steps,
            "time_of_day": self.time_of_day,
            "spike_active": self.spike_active,
        }

        return observation.dict(), Reward(reward_value), done, info

    # ------------------------------------------------------------------
    # STATE / OBSERVATION
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return self._build_observation().dict()

    def _build_observation(self) -> Observation:
        current_email_model = Email(**self.current_email) if self.current_email else None
        inbox_models = [Email(**email) for email in self.inbox]

        extracted = extract_features(self.current_email) if self.current_email else {}

        return Observation(
            inbox=inbox_models,
            current_email=current_email_model,
            stage="triage" if self.current_email else "done",
            time_step=self.time_step,
            time_of_day=self.time_of_day,
            inbox_size=len(self.inbox),
            pending_urgent=self.pending_urgent,
            spike_active=self.spike_active,
            extracted=extracted,
            history=self.history[-self.max_history :],
            metrics=self.metrics.copy(),
            remaining_steps=max(self.max_steps - self.steps, 0),
        )

    def _build_env_context(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "time_step": self.time_step,
            "time_of_day": self.time_of_day,
            "pending_urgent": self.pending_urgent,
            "inbox_size": len(self.inbox),
            "spike_active": self.spike_active,
            "metrics": self.metrics.copy(),
        }

    # ------------------------------------------------------------------
    # ACTION EFFECTS
    # ------------------------------------------------------------------
    def _apply_action_effects(self, email: Email, action: Action, grading: Dict[str, Any]) -> None:
        schedule_action = (action.schedule_action or "").lower()

        self.metrics["emails_processed"] += 1

        true_priority = (email.priority or "").lower()
        is_urgent = true_priority in {"high", "critical"} or email.urgency_score >= 0.7

        if is_urgent:
            if schedule_action in {"reply", "escalate"}:
                self.metrics["urgent_handled"] += 1
            else:
                self.metrics["urgent_missed"] += 1
                self.pending_urgent += 1

        if (email.category or "").lower() in {"spam", "phishing"}:
            if schedule_action == "ignore":
                self.metrics["spam_ignored"] += 1
            if schedule_action == "reply":
                self.metrics["phishing_replied"] += 1

        if schedule_action == "escalate" and not is_urgent and (email.category or "").lower() not in {"escalation"}:
            self.metrics["unnecessary_escalations"] += 1

        if grading.get("sla_score", 0.0) < 0:
            self.metrics["sla_breaches"] += 1

        self.completed.append(
            {
                "email_id": email.id,
                "action": action.dict(),
                "grading": grading,
            }
        )

    def _record_history(
        self,
        email: Email,
        action: Action,
        reward_value: float,
        grading: Dict[str, Any],
    ) -> None:
        self.history.append(
            HistoryItem(
                step=self.steps,
                email_id=email.id,
                predicted_category=action.classify_category,
                predicted_priority=action.classify_priority,
                schedule_action=action.schedule_action,
                reward=reward_value,
                done=False,
                info={
                    "grading": grading,
                    "subject": email.subject,
                },
            )
        )

    # ------------------------------------------------------------------
    # TIME / DYNAMICS
    # ------------------------------------------------------------------
    def _advance_time(self) -> None:
        self.time_step += 1
        self.time_of_day = (self.time_of_day + 1) % 24
        self.spike_active = False

    def _maybe_inject_spike(self) -> None:
        if self.steps == 0:
            return

        peak_hours = {9, 10, 11, 14, 15, 16}
        should_spike = (
            self.steps % self.spike_every_n_steps == 0
            or self.time_of_day in peak_hours and random.random() < 0.35
        )

        if not should_spike:
            return

        self.spike_active = True
        extra_count = random.randint(*self.spike_batch_size)
        self.inbox.extend(self._generate_batch(extra_count, spike=True))

    def _update_deadlines_and_backlog(self) -> None:
        urgent_pending = 0

        for email in self.inbox:
            if email.get("deadline_steps") is not None:
                email["deadline_steps"] -= 1

            priority = (email.get("priority") or "").lower()
            urgency_score = float(email.get("urgency_score", 0.0))

            if priority in {"high", "critical"} or urgency_score >= 0.7:
                urgent_pending += 1

        self.pending_urgent = urgent_pending

    # ------------------------------------------------------------------
    # EPISODE TERMINATION
    # ------------------------------------------------------------------
    def _is_done(self) -> bool:
        if self.steps >= self.max_steps:
            return True
        if self.current_email is None and len(self.inbox) == 0:
            return True
        return False

    # ------------------------------------------------------------------
    # EMAIL GENERATION
    # ------------------------------------------------------------------
    def _generate_initial_batch(self) -> List[Dict[str, Any]]:
        return self._generate_batch(self.base_batch_size, spike=False)

    def _generate_batch(self, count: int, spike: bool = False) -> List[Dict[str, Any]]:
        batch: List[Dict[str, Any]] = []
        for i in range(count):
            template = random.choice(self.task_templates)
            email = self._materialize_email(template, index=i, spike=spike)
            batch.append(email)

        # Higher urgency first for realism, but not fully sorted
        batch.sort(
            key=lambda x: (
                x.get("priority") not in {"critical", "high"},
                -float(x.get("urgency_score", 0.0)),
            )
        )
        return batch

    def _materialize_email(self, template: Dict[str, Any], index: int, spike: bool) -> Dict[str, Any]:
        email = dict(template)

        sender = email.get("sender", "unknown@example.com").lower()
        subject = email.get("subject", "")
        body = email.get("body", "")

        priority = (email.get("priority") or "low").lower()
        category = (email.get("category") or "general").lower()

        sender_importance = self._infer_sender_importance(sender)
        adversarial = self._infer_adversarial(subject, body, category)
        phishing_risk = 0.9 if category == "phishing" else (0.7 if adversarial and category == "spam" else 0.1)
        spam_score = 0.9 if category == "spam" else (0.7 if adversarial else 0.1)

        urgency_score = self._infer_urgency_score(priority, subject, body, spike=spike)
        deadline_steps = self._infer_deadline_steps(priority, urgency_score)

        generated_id = f"{email.get('id', 'email')}-{self.time_step}-{index}-{random.randint(1000, 9999)}"

        return {
            "id": generated_id,
            "sender": sender,
            "subject": subject,
            "body": body,
            "category": category,
            "priority": priority,
            "urgency_score": urgency_score,
            "sender_importance": sender_importance,
            "deadline_steps": deadline_steps,
            "adversarial": adversarial,
            "spam_score": spam_score,
            "phishing_risk": phishing_risk,
            "requires_response": category not in {"spam", "newsletter"},
            "estimated_response_cost": 2.0 if priority in {"high", "critical"} else 0.5,
            "metadata": {
                "spike_generated": spike,
            },
        }

    def _pop_next_email(self) -> Dict[str, Any] | None:
        if not self.inbox:
            return None
        return self.inbox.pop(0)

    # ------------------------------------------------------------------
    # HEURISTICS
    # ------------------------------------------------------------------
    def _infer_sender_importance(self, sender: str) -> int:
        high_value_domains = {"company.com", "enterprise.com", "partner.com"}
        high_value_users = {"ceo@", "cto@", "ops@", "finance@"}

        if any(sender.startswith(prefix) for prefix in high_value_users):
            return 3

        if "@" in sender:
            domain = sender.split("@")[-1]
            if domain in high_value_domains:
                return 2

        if "noreply" in sender or "promo" in sender:
            return 0

        return 1

    def _infer_adversarial(self, subject: str, body: str, category: str) -> bool:
        text = f"{subject} {body}".lower()
        suspicious_tokens = ["click now", "verify account", "free money", "limited time", "password reset"]
        looks_suspicious = any(token in text for token in suspicious_tokens)
        return category in {"phishing"} or looks_suspicious

    def _infer_urgency_score(self, priority: str, subject: str, body: str, spike: bool) -> float:
        base_map = {
            "low": 0.2,
            "medium": 0.45,
            "high": 0.8,
            "critical": 0.95,
        }
        score = base_map.get(priority, 0.3)

        text = f"{subject} {body}".lower()
        if "urgent" in text or "immediately" in text or "asap" in text:
            score += 0.1
        if "production outage" in text or "system down" in text:
            score += 0.1
        if spike:
            score += 0.05

        return min(score, 1.0)

    def _infer_deadline_steps(self, priority: str, urgency_score: float) -> int:
        if priority == "critical" or urgency_score >= 0.9:
            return 1
        if priority == "high" or urgency_score >= 0.7:
            return 2
        if priority == "medium":
            return 4
        return 6