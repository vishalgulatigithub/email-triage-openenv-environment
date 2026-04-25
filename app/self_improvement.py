from __future__ import annotations

from collections import deque
from typing import Dict, List


class CurriculumTracker:
    """
    Minimal self-improvement tracker.

    Important:
    This does NOT make the environment harder during training.
    It only tracks what difficulty level the agent is ready for.

    This avoids PPO collapse while still showing a self-improvement flow:
    agent performance -> curriculum level -> harder scenario recommendation.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.recent_safe_scores = deque(maxlen=window_size)
        self.level = 1

    def update(self, metrics: Dict) -> int:
        urgent_handled = float(metrics.get("urgent_handled", 0))
        urgent_missed = float(metrics.get("urgent_missed", 0))
        sla_breaches = float(metrics.get("sla_breaches", 0))

        safe_score = urgent_handled - urgent_missed - sla_breaches
        self.recent_safe_scores.append(safe_score)

        avg_safe_score = sum(self.recent_safe_scores) / len(self.recent_safe_scores)

        # Conservative thresholds: curriculum increases only after stable success.
        if avg_safe_score >= 8:
            self.level = 3
        elif avg_safe_score >= 3:
            self.level = 2
        else:
            self.level = 1

        return self.level

    def get_recommended_challenges(self) -> List[str]:
        if self.level == 1:
            return [
                "standard inbox triage",
                "basic complaint/support/billing emails",
                "normal SLA deadlines",
            ]

        if self.level == 2:
            return [
                "moderate workload spikes",
                "more high-priority emails",
                "tighter SLA deadlines",
            ]

        return [
            "adversarial phishing emails",
            "heavy workload spikes",
            "critical SLA pressure",
            "mixed spam and urgent emails",
        ]