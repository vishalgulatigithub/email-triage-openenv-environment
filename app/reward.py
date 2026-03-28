from app.models import Reward


def compute_reward(prev_stage, new_stage, score, done=False, repeated=False):
    """
    Compute shaped reward for email triage environment.

    Returns:
        Reward (float subclass)
    """

    # ---------------------------
    # BASE REWARD FROM GRADER
    # ---------------------------
    reward = score * 0.5

    # ---------------------------
    # PROGRESS BONUS
    # ---------------------------
    if new_stage != prev_stage:
        reward += 0.2

    # ---------------------------
    # PENALTY FOR REPEATED ACTIONS
    # ---------------------------
    if repeated:
        reward -= 0.2

    # ---------------------------
    # BONUS FOR SUCCESSFUL COMPLETION
    # ---------------------------
    if done and score > 0.7:
        reward += 0.3

    # ---------------------------
    # PENALTY FOR BAD END
    # ---------------------------
    if done and score < 0.3:
        reward -= 0.3

    # ---------------------------
    # CLAMP BETWEEN 0 and 1
    # ---------------------------
    reward = max(0.0, min(reward, 1.0))

    # ✅ IMPORTANT: return float-based Reward
    return Reward(reward)