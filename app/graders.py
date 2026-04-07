def grade(email, action, extracted):
    text = (email["subject"] + " " + email["body"]).lower()

    score = 0.0

    # detect spam
    is_spam = any(word in text for word in ["offer", "buy", "sale", "discount"])

    # ---------------------------
    # CLASSIFICATION
    # ---------------------------
    if action.action_type == "classify":
        if is_spam and getattr(action, "category", None) == "spam":
            score += 0.6
        elif not is_spam and getattr(action, "category", None) == "important":
            score += 0.6
        else:
            score += 0.2  # partial credit

    # ---------------------------
    # RESPONSE
    # ---------------------------
    if action.action_type == "respond" and getattr(action, "response_text", None):
        score += 0.2

    # ---------------------------
    # FINISH BONUS
    # ---------------------------
    if action.action_type == "finish":
        score += 0.1

    # ---------------------------
    # 🔥 FINAL CLAMP (STRICT RANGE)
    # ---------------------------
    if score <= 0.0:
        score = 0.1
    elif score >= 1.0:
        score = 0.9

    return score
