def grade(email, action, extracted):
    text = (email["subject"] + " " + email["body"]).lower()

    score = 0.0

    # detect spam
    is_spam = any(word in text for word in ["offer", "buy", "sale", "discount"])

    if action.action_type == "classify":
        if is_spam and action.category == "spam":
            score += 0.7
        elif not is_spam and action.category == "important":
            score += 0.7

    if action.action_type == "respond" and action.response_text:
        score += 0.2

    if action.action_type == "finish":
        score += 0.1

    return min(score, 1.0)