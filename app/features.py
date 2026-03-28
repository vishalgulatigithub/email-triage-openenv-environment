def extract_features(email):
    text = (email["subject"] + " " + email["body"]).lower()

    urgency = 0.0
    if "urgent" in text or "asap" in text:
        urgency += 0.6
    if "down" in text:
        urgency += 0.4

    sentiment = "negative" if "broken" in text or "damaged" in text else "neutral"

    return {
        "urgency": urgency,
        "sentiment": sentiment
    }