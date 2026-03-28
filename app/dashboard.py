import json
import os

LOG_FILE = "logs/episodes.json"


def get_stats():
    if not os.path.exists(LOG_FILE):
        return {
            "episodes": 0,
            "rewards": []
        }

    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)

        rewards = [episode.get("total_reward", 0) for episode in data]

        return {
            "episodes": len(data),
            "rewards": rewards
        }

    except Exception:
        return {
            "episodes": 0,
            "rewards": []
        }