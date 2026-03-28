import json
import os
from datetime import datetime

LOG_FILE = "logs/episodes.json"

class Logger:
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.episode = []

    def log_step(self, obs, action, reward):
        self.episode.append({
            "timestamp": str(datetime.now()),
            "action": action.dict(),
            "reward": reward.score,
            "stage": obs.stage,
            "remaining_steps": obs.remaining_steps
        })

    def save(self):
        if not self.episode:
            return

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(self.episode)

        with open(LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)

        self.episode = []