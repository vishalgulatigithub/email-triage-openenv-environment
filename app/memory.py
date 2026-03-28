class Memory:
    def __init__(self):
        self.history = []

    def add(self, email, action, reward):
        self.history.append({
            "email": email,
            "action": action,
            "reward": reward
        })

    def get_recent(self, n=5):
        return self.history[-n:]

    def clear(self):
        self.history = []