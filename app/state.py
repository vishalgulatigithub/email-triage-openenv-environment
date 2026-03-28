class EnvState:
    
    def __init__(self, email=None):
        self.current_email = email
        self.stage = "start"
        self.steps = 0

    def reset(self, email):
        self.current_email = email
        self.stage = "start"
        self.steps = 0

    def update_stage(self, new_stage):
        self.stage = new_stage
        self.steps += 1