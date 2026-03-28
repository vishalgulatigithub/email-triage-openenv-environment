import random
from app.state import EnvState
from app.memory import Memory
from app.features import extract_features
from app.graders import grade
from app.reward import compute_reward
from app.tasks import load_tasks
from app.models import Observation, Reward, Email


class EmailEnv:
    def __init__(self):
        self.tasks = load_tasks()
        self.state = EnvState()
        self.memory = Memory()
        self.last_score = 0.0
        self.max_steps = 5
        self.steps = 0

    # ---------------------------
    # RESET
    # ---------------------------
    def reset(self):
        email = random.choice(self.tasks)

        self.state.reset(email)
        self.memory.clear()
        self.steps = 0
        self.last_score = 0.0  # ✅ reset score

        return {
            "message": "Environment reset",
            "email": email
        }

    # ---------------------------
    # STEP (FINAL FIXED)
    # ---------------------------
    def step(self, action):
        email_data = self.state.current_email

        if not email_data:
            return (
                {"error": "Call /reset first"},
                Reward(0.0),
                True,
                {}
            )

        self.steps += 1

        # Convert dict → Email model
        email = Email(**email_data)

        # ---------------------------
        # FEATURE EXTRACTION
        # ---------------------------
        extracted = extract_features(email_data)

        prev_stage = self.state.stage

        # ---------------------------
        # ✅ GRADING FIX (CRITICAL)
        # ---------------------------
        if action.action_type == "finish":
            # 🔥 DO NOT recompute score
            score = self.last_score
        else:
            score = grade(email_data, action, extracted)
            self.last_score = score  # store for final step

        # ---------------------------
        # STAGE TRANSITIONS
        # ---------------------------
        if action.action_type == "analyze":
            new_stage = "analyzed"
        elif action.action_type == "classify":
            new_stage = "classified"
        elif action.action_type == "respond":
            new_stage = "responded"
        elif action.action_type == "finish":
            new_stage = "done"
        else:
            new_stage = prev_stage

        self.state.update_stage(new_stage)

        # ---------------------------
        # DONE CONDITION
        # ---------------------------
        done = False

        if action.action_type == "finish":
            done = True

        if self.steps >= self.max_steps:
            done = True

        # ---------------------------
        # REWARD (FIXED FLOW)
        # ---------------------------
        reward = compute_reward(
            prev_stage=prev_stage,
            new_stage=new_stage,
            score=score,
            done=done,
            repeated=False
        )

        # ---------------------------
        # MEMORY
        # ---------------------------
        self.memory.add(email_data, action.dict(), float(reward))

        # ---------------------------
        # OBSERVATION
        # ---------------------------
        observation = Observation(
            inbox=[email],
            current_email=email,
            stage=self.state.stage,
            history=self.memory.get_history() if hasattr(self.memory, "get_history") else [],
            extracted=extracted,
            remaining_steps=self.max_steps - self.steps
        )

        # ---------------------------
        # INFO
        # ---------------------------
        info = {
            "score": score,
            "steps": self.steps
        }

        return observation.dict(), reward, done, info

    # ---------------------------
    # STATE
    # ---------------------------
    def get_state(self):
        return {
            "email": self.state.current_email,
            "stage": self.state.stage,
            "steps": self.steps
        }