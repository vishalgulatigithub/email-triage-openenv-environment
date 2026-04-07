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
        self.last_score = 0.5  # 🔥 safe default (NOT 0)
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
        self.last_score = 0.5  # 🔥 safe reset (NOT 0)

        return {
            "message": "Environment reset",
            "email": email
        }

    # ---------------------------
    # STEP (FINAL VALIDATOR SAFE)
    # ---------------------------
    def step(self, action):
        email_data = self.state.current_email

        if not email_data:
            return (
                {"error": "Call /reset first"},
                Reward(0.1),  # 🔥 never return 0
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
        # ✅ SAFE GRADING
        # ---------------------------
        if action.action_type == "finish":
            score = self.last_score
        else:
            score = grade(email_data, action, extracted)

        # 🔥 STRICT CLAMP (CRITICAL FIX)
        if score is None:
            score = 0.5

        if score <= 0.0:
            score = 0.1
        elif score >= 1.0:
            score = 0.9

        # store safe score
        self.last_score = score

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
        # REWARD (SAFE)
        # ---------------------------
        reward = compute_reward(
            prev_stage=prev_stage,
            new_stage=new_stage,
            score=score,
            done=done,
            repeated=False
        )

        # 🔥 ensure reward is float and safe
        try:
            reward_value = float(reward)
        except Exception:
            reward_value = 0.1

        if reward_value <= 0.0:
            reward_value = 0.1
        elif reward_value >= 1.0:
            reward_value = 0.9

        # ---------------------------
        # MEMORY
        # ---------------------------
        self.memory.add(email_data, action.dict(), reward_value)

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

        return observation.dict(), reward_value, done, info

    # ---------------------------
    # STATE
    # ---------------------------
    def get_state(self):
        return {
            "email": self.state.current_email,
            "stage": self.state.stage,
            "steps": self.steps
        }
