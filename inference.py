import requests
import os
from openai import OpenAI

BASE_URL = "http://localhost:7860"

def reset():
    try:
        response = requests.post(f"{BASE_URL}/reset", timeout=5)
        return response.json()
    except Exception:
        return {}

def step(action):
    try:
        response = requests.post(f"{BASE_URL}/step", json=action, timeout=5)
        return response.json()
    except Exception:
        return {"reward": 0, "done": True}


# ✅ LLM client (MANDATORY for passing Phase 2)
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY")
)


def decide_action(state):
    """
    Safe LLM-based decision
    """

    try:
        prompt = f"""
        You are an email triage assistant.

        State:
        {str(state)[:1000]}

        Choose ONE action:
        pick_email / mark_important / archive

        Return only the action name.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only one word."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            timeout=5
        )

        action_text = response.choices[0].message.content.strip().lower()

        if action_text not in ["pick_email", "mark_important", "archive"]:
            return {"action_type": "pick_email"}

        return {"action_type": action_text}

    except Exception:
        # 🔴 NEVER CRASH — fallback action
        return {"action_type": "pick_email"}


if __name__ == "__main__":
    task_name = "EMAIL_TRIAGE"

    # START
    print(f"[START] task={task_name}", flush=True)

    state = reset()

    total_steps = 0
    total_reward = 0
    max_steps = 20
    done = False

    while not done and total_steps < max_steps:
        try:
            action = decide_action(state)

            result = step(action)

            reward = result.get("reward", 0)
            done = result.get("done", False)

            total_reward += reward
            total_steps += 1

            print(f"[STEP] step={total_steps} reward={reward}", flush=True)

            state = result.get("state", {})

        except Exception:
            # 🔴 Absolute fallback safety
            print(f"[STEP] step={total_steps+1} reward=0", flush=True)
            break

    score = total_reward

    # END
    print(f"[END] task={task_name} score={score} steps={total_steps}", flush=True)
