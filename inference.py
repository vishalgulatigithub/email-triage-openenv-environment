import requests
import os
from openai import OpenAI

BASE_URL = "http://localhost:7860"

def reset():
    response = requests.post(f"{BASE_URL}/reset")
    return response.json()

def step(action):
    response = requests.post(f"{BASE_URL}/step", json=action)
    return response.json()


# ✅ Initialize LLM client using provided proxy
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)


def decide_action(state):
    """
    Use LLM to decide next action
    """

    prompt = f"""
    You are an email triage assistant.

    Given the current state:
    {state}

    Decide the next action.
    Return ONLY one action_type from:
    - pick_email
    - mark_important
    - archive
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # safe default via proxy
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    action_text = response.choices[0].message.content.strip()

    # Basic fallback safety
    if action_text not in ["pick_email", "mark_important", "archive"]:
        action_text = "pick_email"

    return {
        "action_type": action_text
    }


if __name__ == "__main__":
    task_name = "EMAIL_TRIAGE"

    # START block
    print(f"[START] task={task_name}", flush=True)

    state = reset()

    total_steps = 0
    total_reward = 0
    max_steps = 20
    done = False

    while not done and total_steps < max_steps:

        # ✅ LLM decides action
        action = decide_action(state)

        result = step(action)

        reward = result.get("reward", 0)
        done = result.get("done", False)

        total_reward += reward
        total_steps += 1

        # STEP block
        print(f"[STEP] step={total_steps} reward={reward}", flush=True)

        # update state
        state = result.get("state", {})

    score = total_reward

    # END block
    print(f"[END] task={task_name} score={score} steps={total_steps}", flush=True)
