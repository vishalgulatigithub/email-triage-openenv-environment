import requests

BASE_URL = "http://localhost:7860"

def reset():
    response = requests.post(f"{BASE_URL}/reset")
    return response.json()

def step(action):
    response = requests.post(f"{BASE_URL}/step", json=action)
    return response.json()


if __name__ == "__main__":
    task_name = "EMAIL_TRIAGE"

    # START block
    print(f"[START] task={task_name}", flush=True)

    # Reset environment
    state = reset()

    total_steps = 0
    total_reward = 0
    max_steps = 20  # safety to avoid infinite loops

    done = False

    while not done and total_steps < max_steps:
        # 🔹 Basic action (you can improve logic later)
        action = {
            "action_type": "pick_email"
        }

        result = step(action)

        # Extract values safely
        reward = result.get("reward", 0)
        done = result.get("done", False)

        total_reward += reward
        total_steps += 1

        # STEP block (printed every iteration)
        print(f"[STEP] step={total_steps} reward={reward}", flush=True)

    # Final score (can be improved later)
    score = total_reward

    # END block
    print(f"[END] task={task_name} score={score} steps={total_steps}", flush=True)
