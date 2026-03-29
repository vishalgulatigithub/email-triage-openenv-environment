import requests

BASE_URL = "http://localhost:7860"

def reset():
    response = requests.post(f"{BASE_URL}/reset")
    return response.json()

def step(action):
    response = requests.post(f"{BASE_URL}/step", json=action)
    return response.json()


if __name__ == "__main__":
    # Simple test run
    state = reset()
    print("Reset:", state)

    action = {
        "action_type": "pick_email"
    }

    result = step(action)
    print("Step:", result)