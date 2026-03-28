import requests

# Try multiple base URLs automatically
BASE_URLS = [
    "http://127.0.0.1:8000",   # local (uvicorn default)
    "http://localhost:8000",
    "http://localhost:7860",   # Docker / HuggingFace
    "http://0.0.0.0:7860"
]


def get_working_url():
    for url in BASE_URLS:
        try:
            res = requests.get(f"{url}/reset", timeout=2)
            if res.status_code == 200:
                return url
        except:
            continue
    return None


def run():
    base_url = get_working_url()

    if not base_url:
        print("❌ No working server found")
        return 0.0

    total_score = 0.0

    try:
        # RESET
        requests.get(f"{base_url}/reset")

        # STEP 1: analyze
        requests.post(f"{base_url}/step", json={
            "action_type": "analyze"
        })

        # STEP 2: classify
        requests.post(f"{base_url}/step", json={
            "action_type": "classify",
            "category": "complaint",
            "priority": "high"
        })

        # STEP 3: respond
        requests.post(f"{base_url}/step", json={
            "action_type": "respond",
            "response_text": "We are sorry, we will resolve this."
        })

        # STEP 4: finish
        res = requests.post(f"{base_url}/step", json={
            "action_type": "finish"
        })

        data = res.json()

        total_score = data.get("info", {}).get("score", 0.0)

    except Exception as e:
        print("❌ Baseline error:", str(e))
        return 0.0

    return total_score