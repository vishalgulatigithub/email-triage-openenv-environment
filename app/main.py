from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import os
from app.env import EmailEnv
from app.models import Action
from app.dashboard import get_stats

# Load environment variables
load_dotenv()

app = FastAPI(title="Email Triage OpenEnv")

# Initialize environment
env = EmailEnv()



@app.get("/")
def health():
    return {"status": "ok"}


# ---------------------------
# RESET
# ---------------------------
@app.post("/reset")
@app.get("/reset")
def reset():
    return env.reset()


# ---------------------------
# STEP
# ---------------------------
@app.post("/step")
def step(action: Action):
    try:
        result = env.step(action)
        print("DEBUG:", result)
        obs, reward, done, info = result
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# STATE
# ---------------------------
@app.get("/state")
def get_state():
    return env.get_state()


# ---------------------------
# TASKS (VALIDATOR IMPORTANT)
# ---------------------------
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            "email_triage_easy",
            "email_triage_medium",
            "email_triage_hard"
        ],
        "action_schema": {
            "action_type": [
                "pick_email",
                "analyze",
                "classify",
                "respond",
                "finish"
            ],
            "fields": {
                "email_id": "string",
                "category": "string",
                "priority": "string",
                "response_text": "string"
            }
        }
    }


# ---------------------------
# GRADER
# ---------------------------
@app.get("/grader")
def grader():
    return {
        "score": getattr(env, "last_score", 0.0)
    }


# ---------------------------
# BASELINE (RUN SCRIPT)
# ---------------------------
@app.get("/baseline")
def baseline():
    try:
        from baseline.run_baseline import run
        score = run()
        return {"baseline_score": score}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# DASHBOARD (JSON)
# ---------------------------
@app.get("/dashboard")
def dashboard():
    return get_stats()


# ---------------------------
# DASHBOARD UI (CHART)
# ---------------------------
@app.get("/dashboard/view", response_class=HTMLResponse)
def dashboard_view():
    return """
    <html>
    <head>
        <title>OpenEnv Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h2>📊 Email Triage Dashboard</h2>
        <canvas id="rewardChart"></canvas>

        <script>
        fetch('/dashboard')
        .then(res => res.json())
        .then(data => {
            const ctx = document.getElementById('rewardChart');

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.rewards.map((_, i) => i + 1),
                    datasets: [{
                        label: 'Reward per Episode',
                        data: data.rewards,
                        borderWidth: 2
                    }]
                }
            });
        });
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)