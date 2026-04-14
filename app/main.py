from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from app.env import EmailEnv
from app.models import Action
from training.evaluate import compare_agents, evaluate_agent

app = FastAPI(title="Email Triage OpenEnv", version="2.0.0")

env = EmailEnv()


@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "2.0.0",
        "description": "Multi-agent RL environment for enterprise email triage under workload spikes, adversarial emails, and SLA pressure.",
        "endpoints": [
            "/reset",
            "/state",
            "/step",
            "/compare_agents",
            "/evaluate_agent",
            "/dashboard/view",
        ],
    }


@app.post("/reset")
def reset_env():
    state = env.reset()
    return {
        "message": "Environment reset successful",
        "state": state,
    }


@app.get("/state")
def get_state():
    return env.get_state()


@app.post("/step")
def step_env(action: Action):
    state, reward, done, info = env.step(action)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/evaluate_agent")
def api_evaluate_agent(
    agent_type: str = Query(..., description="random | rule | ppo"),
    num_episodes: int = Query(10, ge=1, le=100),
    checkpoint_path: str = Query("training/checkpoints/ppo_email_triage.pt"),
):
    result = evaluate_agent(
        agent_type=agent_type,
        num_episodes=num_episodes,
        checkpoint_path=checkpoint_path if agent_type == "ppo" else None,
    )
    return JSONResponse(content=result)


@app.get("/compare_agents")
def api_compare_agents(
    num_episodes: int = Query(10, ge=1, le=100),
    checkpoint_path: str = Query("training/checkpoints/ppo_email_triage.pt"),
):
    result = compare_agents(
        num_episodes=num_episodes,
        checkpoint_path=checkpoint_path,
    )
    return JSONResponse(content=result)


@app.get("/dashboard/view", response_class=HTMLResponse)
def dashboard_view():
    html = """
    <html>
    <head>
        <title>Email Triage RL Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background: #f7f7f7;
                color: #222;
            }
            .card {
                background: white;
                padding: 24px;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                margin-bottom: 20px;
            }
            code {
                background: #f0f0f0;
                padding: 2px 6px;
                border-radius: 6px;
            }
            h1, h2 {
                margin-top: 0;
            }
            ul {
                line-height: 1.8;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Email Triage RL Dashboard</h1>
            <p>
                Multi-agent reinforcement learning environment for enterprise email triage.
            </p>
            <ul>
                <li>Classifier agent: category + priority</li>
                <li>Scheduler agent: reply / escalate / defer / ignore</li>
                <li>Adversarial phishing and spam scenarios</li>
                <li>Workload spikes and SLA pressure</li>
            </ul>
        </div>

        <div class="card">
            <h2>API Endpoints</h2>
            <ul>
                <li><code>POST /reset</code></li>
                <li><code>GET /state</code></li>
                <li><code>POST /step</code></li>
                <li><code>GET /evaluate_agent?agent_type=rule&num_episodes=10</code></li>
                <li><code>GET /compare_agents?num_episodes=10</code></li>
            </ul>
        </div>

        <div class="card">
            <h2>Recommended Local Dashboard</h2>
            <p>Run the richer Streamlit dashboard locally:</p>
            <code>streamlit run visualization/dashboard.py</code>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)