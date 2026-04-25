---
title: Email Triage OpenEnv
emoji: 📬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

🌍 Problem Story: Email Triage at Internet Scale

Every day, billions of emails are sent across the internet.

Some are newsletters.
Some are customer complaints.
Some are critical production alerts.
And some are malicious phishing attacks.

Now imagine a large enterprise receiving millions of emails daily.

Within that flood:

A customer is waiting for a refund
A production system is down
A phishing attack is targeting employees

Traditional systems treat these emails as just messages to classify.

❗ The Real Problem

Most systems focus on:

Spam detection
Email classification
Rule-based routing

But the real question is:

What action should be taken on this email right now — given system state, urgency, and backlog?

This is not classification.
This is decision-making under constraints.

🧠 Our Approach

We model email triage as a multi-agent reinforcement learning system:

A classifier agent understands the email
A scheduler agent decides what action to take

The system learns policies, not just predictions.

⚙️ System Design
🧩 Classifier Agent
Predicts category (support, complaint, spam, phishing)
Estimates urgency (low → critical)
🧩 Scheduler Agent
Reply
Escalate
Defer
Ignore
🌪️ Environment Simulation

We simulate real-world complexity:

📥 Dynamic inbox queues
⏱️ SLA deadlines
📈 Workload spikes
⚠️ Adversarial emails (phishing/spam)
📊 Backlog pressure

This creates a long-horizon decision problem.

🤖 Reinforcement Learning (PPO)

We train a PPO agent to:

Learn from environment interaction
Optimize sequential decisions
Balance safety, efficiency, and SLA compliance
🔁 Self-Improvement (Curriculum Learning)

We implement adaptive curriculum tracking:

Agent performance → curriculum level
Stable performance → harder scenarios
Poor performance → easier scenarios

The system evolves with the agent.

📊 Evaluation Metrics

We evaluate agents on operational metrics, not just accuracy.

🔑 Core Metrics
Metric	Description
Total Reward	Overall performance
Urgent Handled	Critical emails handled correctly
Urgent Missed	Critical emails ignored ❌
SLA Breaches	Deadline violations ❌
Spam Ignored	Correct spam handling
Phishing Replied	Dangerous behavior ❌
Safe Policy Score	Safety-focused evaluation
📈 Example Results
<img width="1904" height="1015" alt="image" src="https://github.com/user-attachments/assets/41e76d69-91f4-448d-8597-cbf7426a1e26" />

<img width="1907" height="1017" alt="image" src="https://github.com/user-attachments/assets/738e34e4-489b-453d-8e78-052a2696dda2" />

<img width="1886" height="1036" alt="image" src="https://github.com/user-attachments/assets/00d95172-6d27-4a43-a170-0b61eafe4add" />

Plot Rewards
<img width="1197" height="591" alt="image" src="https://github.com/user-attachments/assets/cef3fe41-bb6c-41fc-b30a-8485e45c24a5" />

Plot Curriculum
<img width="1194" height="618" alt="image" src="https://github.com/user-attachments/assets/8edcb707-383e-4877-9950-27be9720bccd" />

Compare Agents
<img width="986" height="699" alt="image" src="https://github.com/user-attachments/assets/1a606c1d-a87e-4d47-aed2-27f65bd7e0b1" />
<img width="1005" height="708" alt="image" src="https://github.com/user-attachments/assets/dceff28a-5ba8-40d4-badf-e73f132258c5" />
<img width="988" height="701" alt="image" src="https://github.com/user-attachments/assets/6c35ec49-8c33-49e3-b561-67e763bc91f9" />
<img width="988" height="711" alt="image" src="https://github.com/user-attachments/assets/eed4cbf1-f8f6-4423-8d38-66487295ebca" />

Training Evaluation Metrics

<img width="1760" height="1011" alt="image" src="https://github.com/user-attachments/assets/0c9a4f85-a127-46bb-8d36-9aacdd08e267" />

Streamlit Dashboard

<img width="1863" height="1020" alt="image" src="https://github.com/user-attachments/assets/c3b64741-b722-47ef-8d1b-8e27fdc11ca4" />

<img width="1919" height="1003" alt="image" src="https://github.com/user-attachments/assets/a2ecf12f-5ad3-44a0-8022-fb14ca513bc8" />

<img width="1855" height="1031" alt="image" src="https://github.com/user-attachments/assets/089806e2-bca2-49b0-9af9-a359d2409cd6" />

<img width="1883" height="1020" alt="image" src="https://github.com/user-attachments/assets/59ab0a2c-7c0d-4a45-8094-bb7bb1a28f58" />


RUNNING DATA PIPELINE

## 🚀 Running Data Pipeline

> ⚠️ Make sure your virtual environment is activated before running the commands.

```bash
# 1. Run baseline
python -m baseline.run_baseline

# 2. Train PPO
python -m training.train_ppo

# 3. Generate reward plot
python visualization/plot_rewards.py

# 4. Generate curriculum plot (self-improvement)
python visualization/plot_curriculum.py

# 5. Evaluate agents
python -m training.evaluate

# 6. Compare agents
python -m visualization.compare_agents

# 7. Launch dashboard
streamlit run visualization/dashboard.py



Agent	Total Reward	Urgent Handled	Urgent Missed	SLA Breaches
Random
Rule-Based	
PPO 

🧠 Key Insight
Rule-based systems maximize reward (efficiency)
PPO learns safety-first policy

👉 PPO:

Handles more urgent emails
Reduces critical misses to zero
Maintains SLA compliance
⚖️ Trade-Off Observed

The PPO agent:

✅ Prioritizes critical emails
⚠️ Slightly over-replies (safety bias)

This reflects real-world systems prioritizing risk reduction over efficiency

📉 Training Metrics
🎯 PPO Learning Curve
Episodes: 400+
Reward trend: Increasing moving average
Variance: High early → stabilizes later

👉 Observed behavior:

Early training: unstable decisions
Mid training: learns urgency patterns
Late training: consistent policy
🔁 Curriculum Progression
Starts at Level 1 (easy)
Gradually increases to Level 3 (hard)
Stabilizes as agent performance improves

👉 This shows:

Self-improving learning progression without collapse

📊 Training Artifacts

Generated during training:

episode_rewards.csv → reward per episode
curriculum_log.txt → difficulty progression
reward curve plots
curriculum progression plots
📈 Visualizations
PPO Learning Curve

(Add image here)

Curriculum Progression

(Add image here)

Agent Comparison Dashboard

(Add image here)

🏁 Conclusion

At scale, email triage is not about classification.

It is about:

prioritizing correctly
acting under constraints
managing risk

This system demonstrates that:

RL can model real-world operational decision-making
Multi-agent systems improve decision quality
Curriculum learning enables self-improvement
💥 Final Takeaway

When billions of emails are flowing,
the problem isn’t understanding them —
it’s deciding what matters most.

## Key Result

Our PPO agent reduces:

- Urgent misses by ~95%
- SLA breaches by ~95%

compared to baseline systems, while maintaining competitive reward.

This demonstrates learning of **safe operational policies under real-world constraints**.

# Multi-Agent RL for Email Triage

A benchmark-style reinforcement learning environment for enterprise email triage under:

- workload spikes
- adversarial spam and phishing emails
- SLA pressure
- backlog constraints
- operational cost trade-offs

## Problem Framing

This project is **not** just email classification.

It models **sequential decision-making under operational constraints**:
- classify the email correctly
- estimate urgency correctly
- choose the best operational action:
  - reply
  - escalate
  - defer
  - ignore

## Multi-Agent Design

### Classifier Agent
Predicts:
- category
- priority

### Scheduler Agent
Uses:
- classifier output
- backlog state
- pending urgent count
- spike state
- deadline pressure
- adversarial risk

Then chooses:
- reply
- escalate
- defer
- ignore

## Key Features

- Queue-based inbox environment
- Multi-step RL decision process
- Adversarial phishing/spam scenarios
- Dynamic workload spikes
- SLA-aware reward shaping
- Random, rule-based, and PPO agents
- Reward visualization and evaluation dashboard
- 500+ synthetic scenarios generated from seed emails

## Repository Structure

```text
app/
  env.py
  features.py
  generator.py
  graders.py
  main.py
  models.py
  reward.py
  tasks.py

agents/
  classifier_agent.py
  scheduler_agent.py
  rule_based.py
  random_agent.py

training/
  state_encoder.py
  train_ppo.py
  evaluate.py

demo/
  run_demo.py
  demo_script.md

visualization/
  plot_rewards.py
  dashboard.py

data/
  emails.json


Postman Friendly Curls

1. START SERVER

uvicorn app.main:app --reload


2. RESET

curl -X POST http://127.0.0.1:8000/reset


3. STATE

curl http://127.0.0.1:8000/state


4. Manual step (simulate agent)

curl -X POST http://127.0.0.1:8000/step
-H "Content-Type: application/json"
-d "{\"email_id\": \"seed-001\", \"classify_category\": \"complaint\", \"classify_priority\": \"high\", \"schedule_action\": \"reply\", \"response_text\": \"We are resolving your issue.\"}"


Shows:

reward
grading
metrics update


5. EVALUATE RULE AGENT

curl "http://127.0.0.1:8000/evaluate_agent?agent_type=rule&num_episodes=5"


6.Evaluate PPO agent

curl "http://127.0.0.1:8000/evaluate_agent?agent_type=ppo&num_episodes=5"


7. Compare all agents (IMPORTANT)

curl "http://127.0.0.1:8000/compare_agents?num_episodes=10"



🧠 Optional: output (Windows PowerShell)

curl "http://127.0.0.1:8000/compare_agents?num_episodes=10" | ConvertFrom-Json

RUNNING DATA PIPELINE

python -m baseline.run_baseline
python -m training.train_ppo
python visualization/plot_rewards.py
python visualization/plot_curriculum.py
python -m training.evaluate
python -m visualization.compare_agents
streamlit run visualization/dashboard.py
