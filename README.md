# 📧 Email Triage OpenEnv Environment

An intelligent **Email Triage Environment** built using FastAPI, designed for reinforcement learning agents to process, classify, and respond to emails efficiently.

---

## 🚀 Features

* 📥 Multi-email inbox simulation
* 🧠 Feature extraction from email content
* 🏷️ Email classification (category + priority)
* ✉️ Response generation support
* 🎯 Reward shaping for RL agents
* 📊 Logging + Visualization Dashboard
* 🤖 Baseline agent included
* ⚡ FastAPI-based API (OpenEnv compliant)

---

## 🧩 Environment Design

The environment simulates a real-world email workflow:

1. Pick an email
2. Analyze content
3. Classify category & priority
4. Respond (optional)
5. Finish task

---

## 📡 API Endpoints

| Endpoint          | Method | Description        |
| ----------------- | ------ | ------------------ |
| `/reset`          | GET    | Reset environment  |
| `/step`           | POST   | Perform action     |
| `/state`          | GET    | Get current state  |
| `/tasks`          | GET    | Task definitions   |
| `/grader`         | GET    | Get score          |
| `/baseline`       | GET    | Run baseline agent |
| `/dashboard`      | GET    | JSON stats         |
| `/dashboard/view` | GET    | Visualization UI   |

---

## ⚙️ Action Schema

```json
{
  "action_type": "pick_email | analyze | classify | respond | finish",
  "email_id": "string",
  "category": "string",
  "priority": "string",
  "response_text": "string"
}
```

---

## 🎯 Reward Logic

Reward is based on:

* Correct workflow progression
* Classification accuracy
* Avoiding repeated actions
* Efficient task completion

---

## 🧪 Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload
```

---

## 🔐 Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## 📊 Dashboard

View reward trends:

```
http://localhost:8000/dashboard/view
```

---

## 🤖 Baseline Agent

Run baseline:

```
http://localhost:8000/baseline
```

---

## 🏗️ Project Structure

```
app/
├── main.py
├── env.py
├── models.py
├── tasks.py
├── reward.py
├── graders.py
├── features.py
├── memory.py
├── state.py
├── logger.py
├── dashboard.py

baseline/
└── run_baseline.py

data/
└── emails.json
```

---

## ☁️ Deployment

* Docker-ready
* Hugging Face Spaces compatible
* OpenEnv compliant

---

## 🏆 Highlights

* Modular architecture
* Clean API design
* RL-friendly reward shaping
* Real-world inspired workflow

---
## 👨Working Api's for testing 

🔹 /reset
curl http://127.0.0.1:8000/reset

✔ Must return email + message

🔹 2. ANALYZE
curl -X POST http://127.0.0.1:8000/step \
-H "Content-Type: application/json" \
-d '{"action_type": "analyze"}'

🔹 3. CLASSIFY (VERY IMPORTANT)
curl -X POST http://127.0.0.1:8000/step \
-H "Content-Type: application/json" \
-d '{
  "action_type": "classify",
  "category": "spam",
  "priority": "low"
}'

👉 THIS decides your score

🔹 4. RESPOND
curl -X POST http://127.0.0.1:8000/step \
-H "Content-Type: application/json" \
-d '{
  "action_type": "respond",
  "response_text": "This looks like spam and will be ignored."
}'

🔹 5. FINISH
curl -X POST http://127.0.0.1:8000/step \
-H "Content-Type: application/json" \
-d '{"action_type": "finish"}'


🔹 6. STATE (/state)
curl http://127.0.0.1:8000/state

✔ Must return current state

🔹7.  /tasks ⭐ (VERY IMPORTANT)
curl http://127.0.0.1:8000/tasks

✔ Must return:

task list (3 tasks minimum)
action schema

🔹8.  /grader
curl http://127.0.0.1:8000/grader

✔ Must return score between 0.0 – 1.0



🧪 TEST
curl http://127.0.0.1:8000/baseline

## 👨‍💻 Author

Built for OpenEnv evaluation 🚀
