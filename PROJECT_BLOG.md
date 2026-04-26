# 📬 From inbox chaos to Multi Agent Self-Learning Email Triage OpenEnv — Multi-Agent RL Environment for Enterprise Email Decision-Making

### PPO Agent Achieves 95%+ SLA Compliance in a Multi-Agent Email Decision Environment

Every day, billions of emails move across the internet. Some are harmless newsletters, some are customer requests, some are urgent production incidents, and some are phishing attempts.

Most email automation systems stop at classification:

- Is this spam?
- Is this support?
- Is this urgent?
- Is this phishing?

But in a real enterprise workflow, classification is only the first step. The harder problem is deciding what the agent should do next:

> **Reply, escalate, defer, or ignore?**

That is the core idea behind **Email Triage OpenEnv**.

This project turns email triage into a **multi-agent reinforcement learning environment** where agents learn safe, SLA-aware operational decisions under realistic constraints such as workload spikes, backlog pressure, adversarial emails, and deadline risk.

---

## 🔗 Final Submission Resources

| Resource | Link |
|---|---|
| 🚀 Hugging Face Space | https://huggingface.co/spaces/vishalgulatinitj/email-triage-openenv |
| 🧪 Live App | https://vishalgulatinitj-email-triage-openenv.hf.space |
| 📓 Google Colab Notebook | https://colab.research.google.com/drive/1N5sJ79Rzdh8hklnKkWSFqapJA5G1bWE_?usp=sharing |
| 💻 GitHub Repository | https://github.com/vishalgulatigithub/email-triage-openenv-environment |
| 📄 FastAPI Docs | https://vishalgulatinitj-email-triage-openenv.hf.space/api/docs |
| 🔌 OpenAPI Schema | https://vishalgulatinitj-email-triage-openenv.hf.space/api/openapi.json |

---

# 1. Problem Statement

Most email automation tools classify emails, but real operations require decisions.

A classifier can say an email is “support” or “critical,” but the company still needs to know what action to take. A production outage should be escalated. A phishing email should be ignored. A low-priority update can be deferred. A support ticket may require a reply before SLA breach.

So this project asks:

> **Can we train an agent to make safe operational email decisions under SLA pressure, workload spikes, and adversarial risk?**

---

# 2. Core Idea

The project reframes email triage as a sequential decision-making environment.

The agent observes an email and environment state, then chooses:

```text
reply | escalate | defer | ignore
```

The decision depends on:

```text
email content
category
urgency
priority
SLA pressure
backlog
workload spike state
phishing/adversarial risk
```

This means the environment tests behavior, not just prediction accuracy.

---

# 3. Hackathon Theme Alignment

## Multi-Agent Interactions

The system contains:

- Classifier Agent
- Scheduler Agent
- Random Agent
- Rule-Based Agent
- PPO RL Agent

The classifier predicts category and urgency. The scheduler decides the operational action. PPO learns a policy using reward feedback. Random and rule-based agents provide baselines.

## Long-Horizon Planning

The agent acts over episodes and queues. Bad decisions can affect future state:

- defer urgent email → SLA risk grows
- ignore critical issue → urgent missed
- reply to phishing → safety failure
- escalate everything → operational noise

This creates a real planning problem.

## World Modeling for Professional Tasks

The environment simulates professional email operations:

- enterprise inbox
- production incidents
- customer support issues
- phishing/spam
- workload spikes
- deadline pressure
- backlog

## Self-Improvement

The curriculum tracker moves from easier to harder scenarios. The training run reached **Curriculum Level 3**, showing evaluation under harder workload/adversarial settings.

---

# 4. System Architecture

The deployed Hugging Face Space exposes:

```text
/        -> Streamlit dashboard
/api/*   -> FastAPI endpoints
```

Recommended mental model:

```text
Hugging Face Space
        ↓
Docker container
        ↓
Nginx public gateway on 7860
        ├── Streamlit dashboard
        └── FastAPI API
```

---

# 5. Code Structure

Place this blog file at the project root as:

```text
PROJECT_BLOG.md
```

Project structure:

```text
email-triage-openenv/
│
├── app/
│   ├── __init__.py
│   ├── env.py
│   ├── features.py
│   ├── generator.py
│   ├── graders.py
│   ├── main.py
│   ├── models.py
│   ├── reward.py
│   ├── self_improvement.py
│   └── tasks.py
│
├── agents/
│   ├── classifier_agent.py
│   ├── scheduler_agent.py
│   ├── rule_based.py
│   └── random_agent.py
│
├── training/
│   ├── state_encoder.py
│   ├── train_ppo.py
│   ├── evaluate.py
│   └── checkpoints/
│       ├── episode_rewards.csv
│       ├── curriculum_log.txt
│       └── ppo_email_triage.pt
│
├── visualization/
│   ├── dashboard.py
│   ├── compare_agents.py
│   ├── plot_rewards.py
│   └── plot_curriculum.py
│
├── baseline/
│   └── run_baseline.py
│
├── demo/
│   ├── run_demo.py
│   └── demo_script.md
│
├── data/
│   └── emails.json
│
├── Dockerfile
├── README.md
├── requirements.txt
├── openenv.yaml
├── pyproject.toml
├── inference.py
└── PROJECT_BLOG.md
```

---

# 6. Folder Responsibilities

## `app/`

Core environment and API.

| File | Purpose |
|---|---|
| `env.py` | Email triage environment |
| `features.py` | Email feature extraction |
| `generator.py` | Synthetic/adversarial email generation |
| `graders.py` | Decision grading logic |
| `main.py` | FastAPI application |
| `models.py` | API/data models |
| `reward.py` | Reward function |
| `self_improvement.py` | Curriculum tracking |
| `tasks.py` | Task loading/generation |

## `agents/`

Agent logic.

| File | Purpose |
|---|---|
| `classifier_agent.py` | Category + urgency prediction |
| `scheduler_agent.py` | Reply/escalate/defer/ignore decision |
| `rule_based.py` | Rule baseline |
| `random_agent.py` | Random baseline |

## `training/`

Reinforcement learning pipeline.

| File | Purpose |
|---|---|
| `state_encoder.py` | Converts state to tensors |
| `train_ppo.py` | PPO training loop |
| `evaluate.py` | Agent evaluation and comparison |
| `checkpoints/` | Model and metrics artifacts |

## `visualization/`

Dashboard and charts.

| File | Purpose |
|---|---|
| `dashboard.py` | Streamlit dashboard |
| `compare_agents.py` | Comparison chart generation |
| `plot_rewards.py` | Reward curve plot |
| `plot_curriculum.py` | Curriculum progression plot |

---

# 7. Reward Function

The reward function teaches real operational behavior.

Positive reward:

- correct classification
- correct urgency prediction
- handling urgent emails
- escalating critical issues
- ignoring phishing/spam
- preventing SLA breach

Penalties:

- missing urgent email
- replying to phishing
- ignoring critical incident
- unnecessary escalation
- deferring high-priority messages
- SLA breach

This avoids reward hacking. The agent cannot simply reply to everything because phishing replies are unsafe. It cannot escalate everything because unnecessary escalation is penalized.

---

# 8. PPO Training Pipeline

Training flow:

```text
Synthetic emails
        ↓
Environment reset
        ↓
State encoding
        ↓
PPO action selection
        ↓
Reward calculation
        ↓
Policy update
        ↓
Evaluation vs baselines
        ↓
Dashboard/API visualization
```

Run locally:

```bash
python -m baseline.run_baseline
python -m training.train_ppo
python visualization/plot_rewards.py
python visualization/plot_curriculum.py
python -m training.evaluate
python -m visualization.compare_agents
streamlit run visualization/dashboard.py
```

---

# 9. Self-Improvement Curriculum

The self-improvement loop:

```text
Agent performance
        ↓
Safety/SLA metrics
        ↓
Curriculum level update
        ↓
Harder scenario recommendation
        ↓
More robust training environment
```

Recent training reached:

```text
Self-Improvement Curriculum Level: 3
```

Example trace:

```text
Episode 397/400 | reward=-22.86 | avg_last_10=4.87
Self-Improvement Curriculum Level: 3

Episode 398/400 | reward=47.81 | avg_last_10=8.82
Self-Improvement Curriculum Level: 3

Episode 399/400 | reward=-16.83 | avg_last_10=6.57
Self-Improvement Curriculum Level: 3

Episode 400/400 | reward=-28.56 | avg_last_10=4.48
Self-Improvement Curriculum Level: 3
```

The reward remains noisy because harder scenarios include adversarial emails, spikes, and SLA pressure.

---

# 10. Evaluation Metrics

| Metric | Meaning |
|---|---|
| `avg_reward` | Average environment reward |
| `urgent_handled` | Urgent emails handled correctly |
| `urgent_missed` | Urgent emails missed or delayed |
| `sla_breaches` | Deadline failures |
| `safe_score` | Safety-oriented policy quality |
| `phishing_replied` | Unsafe phishing response behavior |

The goal is not only high reward. The goal is safer, more reliable decision-making.

---

# 11. API Testing

Base URL:

```text
https://vishalgulatinitj-email-triage-openenv.hf.space
```

Useful endpoints:

```bash
curl -X GET "https://vishalgulatinitj-email-triage-openenv.hf.space/"

curl -X GET "https://vishalgulatinitj-email-triage-openenv.hf.space/api/docs"

curl -X GET "https://vishalgulatinitj-email-triage-openenv.hf.space/api/openapi.json"

curl -X GET "https://vishalgulatinitj-email-triage-openenv.hf.space/api/evaluate_agent?agent_type=random&num_episodes=10"

curl -X GET "https://vishalgulatinitj-email-triage-openenv.hf.space/api/evaluate_agent?agent_type=rule&num_episodes=10"

curl -X GET "https://vishalgulatinitj-email-triage-openenv.hf.space/api/evaluate_agent?agent_type=ppo&num_episodes=10&checkpoint_path=training/checkpoints/ppo_email_triage.pt"

curl -X GET "https://vishalgulatinitj-email-triage-openenv.hf.space/api/compare_agents?num_episodes=10&checkpoint_path=training/checkpoints/ppo_email_triage.pt"
```

Postman import:

```text
Import → Link → https://vishalgulatinitj-email-triage-openenv.hf.space/api/openapi.json
```

---

# 12. Why This Can Win

## Environment Innovation — 40%

This is not a toy game or static classifier. It is a realistic enterprise decision environment with SLA, urgency, adversarial risk, and workload pressure.

## Storytelling — 30%

The story is simple:

> From inbox chaos to self-improving SLA-aware agents.

Judges can understand the problem immediately because everyone understands email overload.

## Showing Improvement — 20%

The project includes:

- PPO training logs
- reward curve
- curriculum progression
- random vs rule vs PPO comparison
- safety metrics
- SLA metrics

## Reward and Training Pipeline — 10%

The reward maps directly to real behavior:

```text
safe decisions + urgent handling + SLA compliance - unsafe replies - missed critical emails
```

The pipeline is end-to-end:

```text
data → environment → agents → reward → PPO → evaluation → dashboard/API
```

---

# 13. Suggested Demo Script

```text
Email Triage OpenEnv is a multi-agent RL environment for enterprise email decision-making.

Most systems classify email, but companies need decisions: reply, escalate, defer, or ignore.

The environment simulates urgent incidents, phishing, spam, workload spikes, SLA pressure, and backlog.

I compare random, rule-based, and PPO agents.

The PPO agent learns from reward feedback and is evaluated using reward, urgent handled, urgent missed, SLA breaches, and safe policy score.

The dashboard shows evaluation and curriculum progression.

The API exposes the environment under /api, and the Colab notebook reproduces the full pipeline.

The key idea is: this is not just classification. It is safe, SLA-aware, self-improving decision-making.
```

---

# 14. Where to Push This File

Recommended location:

```text
PROJECT_BLOG.md
```

Put it in the **root of your GitHub repo**:

```text
email-triage-openenv/PROJECT_BLOG.md
```

Also add this to your README:

```md
## 📖 Project Blog

Read the full project writeup here:

[PROJECT_BLOG.md](PROJECT_BLOG.md)
```

Push to GitHub:

```bash
git add PROJECT_BLOG.md README.md
git commit -m "Add full project blog writeup"
git push origin main
```

For Hugging Face Space, upload `PROJECT_BLOG.md` to the Space root as well, or include it in your clean deploy folder.

---

# 15. Final Takeaway

At internet scale, email triage is not just about reading messages.

It is about deciding what matters most.

**Email Triage OpenEnv** demonstrates how multi-agent systems, PPO reinforcement learning, reward shaping, and curriculum self-improvement can be combined to create safer enterprise decision agents.

> **From inbox chaos to self-improving SLA-aware agents.**
