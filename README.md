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