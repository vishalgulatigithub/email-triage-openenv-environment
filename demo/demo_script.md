# Demo Sequence

## 1. Opening
We built a multi-agent reinforcement learning benchmark for enterprise email triage.

This is not a plain classification task.
It models sequential decision-making under workload spikes, adversarial emails, SLA deadlines, and backlog pressure.

## 2. Show the environment
Open the dashboard or run a demo episode.
Explain that the state includes:
- current email
- inbox size
- pending urgent count
- extracted risk features
- spike state
- recent history

## 3. Show the two-agent flow
Classifier agent predicts:
- category
- priority

Scheduler agent decides:
- reply
- escalate
- defer
- ignore

Explain that PPO learns these jointly through multiple action heads.

## 4. Show adversarial and workload realism
Mention that the dataset includes:
- spam
- phishing
- complaints
- billing issues
- production issues
- escalation emails
- spike batches

## 5. Show learning curve
Display the reward curve and say:
The policy starts unstable, explores widely, and then improves as it learns better operational decisions.

## 6. Show comparison charts
Highlight:
- PPO vs random vs rule on reward
- PPO vs random vs rule on urgent handled
- PPO vs random vs rule on urgent missed
- PPO vs random vs rule on SLA breaches

## 7. Key insight
Even when the rule baseline is efficient, PPO learns a safer operational policy with dramatically fewer urgent misses and SLA breaches.

## 8. Closing line
This environment is designed as a benchmark for decision-making under operational constraints, with direct applications to customer support, IT operations, and enterprise workflow automation.