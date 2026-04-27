---
title: Adversarial API Arena
emoji: 🤺
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Adversarial API Arena — Environment Server

Multi-agent adversarial RL environment exposed as a FastAPI HTTP server.

## How to deploy to HuggingFace Spaces

```bash
# 1. Create a new Space (Docker SDK)
huggingface-cli repo create adversarial-api-arena --type space --space-sdk docker

# 2. Clone the space
git clone https://huggingface.co/spaces/<your-username>/adversarial-api-arena
cd adversarial-api-arena

# 3. Copy project files
cp -r /path/to/Finale-meta/* .
cp deploy/hf_Dockerfile Dockerfile
cp deploy/README_hf_space.md README.md

# 4. Push
git add .
git commit -m "deploy adversarial api arena"
git push
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| POST | `/reset` | Start new episode (task agent) |
| POST | `/step` | Execute task action |
| GET | `/state` | Current task observation |
| POST | `/adversary/step` | Stage adversary action |
| GET | `/adversary/state` | Adversary observation (includes ground truth) |
| GET | `/adversary/reward` | Adversary terminal reward |
| GET | `/episode/log` | Full step log for inspection |
| GET | `/docs` | OpenAPI docs |

## Usage from trainer

```python
from client import TaskAgentClient, AdversaryClient

task_client = TaskAgentClient("https://<your-username>-adversarial-api-arena.hf.space")
adv_client  = AdversaryClient("https://<your-username>-adversarial-api-arena.hf.space")

obs = task_client.reset(seed=42)
print(obs.task_description)
```
