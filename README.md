---
title: Support Env
emoji: 🎧
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---
# 🎧 Support Env — OpenEnv Customer Support Environment

An **OpenEnv**-compliant environment where AI agents learn to resolve customer support tickets by taking a sequence of structured actions.

---

## 🌍 Real-World Task

Customer support ticket resolution is a task millions of agents (human and AI) perform daily. This environment models the decision pipeline:

**Classify → Respond / Escalate → Close**

Agents must read a support ticket, decide how to handle it (simple reply vs. multi-step escalation), and close it correctly — earning rewards proportional to the quality of their handling.

---

## 📋 Tasks

| # | Difficulty | Ticket | Ideal Sequence |
|---|------------|--------|----------------|
| 1 | **Easy**   | Customer wants a refund | `respond` (with policy) → `close` |
| 2 | **Medium** | Payment failed, money deducted | `classify` → `respond` → `close` |
| 3 | **Hard**   | Charged twice + angry customer | `classify` → `escalate` → `respond` → `close` |

---

## 🕹 Action Space

| Action | `action_type` | `content` |
|--------|--------------|-----------|
| Classify the ticket | `classify` | _(optional)_ |
| Send a response | `respond` | Required: your message text |
| Escalate to senior support | `escalate` | _(optional)_ |
| Close the ticket | `close` | _(optional)_ |

---

## 👁 Observation Space

```json
{
  "ticket_id": "TKT-001",
  "customer_query": "I want to request a refund...",
  "history": ["[Step 1] action=respond content='...'"],
  "status": "open | in_progress | closed"
}
```

---

## 🏆 Reward Function

Rewards are **dense** (partial credit at every step):

- **+0.1–0.3** for classification
- **+0.3–0.6** for relevant responses
- **+0.3** for correct escalation on hard cases
- **+1.0** for closing after completing the full correct workflow
- **-0.1** for unnecessary escalation on easy tasks
- **0.0** for invalid actions

Final episode scores range `0.0–1.0` per grader.

---

## 🚀 Setup & Usage

### Local

```bash
git clone https://github.com/YOUR_USERNAME/support-env
cd support-env
pip install -r requirements.txt

# Run server
python server.py
# Open http://localhost:7860 in browser

# Run tests
python test_env.py
python test_grader.py

# Run baseline inference (requires API key)
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-3.5-turbo
export HF_TOKEN=sk-...
python inference.py
```

### Docker

```bash
docker build -t support-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-3.5-turbo \
  -e HF_TOKEN=sk-... \
  support-env
```

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start new episode (`{"task_index": 0}`) |
| `POST` | `/step` | Take action (`{"action_type": "classify"}`) |
| `GET`  | `/state` | Current episode state |
| `GET`  | `/tasks` | List all tasks |
| `GET`  | `/health` | Health check |

---

## 📊 Baseline Scores

Scores from a GPT-3.5-turbo baseline agent:

| Task | Difficulty | Score |
|------|------------|-------|
| Refund inquiry | Easy | ~1.4 |
| Payment failure | Medium | ~1.3 |
| Double charge + escalation | Hard | ~1.1 |

*(Cumulative per-step rewards; final grader scores 0.0–1.0)*

---

## 🏗 Project Structure

```
support-env/
├── Dockerfile
├── requirements.txt
├── openenv.yaml
├── server.py          # FastAPI server + HTML/CSS/JS UI
├── models.py          # Observation, Action, Reward
├── inference.py       # Baseline agent (OpenAI client)
├── test_env.py        # Environment tests
├── test_grader.py     # Grader tests
└── app/
    ├── env.py         # SupportEnv class
    └── graders/
        └── grader.py  # grade_easy, grade_medium, grade_hard
```

---

## 🏷 Tags

`openenv` · `customer-support` · `real-world` · `reasoning`
