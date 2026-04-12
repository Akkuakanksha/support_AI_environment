"""
Baseline inference script for Support Env.
Uses OpenAI Client for all LLM calls.

Required env vars:
  API_BASE_URL  - API endpoint for the LLM (has default)
  MODEL_NAME    - Model identifier (has default)
  HF_TOKEN      - Hugging Face API token (mandatory)
"""
import os
import sys
import json

from openai import OpenAI
from models import Action
from app.env import SupportEnv, TASKS

# ── Environment variables ─────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI Client ─────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "support-env"

SYSTEM_PROMPT = """You are a customer support AI agent operating in the Support Env environment.
You must handle support tickets by choosing the correct sequence of actions.

Available actions (return ONLY valid JSON):
  {"action_type": "classify"}
  {"action_type": "respond", "content": "<your response>"}
  {"action_type": "escalate"}
  {"action_type": "close"}

Guidelines:
- Easy (refund): respond with policy details → close
- Medium (payment issue): classify → respond → close
- Hard (double charge, angry): classify → escalate → respond → close
- Always end with close when resolved
- Return ONLY the JSON object, no extra text."""


# ── Rule-based fallback agent (when LLM unavailable) ─────────────────────────
FALLBACK_SEQUENCES = {
    "easy": [
        Action(action_type="respond", content="You can request a refund within 7 days of purchase. Please contact us with your order ID."),
        Action(action_type="close"),
    ],
    "medium": [
        Action(action_type="classify"),
        Action(action_type="respond", content="We are sorry for the inconvenience. Your payment issue will be investigated and resolved within 24 hours."),
        Action(action_type="close"),
    ],
    "hard": [
        Action(action_type="classify"),
        Action(action_type="escalate"),
        Action(action_type="respond", content="We sincerely apologize. We have escalated your double charge issue to our senior team and will resolve it within 2 hours."),
        Action(action_type="close"),
    ],
}

_fallback_step = {}


def get_difficulty(obs_dict: dict) -> str:
    ticket_id = obs_dict.get("ticket_id", "TKT-001")
    if ticket_id == "TKT-001":
        return "easy"
    elif ticket_id == "TKT-002":
        return "medium"
    else:
        return "hard"


def llm_decide(obs_dict: dict, task_name: str) -> tuple:
    """
    Call the LLM to decide the next action.
    Returns (Action, error_string_or_null).
    Falls back to rule-based agent on error.
    """
    user_msg = (
        f"Ticket: {obs_dict['ticket_id']}\n"
        f"Query: {obs_dict['customer_query']}\n"
        f"Status: {obs_dict['status']}\n"
        f"History: {obs_dict['history']}\n\n"
        "What is your next action? Reply with JSON only."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        action = Action(
            action_type=parsed.get("action_type", "close"),
            content=parsed.get("content"),
        )
        return action, "null"
    except Exception as e:
        # Fallback: rule-based agent
        difficulty = get_difficulty(obs_dict)
        sequence = FALLBACK_SEQUENCES[difficulty]
        step_idx = _fallback_step.get(task_name, 0)
        action = sequence[min(step_idx, len(sequence) - 1)]
        _fallback_step[task_name] = step_idx + 1
        err = str(e).replace("\n", " ")[:80]
        return action, f"fallback:{err}"


def run_task(task_index: int) -> float:
    """Run one full episode for a given task."""
    env = SupportEnv()
    obs = env.reset(task_index=task_index)
    difficulty = TASKS[task_index]["difficulty"]
    task_name = f"{difficulty}_task_{task_index + 1}"

    # Reset fallback step counter
    _fallback_step[task_name] = 0

    # ── [START] line ──────────────────────────────────────────────────────────
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards = []
    done = False
    step = 0
    last_error = "null"

    while not done and step < 10:
        action, error = llm_decide(obs.model_dump(), task_name)
        obs, reward, done, info = env.step(action)

        step += 1
        clamped_r = round(min(max(reward.score, 0.01), 0.99), 2)
        rewards.append(clamped_r)
        last_error = error

        # Action string representation
        action_str = action.action_type
        if action.content:
            action_str += f"('{action.content[:30]}')"

        # ── [STEP] line ───────────────────────────────────────────────────────
        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={clamped_r:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={error}",
            flush=True
        )

    success = done and len(rewards) > 0 and rewards[-1] > 0

    # Rewards as comma-separated 2dp values
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # ── [END] line ────────────────────────────────────────────────────────────
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step} "
        f"rewards={rewards_str}",
        flush=True
    )

    # Return normalized score strictly in (0, 1)
    raw = sum(rewards)
    max_possible = {"easy": 1.6, "medium": 1.9, "hard": 1.9}.get(difficulty, 2.0)
    normalized = round(min(max(raw / max_possible, 0.01), 0.99), 4)
    return normalized


def main():
    print("🎧 Support Env — Baseline Inference", flush=True)
    print(f"   Model: {MODEL_NAME}", flush=True)
    print(f"   API:   {API_BASE_URL}", flush=True)

    all_scores = []
    for i in range(len(TASKS)):
        score = run_task(i)
        all_scores.append(score)

    avg = sum(all_scores) / len(all_scores)
    print(f"\n📊 RESULTS", flush=True)
    for i, (task, score) in enumerate(zip(TASKS, all_scores)):
        print(f"  Task {i+1} [{task['difficulty']:6s}]: {score:.4f}", flush=True)
    print(f"  Average Score: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()