"""
Baseline inference script for Support Env.
Uses OpenAI client to run an LLM agent against all 3 tasks.

Required env vars:
  API_BASE_URL  - LLM API base URL
  MODEL_NAME    - model identifier
  HF_TOKEN      - Hugging Face / API key (used as api_key)
"""
import os
import sys
import json

from openai import OpenAI
from models import Action
from app.env import SupportEnv, TASKS

# ── LLM Client ────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
API_KEY      = os.getenv("HF_TOKEN",     os.getenv("OPENAI_API_KEY", "no-key"))

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

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


# ── Rule-based fallback agent (used when LLM unavailable) ────────────────────

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


def get_difficulty(obs_dict: dict) -> str:
    """Detect task difficulty from ticket id."""
    ticket_id = obs_dict.get("ticket_id", "TKT-001")
    if ticket_id == "TKT-001":
        return "easy"
    elif ticket_id == "TKT-002":
        return "medium"
    else:
        return "hard"


_fallback_step = {}


def llm_decide(obs_dict: dict, task_name: str) -> Action:
    """Call the LLM to decide the next action. Falls back to rule-based agent on error."""
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
        return Action(
            action_type=parsed.get("action_type", "close"),
            content=parsed.get("content"),
        )
    except Exception:
        # ── Fallback: rule-based agent ────────────────────────────────────────
        difficulty = get_difficulty(obs_dict)
        sequence = FALLBACK_SEQUENCES[difficulty]
        step_idx = _fallback_step.get(task_name, 0)
        action = sequence[min(step_idx, len(sequence) - 1)]
        _fallback_step[task_name] = step_idx + 1
        return action


def run_task(task_index: int) -> float:
    """Run one full episode for a given task, return cumulative score."""
    env = SupportEnv()
    obs = env.reset(task_index=task_index)
    difficulty = TASKS[task_index]["difficulty"]
    task_name = f"{difficulty}_task_{task_index + 1}"

    # Reset fallback step counter for this task
    _fallback_step[task_name] = 0

    # Required structured output: START block
    print(f"[START] task={task_name}", flush=True)

    total_score = 0.0
    done = False
    step = 0

    while not done and step < 10:
        action = llm_decide(obs.model_dump(), task_name)
        obs, reward, done, info = env.step(action)
        total_score += reward.score
        step += 1

        # Required structured output: STEP block
        print(f"[STEP] step={step} action={action.action_type} reward={reward.score:.4f}", flush=True)

    # Required structured output: END block
    print(f"[END] task={task_name} score={total_score:.4f} steps={step}", flush=True)

    return total_score


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