"""
Baseline inference script for Support Env.
Uses OpenAI client to run an LLM agent against all 3 tasks.

Required env vars:
  API_BASE_URL  - LLM API base URL
  MODEL_NAME    - model identifier
  HF_TOKEN      - Hugging Face / API key  (used as api_key)
"""
import os
import sys
import json

from openai import OpenAI
from models import Action
from app.env import SupportEnv, TASKS

# ── LLM Client ────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-3.5-turbo")
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


def llm_decide(obs_dict: dict) -> Action:
    """Call the LLM to decide the next action."""
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
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return Action(
            action_type=parsed.get("action_type", "close"),
            content=parsed.get("content"),
        )
    except Exception as e:
        print(f"  [LLM error] {e} — defaulting to close")
        return Action(action_type="close")


def run_task(task_index: int) -> float:
    """Run one full episode for a given task, return cumulative score."""
    env = SupportEnv()
    obs = env.reset(task_index=task_index)
    difficulty = TASKS[task_index]["difficulty"]
    print(f"\n{'='*55}")
    print(f"  Task {task_index+1} [{difficulty.upper()}]: {obs.ticket_id}")
    print(f"  Query: {obs.customer_query[:70]}...")
    print(f"{'='*55}")

    total_score = 0.0
    done = False
    step = 0

    while not done and step < 10:
        action = llm_decide(obs.model_dump())
        print(f"  Step {step+1}: action={action.action_type}" +
              (f" | content={action.content[:40]!r}" if action.content else ""))

        obs, reward, done, info = env.step(action)
        total_score += reward.score
        print(f"           reward={reward.score:.2f} — {reward.reason}")
        step += 1

    print(f"\n  ► Total score: {total_score:.3f}  |  Steps: {step}")
    return total_score


def main():
    print("\n🎧 Support Env — Baseline Inference")
    print(f"   Model: {MODEL_NAME}")
    print(f"   API:   {API_BASE_URL}")

    all_scores = []
    for i in range(len(TASKS)):
        score = run_task(i)
        all_scores.append(score)

    print(f"\n{'='*55}")
    print("📊 RESULTS")
    print(f"{'='*55}")
    for i, (task, score) in enumerate(zip(TASKS, all_scores)):
        bar = "█" * int(score * 20)
        print(f"  Task {i+1} [{task['difficulty']:6s}]: {score:.3f}  {bar}")
    avg = sum(all_scores) / len(all_scores)
    print(f"\n  Average Score: {avg:.3f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
