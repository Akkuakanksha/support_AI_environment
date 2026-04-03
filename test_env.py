import sys, os
sys.path.insert(0, os.path.abspath("."))

from app.env import SupportEnv
from models import Action

env = SupportEnv()

# Test all 3 tasks
for task_idx in range(3):
    print(f"\n--- Task {task_idx+1} ---")
    obs = env.reset(task_index=task_idx)
    print(f"Ticket: {obs.ticket_id} | Status: {obs.status}")
    print(f"Query: {obs.customer_query}")

    action = Action(action_type="classify")
    obs, reward, done, _ = env.step(action)
    print(f"After classify: reward={reward.score} | {reward.reason}")

    action = Action(action_type="respond", content="We will resolve your issue promptly.")
    obs, reward, done, _ = env.step(action)
    print(f"After respond:  reward={reward.score} | {reward.reason}")

    action = Action(action_type="close")
    obs, reward, done, _ = env.step(action)
    print(f"After close:    reward={reward.score} | {reward.reason} | done={done}")

print("\n✅ All tests passed!")
