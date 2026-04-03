import sys, os
sys.path.insert(0, os.path.abspath("."))

from models import Action
from app.graders.grader import grade_easy, grade_medium, grade_hard

# EASY TEST
score = grade_easy("You can request a refund within 7 days of purchase.", ["refund", "7 days"])
print(f"Easy grader:   {score}")  # Expected: 1.0

# MEDIUM TEST
actions = [
    Action(action_type="classify"),
    Action(action_type="respond", content="Payment issue will be resolved"),
    Action(action_type="close"),
]
score = grade_medium(actions, ["classify", "respond", "close"])
print(f"Medium grader: {score}")  # Expected: 1.0

# HARD TEST
actions = [
    Action(action_type="classify"),
    Action(action_type="escalate"),
    Action(action_type="respond", content="We escalated your double-charge issue"),
    Action(action_type="close"),
]
score = grade_hard(actions, ["classify", "escalate", "respond", "close"])
print(f"Hard grader:   {score}")  # Expected: 1.0

# Partial credit test
actions_partial = [Action(action_type="respond"), Action(action_type="close")]
score = grade_hard(actions_partial, ["classify", "escalate", "respond", "close"])
print(f"Hard partial:  {score}")  # Expected: ~0.5

print("\n✅ Grader tests complete!")
