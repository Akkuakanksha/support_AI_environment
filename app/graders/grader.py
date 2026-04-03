"""
Deterministic graders for the 3 support tasks.
Each returns a float in [0.0, 1.0].
"""
from typing import List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models import Action


def grade_easy(response_content: str, keywords: List[str]) -> float:
    """
    Easy task grader: score how well the agent's response addresses a refund query.
    - 1.0 if all keywords present
    - Partial credit per keyword matched
    - 0.0 if empty
    """
    if not response_content:
        return 0.0
    content = response_content.lower()
    matched = sum(1 for kw in keywords if kw.lower() in content)
    return round(matched / len(keywords), 2) if keywords else 0.0


def grade_medium(actions: List[Action], expected_sequence: List[str]) -> float:
    """
    Medium task grader: score based on how closely actions match the expected sequence.
    - 1.0 for perfect match
    - Partial credit for each correct step in order
    """
    if not actions:
        return 0.0
    taken = [a.action_type for a in actions]
    matches = 0
    j = 0
    for step in expected_sequence:
        for k in range(j, len(taken)):
            if taken[k] == step:
                matches += 1
                j = k + 1
                break
    return round(matches / len(expected_sequence), 2)


def grade_hard(actions: List[Action], expected_sequence: List[str]) -> float:
    """
    Hard task grader: strict ordering matters.
    - 1.0 only if classify → escalate → respond → close all present in correct order
    - Partial scores per phase completed
    """
    if not actions:
        return 0.0
    taken = [a.action_type for a in actions]

    # Check each required step appears in order
    score = 0.0
    weights = {"classify": 0.2, "escalate": 0.3, "respond": 0.3, "close": 0.2}
    last_idx = -1
    for step in expected_sequence:
        for i in range(last_idx + 1, len(taken)):
            if taken[i] == step:
                score += weights.get(step, 0.25)
                last_idx = i
                break

    return round(min(score, 1.0), 2)
