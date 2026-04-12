"""
Deterministic graders for the 3 support tasks.
Each returns a float strictly in (0.01, 0.99).
"""
from typing import List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models import Action


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1."""
    return round(min(max(score, 0.01), 0.99), 4)


def grade_easy(response_content: str, keywords: List[str]) -> float:
    """
    Easy task grader: score how well the agent's response addresses a refund query.
    """
    if not response_content:
        return 0.01
    content = response_content.lower()
    matched = sum(1 for kw in keywords if kw.lower() in content)
    raw = matched / len(keywords) if keywords else 0.0
    return _clamp(raw * 0.98 + 0.01)


def grade_medium(actions: List[Action], expected_sequence: List[str]) -> float:
    """
    Medium task grader: score based on how closely actions match the expected sequence.
    """
    if not actions:
        return 0.01
    taken = [a.action_type for a in actions]
    matches = 0
    j = 0
    for step in expected_sequence:
        for k in range(j, len(taken)):
            if taken[k] == step:
                matches += 1
                j = k + 1
                break
    raw = matches / len(expected_sequence)
    return _clamp(raw * 0.98 + 0.01)


def grade_hard(actions: List[Action], expected_sequence: List[str]) -> float:
    """
    Hard task grader: strict ordering matters.
    """
    if not actions:
        return 0.01
    taken = [a.action_type for a in actions]

    score = 0.0
    weights = {"classify": 0.2, "escalate": 0.3, "respond": 0.3, "close": 0.2}
    last_idx = -1
    for step in expected_sequence:
        for i in range(last_idx + 1, len(taken)):
            if taken[i] == step:
                score += weights.get(step, 0.25)
                last_idx = i
                break

    return _clamp(min(score, 0.98) * 0.98 + 0.01)