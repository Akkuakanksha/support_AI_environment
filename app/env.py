import random
import uuid
from typing import List, Tuple, Any, Dict
from models import Observation, Action, Reward
from app.graders.grader import grade_easy, grade_medium, grade_hard

# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = [
    {
        "difficulty": "easy",
        "ticket_id": "TKT-001",
        "customer_query": "Hi, I want to request a refund for my order. How do I get a refund?",
        "max_steps": 5,
    },
    {
        "difficulty": "medium",
        "ticket_id": "TKT-002",
        "customer_query": "I made a payment but it failed and money was deducted from my account. Please help.",
        "max_steps": 6,
    },
    {
        "difficulty": "hard",
        "ticket_id": "TKT-003",
        "customer_query": "I was charged twice for the same order and nobody from support has helped me. This is unacceptable!",
        "max_steps": 8,
    },
]


def _clamp(score: float) -> float:
    """Clamp score strictly to (0.01, 0.99) as required by validator."""
    return round(min(max(score, 0.01), 0.99), 4)


class SupportEnv:
    """
    Customer Support Ticket Resolution Environment.

    The agent must handle support tickets by taking a sequence of actions:
    classify → respond / escalate → close

    Tasks:
      easy   – simple refund inquiry
      medium – failed payment, needs multi-step resolution
      hard   – double charge + angry customer, requires escalation
    """

    VALID_ACTIONS = {"classify", "respond", "escalate", "close"}

    def __init__(self):
        self._task_index = 0
        self._current_task: Dict = {}
        self._history: List[str] = []
        self._actions_taken: List[Action] = []
        self._step_count = 0
        self._done = False
        self._episode_id = str(uuid.uuid4())

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, task_index: int = None) -> Observation:
        """Start a new episode. Cycles through tasks if task_index not given."""
        if task_index is not None:
            self._task_index = task_index % len(TASKS)
        else:
            self._task_index = (self._task_index) % len(TASKS)

        self._current_task = TASKS[self._task_index]
        self._history = []
        self._actions_taken = []
        self._step_count = 0
        self._done = False
        self._episode_id = str(uuid.uuid4())
        self._task_index += 1

        return Observation(
            ticket_id=self._current_task["ticket_id"],
            customer_query=self._current_task["customer_query"],
            history=self._history.copy(),
            status="open",
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """Process one action and return (observation, reward, done, info)."""
        if self._done:
            obs = self._build_observation()
            return obs, Reward(score=0.01, reason="Episode already done"), True, {}

        # Validate action
        if action.action_type not in self.VALID_ACTIONS:
            reward = Reward(score=0.01, reason=f"Invalid action: {action.action_type}")
            return self._build_observation(), reward, False, {}

        self._actions_taken.append(action)
        self._step_count += 1
        self._history.append(f"[Step {self._step_count}] action={action.action_type}" +
                              (f" content='{action.content}'" if action.content else ""))

        difficulty = self._current_task["difficulty"]
        max_steps = self._current_task["max_steps"]

        # ── Compute reward ────────────────────────────────────────────────────
        reward, done = self._compute_reward(action, difficulty)

        # Clamp score strictly to (0.01, 0.99) — validator requires (0, 1)
        clamped_score = _clamp(reward.score)
        reward = Reward(score=clamped_score, reason=reward.reason)

        # Force done if max steps reached
        if self._step_count >= max_steps:
            done = True
            if reward.score < 0.5:
                reward = Reward(score=reward.score, reason=reward.reason + " [max steps reached]")

        self._done = done
        status = "closed" if done else "in_progress"

        obs = Observation(
            ticket_id=self._current_task["ticket_id"],
            customer_query=self._current_task["customer_query"],
            history=self._history.copy(),
            status=status,
        )
        return obs, reward, done, {"step": self._step_count, "difficulty": difficulty}

    def state(self) -> Dict:
        """Return current environment state."""
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "task": self._current_task,
            "history": self._history,
            "done": self._done,
        }

    # ── Reward computation ────────────────────────────────────────────────────

    def _compute_reward(self, action: Action, difficulty: str) -> Tuple[Reward, bool]:
        action_types = [a.action_type for a in self._actions_taken]

        if difficulty == "easy":
            return self._reward_easy(action, action_types)
        elif difficulty == "medium":
            return self._reward_medium(action, action_types)
        else:
            return self._reward_hard(action, action_types)

    def _reward_easy(self, action: Action, action_types: List[str]) -> Tuple[Reward, bool]:
        """
        Easy: respond with refund info → close.
        Ideal: respond → close
        """
        if action.action_type == "respond":
            content = (action.content or "").lower()
            if any(k in content for k in ["refund", "7 days", "return", "policy"]):
                return Reward(score=0.6, reason="Good refund response"), False
            return Reward(score=0.3, reason="Response given but lacks refund details"), False

        if action.action_type == "close":
            if "respond" in action_types[:-1]:
                return Reward(score=0.95, reason="Ticket properly responded and closed"), True
            return Reward(score=0.2, reason="Closed without responding"), True

        if action.action_type == "escalate":
            return Reward(score=0.02, reason="Unnecessary escalation for simple refund"), False

        if action.action_type == "classify":
            return Reward(score=0.1, reason="Classification step (optional for easy task)"), False

        return Reward(score=0.02, reason="No progress"), False

    def _reward_medium(self, action: Action, action_types: List[str]) -> Tuple[Reward, bool]:
        """
        Medium: classify → respond → close.
        Ideal: classify → respond → close
        """
        if action.action_type == "classify":
            if action_types.count("classify") == 1:
                return Reward(score=0.3, reason="Issue classified"), False
            return Reward(score=0.02, reason="Already classified"), False

        if action.action_type == "respond":
            content = (action.content or "").lower()
            if "classify" in action_types:
                if any(k in content for k in ["payment", "resolve", "inconvenience", "refund"]):
                    return Reward(score=0.6, reason="Relevant response after classification"), False
                return Reward(score=0.3, reason="Response given but vague"), False
            return Reward(score=0.2, reason="Responded without classifying first"), False

        if action.action_type == "close":
            has_classify = "classify" in action_types[:-1]
            has_respond = "respond" in action_types[:-1]
            if has_classify and has_respond:
                return Reward(score=0.95, reason="Full workflow: classify → respond → close"), True
            elif has_respond:
                return Reward(score=0.6, reason="Responded then closed (no classification)"), True
            return Reward(score=0.1, reason="Closed without proper handling"), True

        if action.action_type == "escalate":
            return Reward(score=0.1, reason="Escalation not necessary for payment issue"), False

        return Reward(score=0.02, reason="No progress"), False

    def _reward_hard(self, action: Action, action_types: List[str]) -> Tuple[Reward, bool]:
        """
        Hard: classify → escalate → respond → close.
        Requires escalation for double-charge angry customer.
        """
        if action.action_type == "classify":
            if action_types.count("classify") == 1:
                return Reward(score=0.2, reason="Issue classified"), False
            return Reward(score=0.02, reason="Already classified"), False

        if action.action_type == "escalate":
            if "classify" in action_types:
                return Reward(score=0.3, reason="Correctly escalated after classification"), False
            return Reward(score=0.15, reason="Escalated (without classifying first)"), False

        if action.action_type == "respond":
            content = (action.content or "").lower()
            has_escalate = "escalate" in action_types
            if has_escalate:
                if any(k in content for k in ["escalat", "resolve", "charged", "apologize", "sorry", "priority"]):
                    return Reward(score=0.4, reason="Good response after escalation"), False
                return Reward(score=0.2, reason="Response after escalation but vague"), False
            return Reward(score=0.1, reason="Response without escalation (required for hard case)"), False

        if action.action_type == "close":
            has_classify = "classify" in action_types[:-1]
            has_escalate = "escalate" in action_types[:-1]
            has_respond = "respond" in action_types[:-1]
            if has_classify and has_escalate and has_respond:
                return Reward(score=0.95, reason="Full workflow: classify → escalate → respond → close"), True
            elif has_escalate and has_respond:
                return Reward(score=0.7, reason="Escalated and responded (no classification)"), True
            elif has_respond:
                return Reward(score=0.4, reason="Responded but did not escalate"), True
            return Reward(score=0.1, reason="Closed without proper handling"), True

        return Reward(score=0.02, reason="No progress"), False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        status = "closed" if self._done else ("in_progress" if self._history else "open")
        return Observation(
            ticket_id=self._current_task.get("ticket_id", "TKT-000"),
            customer_query=self._current_task.get("customer_query", ""),
            history=self._history.copy(),
            status=status,
        )

    @staticmethod
    def task_list():
        return [
            {"id": i, "difficulty": t["difficulty"], "ticket_id": t["ticket_id"],
             "query": t["customer_query"]}
            for i, t in enumerate(TASKS)
        ]