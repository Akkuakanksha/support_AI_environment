from pydantic import BaseModel
from typing import List, Optional


class Observation(BaseModel):
    ticket_id: str
    customer_query: str
    history: List[str]
    status: str  # open / in_progress / closed


class Action(BaseModel):
    action_type: str  # classify, respond, escalate, close
    content: Optional[str] = None


class Reward(BaseModel):
    score: float
    reason: Optional[str] = None
