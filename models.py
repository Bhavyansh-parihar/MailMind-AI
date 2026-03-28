from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class EmailUrgency(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    content: str
    urgency: EmailUrgency
    timestamp: datetime

class ActionType(str, Enum):
    MARK_IMPORTANT = "mark_important"
    REPLY = "reply"
    ARCHIVE = "archive"
    SCHEDULE_FOLLOWUP = "schedule_followup"
    NO_OP = "no_op"

class Action(BaseModel):
    type: ActionType
    email_id: Optional[str] = None
    reply_text: Optional[str] = None
    reasoning: Optional[str] = "No explanation provided."

class Observation(BaseModel):
    pending_emails: List[Email]
    current_email: Optional[Email]
    emails_handled: int
    emails_pending: int
    time_remaining: float
    last_action_status: str

class Reward(BaseModel):
    value: float
    reason: str

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict

class State(BaseModel):
    inbox: List[Email]
    history: List[dict]
    stats: dict
