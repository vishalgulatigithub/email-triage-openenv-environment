from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ---------------------------
# EMAIL
# ---------------------------
class Email(BaseModel):
    id: str
    subject: str
    body: str


# ---------------------------
# ACTION (VERY IMPORTANT)
# ---------------------------
class Action(BaseModel):
    action_type: str
    email_id: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    response_text: Optional[str] = None


# ---------------------------
# OBSERVATION
# ---------------------------
class Observation(BaseModel):
    inbox: List[Email]
    current_email: Optional[Email] = None
    stage: str
    history: List[str]
    extracted: Dict[str, Any]
    remaining_steps: int


# ---------------------------
# REWARD
# ---------------------------
class Reward(float):
    pass