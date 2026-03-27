from pydantic import BaseModel
from typing import List, Dict, Any


class State(BaseModel):
    price_history: List[float]
    current_price: float
    position: float
    cash: float
    ma5: float
    ma10: float
    sharpe: float
    step: int


class Action(BaseModel):
    action: str
    quantity: float


class StepResult(BaseModel):
    observation: State
    reward: float
    done: bool
    info: Dict[str, Any]