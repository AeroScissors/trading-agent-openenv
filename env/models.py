# File: env/models.py

from pydantic import BaseModel, Field
from typing import Dict, Any


class State(BaseModel):
    step: int
    prices: Dict[str, float]                # {"AAPL": 180.2, "MSFT": 320.1}
    portfolio_value: float                 # total portfolio value
    weights: Dict[str, float]              # {"AAPL": 0.4, "MSFT": 0.3}
    cash_fraction: float                   # remaining cash (0 → 1)


class Action(BaseModel):
    weights: Dict[str, float] = Field(default_factory=dict)
    # Example:
    # {"AAPL": 0.4, "MSFT": 0.3}


class StepResult(BaseModel):
    observation: State
    reward: float
    done: bool
    info: Dict[str, Any]