from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class State(BaseModel):
    price_history: List[float]
    current_price: float
    position: float        # number of shares/units held
    cash: float            # available cash
    ma5: float             # 5-period moving average
    ma10: float            # 10-period moving average
    sharpe: float          # rolling Sharpe ratio
    step: int              # current timestep


class Action(BaseModel):
    action: str            # "BUY", "SELL", or "HOLD"
    quantity: float        # number of units to trade (0.0 for HOLD)


class StepResult(BaseModel):
    observation: State
    reward: float
    done: bool
    info: Dict[str, Any]