# File: api/routes.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator
from typing import Optional, Dict
from fastapi.responses import FileResponse

from env.core_env import TradingEnv
from env.models import Action

# ------------------------------------------------------------------ #

router = APIRouter()

@router.get("/")
def serve_dashboard():
    return FileResponse("frontend/index.html")

_envs: dict[str, TradingEnv] = {}


def _get_env(task: str) -> TradingEnv:
    if task not in _envs:
        # Auto-initialize so validator can call /step or /grader before /reset
        env = TradingEnv(task=task)
        env.reset()
        _envs[task] = env
    return _envs[task]

# ------------------------------------------------------------------ #
# Schemas
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    task: Optional[str] = None
    task_id: Optional[str] = None

    @model_validator(mode="after")
    def resolve_task(self):
        self.task = self.task or self.task_id or "easy"
        return self


class StepRequest(BaseModel):
    task: str = "easy"
    weights: Dict[str, float]


class GraderRequest(BaseModel):
    task: Optional[str] = None
    task_id: Optional[str] = None

    @model_validator(mode="after")
    def resolve_task(self):
        self.task = self.task or self.task_id or "easy"
        return self


# ------------------------------------------------------------------ #
# Tasks
# ------------------------------------------------------------------ #

@router.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "easy"},
            {"id": "medium"},
            {"id": "hard"}
        ]
    }

# ------------------------------------------------------------------ #
# Reset
# ------------------------------------------------------------------ #

@router.post("/reset")
def reset_env(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest(task="easy")

    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="Invalid task")

    env = TradingEnv(task=req.task)
    state = env.reset()

    _envs[req.task] = env

    return {
        "task": req.task,
        "observation": state.model_dump()
    }

# ------------------------------------------------------------------ #
# State
# ------------------------------------------------------------------ #

@router.get("/state")
def get_state(task: str = "easy"):
    env = _get_env(task)
    return {
        "task": task,
        "observation": env.state().model_dump()
    }

# ------------------------------------------------------------------ #
# Step
# ------------------------------------------------------------------ #

@router.post("/step")
def take_step(req: StepRequest):
    env = _get_env(req.task)

    # Strip unknown symbols, clamp negatives to 0
    weights = {k: max(0.0, v) for k, v in req.weights.items() if k in env.symbols}

    # If validator sends empty or all-unknown weights, use equal weight
    if not weights:
        weights = {s: 1.0 / len(env.symbols) for s in env.symbols}

    action = Action(weights=weights)

    try:
        result = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "task": req.task,
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

# ------------------------------------------------------------------ #
# Grader
# ------------------------------------------------------------------ #

@router.post("/grader")
def grade_episode(req: GraderRequest):
    env = _get_env(req.task)
    result = env.final_score()
    
    # Normalize Sharpe to 0-1 range
    # Target Sharpe values from openenv.yaml: Easy=1.0, Medium=0.9, Hard=1.2
    sharpe_targets = {"easy": 1.0, "medium": 0.9, "hard": 1.2}
    target = sharpe_targets.get(req.task, 1.0)
    
    # Normalized score: min(sharpe / target, 1.0)
    normalized_score = min(1.0, max(0.0, result["sharpe"] / target))

    return {
        "task": req.task,
        "portfolio_value": result["portfolio_value"],
        "sharpe": result["sharpe"],  # Raw Sharpe (for reference)
        "score": round(normalized_score, 4)  # Normalized 0-1 score
    }