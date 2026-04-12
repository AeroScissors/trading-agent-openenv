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
        raise HTTPException(
            status_code=400,
            detail=f"No active environment for task='{task}'. Call POST /reset first."
        )
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

    # validate weights
    if any(v < 0 for v in req.weights.values()):
        raise HTTPException(status_code=400, detail="Weights must be non-negative")

    action = Action(weights=req.weights)

    try:
        result = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

    return {
        "task": req.task,
        "portfolio_value": result["portfolio_value"],
        "sharpe": result["sharpe"]
    }