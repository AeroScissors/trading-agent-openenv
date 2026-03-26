import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from env.core_env import TradingEnv
from env.models import Action
from env.tasks.easy import TASK_INFO as EASY_INFO
from env.tasks.medium import TASK_INFO as MEDIUM_INFO
from env.tasks.hard import TASK_INFO as HARD_INFO

# ------------------------------------------------------------------ #
#  Router + shared environment state                                   #
# ------------------------------------------------------------------ #

router = APIRouter()

# One environment instance per session (stateful)
_envs: dict[str, TradingEnv] = {}


def _get_env(task: str) -> TradingEnv:
    """Return existing env for task or raise 400 if not reset yet."""
    if task not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"No active environment for task='{task}'. Call POST /reset first."
        )
    return _envs[task]


# ------------------------------------------------------------------ #
#  Request / Response schemas                                          #
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    task:     str   = "easy"
    action:   str   = "HOLD"
    quantity: float = 0.0


class GraderRequest(BaseModel):
    task: str = "easy"


class BaselineRequest(BaseModel):
    task: Optional[str] = None   # None = run all tasks


# ------------------------------------------------------------------ #
#  GET /tasks  — list all tasks                                        #
# ------------------------------------------------------------------ #

@router.get("/tasks")
def list_tasks():
    """Return metadata for all three tasks."""
    return {
        "tasks": [EASY_INFO, MEDIUM_INFO, HARD_INFO]
    }


# ------------------------------------------------------------------ #
#  POST /reset  — reset environment                                    #
# ------------------------------------------------------------------ #

@router.post("/reset")
def reset_env(req: ResetRequest):
    """
    Reset the environment for a given task.
    Returns the initial state.
    """
    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task '{req.task}'")

    env = TradingEnv(task=req.task)
    initial_state = env.reset()
    _envs[req.task] = env

    return {
        "message":       f"Environment reset for task='{req.task}'",
        "task":          req.task,
        "initial_state": initial_state.model_dump(),
    }


# ------------------------------------------------------------------ #
#  GET /state  — current state snapshot                                #
# ------------------------------------------------------------------ #

@router.get("/state")
def get_state(task: str = "easy"):
    """Return current state without advancing the timestep."""
    env = _get_env(task)
    return {
        "task":  task,
        "state": env.state().model_dump(),
    }


# ------------------------------------------------------------------ #
#  POST /step  — take one action                                       #
# ------------------------------------------------------------------ #

@router.post("/step")
def take_step(req: StepRequest):
    """
    Execute one action in the environment.
    Returns observation, reward, done flag, and info dict.
    """
    env = _get_env(req.task)

    if req.action.upper() not in ("BUY", "SELL", "HOLD"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{req.action}'. Must be BUY, SELL, or HOLD."
        )

    action = Action(action=req.action.upper(), quantity=req.quantity)

    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "task":        req.task,
        "observation": result.observation.model_dump(),
        "reward":      result.reward,
        "done":        result.done,
        "info":        result.info,
    }


# ------------------------------------------------------------------ #
#  POST /grader  — final score                                         #
# ------------------------------------------------------------------ #

@router.post("/grader")
def grade_episode(req: GraderRequest):
    """
    Compute and return the final score for the current episode.
    Can be called at any time (not just when done=True).
    """
    env    = _get_env(req.task)
    result = env.final_score()

    return {
        "task":      req.task,
        "score":     result["score"],
        "profit":    result["profit"],
        "breakdown": result["breakdown"],
    }


# ------------------------------------------------------------------ #
#  POST /baseline  — run rule-based baseline agent                     #
# ------------------------------------------------------------------ #

@router.post("/baseline")
def run_baseline(req: BaselineRequest):
    """
    Run the MA5 > MA10 crossover baseline agent.
    If task is None, runs all three tasks and returns all scores.
    """
    from baseline import run_baseline_agent

    tasks_to_run = ("easy", "medium", "hard") if req.task is None else (req.task,)
    results      = {}

    for task in tasks_to_run:
        score_info       = run_baseline_agent(task)
        results[task]    = score_info

    return {
        "agent":   "MA Crossover Baseline",
        "results": results,
    }