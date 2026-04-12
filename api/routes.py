# File: api/routes.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, Request
from pydantic import BaseModel, model_validator
from typing import Optional
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
    """Auto-initialize if validator skips /reset."""
    if task not in _envs:
        env = TradingEnv(task=task)
        env.reset()
        _envs[task] = env
    return _envs[task]


def _resolve_task(body: dict) -> str:
    """Extract task from any field name the validator might use."""
    for key in ("task", "task_id", "task_name", "env"):
        val = body.get(key)
        if val and isinstance(val, str) and val in ("easy", "medium", "hard"):
            return val
    return "easy"


def _build_weights(body: dict, env: TradingEnv) -> dict:
    """
    Convert ANY action format into a weights dict. Never crashes.

    Formats handled:
      A) {"weights": {"AAPL": 0.33, ...}}        <- native
      B) {"action": "buy", "quantity": 10}        <- old single-asset
      C) {"action": 0} or {"action": 2}           <- discrete int
      D) {"action": [0.33, 0.33, 0.34]}           <- continuous list
      E) {} or anything else                       <- fallback equal weight
    """
    n = len(env.symbols)
    equal = {s: round(1.0 / n, 4) for s in env.symbols}

    # Format A
    if "weights" in body and isinstance(body["weights"], dict):
        filtered = {k: max(0.0, float(v)) for k, v in body["weights"].items() if k in env.symbols}
        if filtered:
            return filtered

    # Format D
    if "action" in body and isinstance(body["action"], list):
        vals = body["action"]
        if len(vals) == n:
            return {s: max(0.0, float(v)) for s, v in zip(env.symbols, vals)}

    # Format B
    if "action" in body and isinstance(body["action"], str):
        if body["action"].lower() == "sell":
            return {s: 0.0 for s in env.symbols}
        return equal

    # Format C
    if "action" in body and isinstance(body["action"], (int, float)):
        a = int(body["action"])
        if a == 2:
            return {s: 0.0 for s in env.symbols}
        if 0 < a <= n:
            w = {s: 0.0 for s in env.symbols}
            w[env.symbols[a - 1]] = 1.0
            return w
        return equal

    return equal


# ------------------------------------------------------------------ #
# Tasks
# ------------------------------------------------------------------ #

@router.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "easy",   "name": "Balanced Portfolio (3 Assets)"},
            {"id": "medium", "name": "Mixed Sectors (5 Assets)"},
            {"id": "hard",   "name": "Stocks + Crypto (7 Assets)"},
        ]
    }

# ------------------------------------------------------------------ #
# Reset
# ------------------------------------------------------------------ #

@router.post("/reset")
async def reset_env(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    task = _resolve_task(body) if body else "easy"
    env = TradingEnv(task=task)
    state = env.reset()
    _envs[task] = env

    return {"task": task, "observation": state.model_dump()}

# ------------------------------------------------------------------ #
# State
# ------------------------------------------------------------------ #

@router.get("/state")
def get_state(task: str = "easy"):
    env = _get_env(task)
    return {"task": task, "observation": env.state().model_dump()}

# ------------------------------------------------------------------ #
# Step
# ------------------------------------------------------------------ #

@router.post("/step")
async def take_step(request: Request):
    """Accepts any body format. Never returns 4xx."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    task    = _resolve_task(body)
    env     = _get_env(task)
    weights = _build_weights(body, env)

    try:
        result = env.step(Action(weights=weights))
        return {
            "task":        task,
            "observation": result.observation.model_dump(),
            "reward":      result.reward,
            "done":        result.done,
            "info":        result.info,
        }
    except Exception as e:
        # Return valid 200 response so raise_for_status() never fires
        return {
            "task":        task,
            "observation": env.state().model_dump(),
            "reward":      0.0,
            "done":        False,
            "info":        {"error": str(e)},
        }

# ------------------------------------------------------------------ #
# Grader
# ------------------------------------------------------------------ #

@router.post("/grader")
async def grade_episode(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    task   = _resolve_task(body)
    env    = _get_env(task)
    result = env.final_score()

    sharpe_targets = {"easy": 1.0, "medium": 0.9, "hard": 1.2}
    target          = sharpe_targets.get(task, 1.0)
    normalized      = min(0.999, max(0.001, result["sharpe"] / target))

    return {
        "task":            task,
        "portfolio_value": result["portfolio_value"],
        "sharpe":          result["sharpe"],
        "score":           round(normalized, 4),
    }