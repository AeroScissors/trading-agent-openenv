"""
inference.py — Trading Agent OpenEnv
Emits [START] / [STEP] / [END] logs as required by the OpenEnv validator.
Uses an LLM (once per task) to allocate portfolio weights, then runs full episode.
"""

import os
import json
import requests
from typing import List, Optional
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

TASKS       = ["easy", "medium", "hard"]
MAX_STEPS   = 500   # covers full ~420 trading-day episodes
TEMPERATURE = 0.3
MAX_TOKENS  = 256

SYSTEM_PROMPT = (
    "You are a portfolio manager. Given asset prices, output ONLY a JSON object "
    "mapping each symbol to an allocation weight (0.0-1.0). Weights must sum to ≤ 1.0. "
    "Example: {\"AAPL\": 0.4, \"MSFT\": 0.3, \"GOOGL\": 0.3} "
    "No markdown, no explanation. Just the JSON object."
)

# ── Stdout logging ────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )

# ── LLM: get weights once per task ───────────────────────────────────────────
def get_weights(client: OpenAI, symbols: list, prices: dict) -> dict:
    """Call LLM for weights. Falls back to equal weight on any error."""
    n     = len(symbols)
    equal = {s: round(1.0 / n, 4) for s in symbols}
    try:
        prices_str = ", ".join(f"{s}=${prices.get(s, 0):.2f}" for s in symbols)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Prices: {prices_str}. Allocate weights for max Sharpe ratio."},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text    = (resp.choices[0].message.content or "").strip()
        text    = text.replace("```json", "").replace("```", "").strip()
        weights = json.loads(text)
        weights = {k: max(0.0, float(v)) for k, v in weights.items() if k in symbols}
        if not weights:
            return equal
        total = sum(weights.values())
        if total > 1.0:
            weights = {k: v / total for k, v in weights.items()}
        return weights
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return equal

# ── Env HTTP helpers ──────────────────────────────────────────────────────────
def env_reset(task: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset",  json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(task: str, weights: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step",   json={"task": task, "weights": weights}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_grade(task: str) -> dict:
    r = requests.post(f"{ENV_URL}/grader", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()

# ── Single task episode ───────────────────────────────────────────────────────
def run_task(task: str, client: OpenAI) -> float:
    log_start(task=task, env="trading-agent-openenv", model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    try:
        # 1. Reset
        obs     = env_reset(task).get("observation", {})
        symbols = list(obs.get("prices", {}).keys())
        prices  = obs.get("prices", {})
        done    = False

        # 2. Ask LLM once for weights
        weights = get_weights(client, symbols, prices)
        action_str = json.dumps(weights, separators=(",", ":"))

        # 3. Step loop — reuse same weights for the full episode (fast)
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            result     = env_step(task, weights)
            obs        = result.get("observation", {})
            reward     = float(result.get("reward", 0.0))
            done       = bool(result.get("done", False))
            prices     = obs.get("prices", prices)
            steps_taken = step

            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

        # 4. Grade
        try:
            grade   = env_grade(task)
            score   = float(grade.get("score", 0.0))
            score   = min(max(score, 0.0), 1.0)
            success = score > 0.0
        except Exception as e:
            print(f"[DEBUG] Grader error: {e}", flush=True)
            score   = 0.0
            success = False

    except Exception as e:
        print(f"[DEBUG] Task {task} failed: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    client     = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    all_scores = []

    for task in TASKS:
        score = run_task(task, client)
        all_scores.append(score)
        print(f"[DEBUG] {task} score: {score:.4f}", flush=True)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[DEBUG] Average score: {avg:.4f}", flush=True)

if __name__ == "__main__":
    main()