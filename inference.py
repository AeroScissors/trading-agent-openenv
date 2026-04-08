"""
inference.py — Trading-Agent-OpenEnv
LLM-based agent using OpenAI client format.

Required env vars:
    API_BASE_URL  — LLM API base URL
    MODEL_NAME    — model identifier
    HF_TOKEN      — API key / Hugging Face token

Usage:
    python inference.py
"""

import os
import re
import json
import requests
from openai import OpenAI

# ------------------------------------------------------------------ #
#  Config                                                             #
# ------------------------------------------------------------------ #

API_BASE_URL  = os.environ.get("API_BASE_URL",  "https://api-inference.huggingface.co/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN      = os.environ.get("HF_TOKEN",      "")

ENV_BASE_URL  = os.environ.get("ENV_BASE_URL",  "http://localhost:7860")

TASKS         = ["easy", "medium", "hard"]
MAX_STEPS     = {"easy": 249, "medium": 249, "hard": 363}
LLM_EVERY_N   = 10        # call LLM every N steps (saves time + cost)
TEMPERATURE   = 0.1
MAX_TOKENS    = 64
FALLBACK      = "HOLD"

# ------------------------------------------------------------------ #
#  OpenAI client                                                      #
# ------------------------------------------------------------------ #

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ------------------------------------------------------------------ #
#  Prompts                                                            #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are a trading agent. You will receive the current market state and must decide to BUY, SELL, or HOLD.

Rules:
- BUY: enter a long position when you expect price to rise
- SELL: exit your position when you expect price to fall or to lock in profit
- HOLD: do nothing

Respond with ONLY one word: BUY, SELL, or HOLD. No explanation."""


def build_user_prompt(task: str, step: int, state: dict, max_steps: int) -> str:
    price    = state.get("current_price", 0)
    cash     = state.get("cash", 0)
    position = state.get("position", 0)
    ma5      = state.get("ma5", 0)
    ma10     = state.get("ma10", 0)
    sharpe   = state.get("sharpe", 0)

    portfolio = cash + position * price

    return f"""Task: {task.upper()} | Step: {step}/{max_steps}

Market State:
- Current price : ${price:,.2f}
- MA5           : ${ma5:,.2f}
- MA10          : ${ma10:,.2f}
- Sharpe ratio  : {sharpe:.4f}

Portfolio:
- Cash          : ${cash:,.2f}
- Position      : {position:.4f} units
- Portfolio val : ${portfolio:,.2f}

What is your action? Reply with BUY, SELL, or HOLD only."""


# ------------------------------------------------------------------ #
#  LLM call                                                           #
# ------------------------------------------------------------------ #

def get_llm_action(task: str, step: int, state: dict, max_steps: int) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(task, step, state, max_steps)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response = completion.choices[0].message.content or ""
        response = response.strip().upper()

        # extract first valid action word
        for word in re.split(r'\W+', response):
            if word in ("BUY", "SELL", "HOLD"):
                return word

        return FALLBACK

    except Exception as e:
        print(f"  [LLM error] {e} — using FALLBACK ({FALLBACK})")
        return FALLBACK


# ------------------------------------------------------------------ #
#  Env API helpers                                                    #
# ------------------------------------------------------------------ #

def env_reset(task: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task})
    r.raise_for_status()
    return r.json().get("initial_state", {})


def env_step(task: str, action: str, quantity: float) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={
        "task": task, "action": action, "quantity": quantity
    })
    r.raise_for_status()
    return r.json()


def env_grader(task: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/grader", json={"task": task})
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ #
#  Run agent for one task                                             #
# ------------------------------------------------------------------ #

def run_task(task: str) -> dict:
    print(f"\n{'='*50}")
    print(f"Task: {task.upper()}")
    print(f"{'='*50}")

    state    = env_reset(task)
    max_steps = MAX_STEPS[task]

    current_action = FALLBACK
    step           = 0
    done           = False
    total_reward   = 0.0

    steps_log = []

    while not done:
        step += 1

        if step == 1 or step % LLM_EVERY_N == 0:
            current_action = get_llm_action(task, step, state, max_steps)

        price    = state.get("current_price", 1)
        cash     = state.get("cash", 0)
        position = state.get("position", 0)

        if current_action == "BUY":
            quantity = round((cash * 0.95) / max(price, 1e-8), 4)
        elif current_action == "SELL":
            quantity = round(position, 4)
        else:
            quantity = 0.0

        result       = env_step(task, current_action, quantity)
        done         = result.get("done", False)
        state        = result.get("observation", {})
        reward       = result.get("reward", 0.0)
        total_reward += reward

        portfolio = state.get("cash", 0) + state.get("position", 0) * state.get("current_price", 0)

        steps_log.append({
            "step": step,
            "action": current_action,
            "reward": round(reward, 6),
            "portfolio": round(portfolio, 2),
        })

        if step % 50 == 0 or done:
            print(f"  Step {step:3d} | {current_action:4s} | reward={reward:+.4f} | portfolio=${portfolio:,.2f}")

    grade = env_grader(task)
    print(f"\n  Score  : {grade['score']}")
    print(f"  Profit : ${grade['profit']:,.2f}")
    print(f"  Reward : {total_reward:.4f}")

    # --- structured output block ---
    print(f"[START]")
    for s in steps_log:
        print(f"[STEP] task={task} step={s['step']} action={s['action']} reward={s['reward']} portfolio={s['portfolio']}")
    print(f"[END] task={task} score={grade['score']} profit={round(grade['profit'],2)} total_reward={round(total_reward,4)}")

    return grade


def main():
    print("=" * 50)
    print("TRADING-AGENT-OPENENV — LLM INFERENCE")
    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
    print(f"Env   : {ENV_BASE_URL}")
    print("=" * 50)

    if not HF_TOKEN:
        print("\n⚠️  WARNING: HF_TOKEN not set. LLM calls may fail.")

    results = {}
    for task in TASKS:
        results[task] = run_task(task)

    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    for task, result in results.items():
        print(f"  {task:6s} : {result['score']:.4f}")

    avg = sum(r["score"] for r in results.values()) / len(results)
    print(f"\n  AVG    : {avg:.4f}")
    print(f"{'='*50}")

    output = {
        "scores": {t: r["score"] for t, r in results.items()},
        "average": round(avg, 4),
        "model": MODEL_NAME,
    }
    print(f"\nJSON output:\n{json.dumps(output, indent=2)}")