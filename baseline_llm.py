import numpy as np

from env.core_env import TradingEnv
from env.models import Action

TASKS = ["easy", "medium", "hard"]

# ------------------------------------------------------------------ #
# SAFE DEFAULT STRATEGIES (OPTIMAL BASELINES)
# ------------------------------------------------------------------ #

SAFE_DEFAULTS = {
    "easy": {
        "type": "buy_and_hold",
        "entry_step": 1,
        "position_size": 0.95,
        "buy_ma_signal": False,
        "sell_ma_signal": False,
        "stop_loss_pct": 0.7,
        "sell_at_end": False,
    },
    "medium": {
        "type": "buy_and_hold_sell_end",
        "entry_step": 1,
        "position_size": 0.90,
        "buy_ma_signal": False,
        "sell_ma_signal": False,
        "stop_loss_pct": 0.8,
        "sell_at_end": True,
    },
    "hard": {
        "type": "buy_and_hold",
        "entry_step": 1,
        "position_size": 0.90,
        "buy_ma_signal": False,
        "sell_ma_signal": False,
        "stop_loss_pct": 0.75,
        "sell_at_end": False,
    },
}

# ------------------------------------------------------------------ #
# LOCAL STRATEGY GENERATOR (NO API)
# ------------------------------------------------------------------ #

def get_strategy(task: str) -> dict:
    print(f" Using LOCAL strategy for task='{task}'")
    return SAFE_DEFAULTS.get(task, SAFE_DEFAULTS["easy"])

# ------------------------------------------------------------------ #
# INDICATOR
# ------------------------------------------------------------------ #

def compute_momentum(history: list, window: int = 10) -> float:
    if len(history) < 2:
        return 0.0
    w = min(window, len(history))
    seg = np.array(history[-w:], dtype=float)
    slope = np.polyfit(np.arange(w), seg, 1)[0]
    return float(slope / (np.mean(seg) + 1e-8))

# ------------------------------------------------------------------ #
# DECISION ENGINE
# ------------------------------------------------------------------ #

def decide_action(state, step, strategy, entry_price, max_steps):
    price    = state.get("current_price", 0.0)
    position = state.get("position", 0.0)
    cash     = state.get("cash", 0.0)
    ma5      = state.get("ma5", 0.0)
    ma10     = state.get("ma10", 0.0)
    history  = state.get("price_history", [])

    if price <= 0:
        return {"action": "HOLD", "quantity": 0.0}

    entry_step    = strategy["entry_step"]
    pos_size      = strategy["position_size"]
    buy_ma        = strategy["buy_ma_signal"]
    sell_ma       = strategy["sell_ma_signal"]
    stop_loss_pct = strategy["stop_loss_pct"]
    sell_at_end   = strategy["sell_at_end"]

    ma_trend   = (ma5 - ma10) / (abs(ma10) + 1e-8)
    ma_bullish = ma5 > ma10
    ma_bearish = ma5 < ma10
    momentum   = compute_momentum(history)

    # ---------------- ENTRY ---------------- #
    if position == 0 and cash > 0:
        if step == entry_step or (buy_ma and ma_bullish):
            qty = (cash * pos_size) / price
            return {"action": "BUY", "quantity": round(qty, 4)}

        if step == 10:  # fallback
            qty = (cash * pos_size) / price
            return {"action": "BUY", "quantity": round(qty, 4)}

    # ---------------- EXIT ---------------- #
    if position > 0:

        if sell_at_end and step >= max_steps:
            return {"action": "SELL", "quantity": round(position, 4)}

        if entry_price > 0 and price < entry_price * stop_loss_pct:
            return {"action": "SELL", "quantity": round(position, 4)}

        if sell_ma and ma_bearish and price > entry_price:
            return {"action": "SELL", "quantity": round(position, 4)}

    return {"action": "HOLD", "quantity": 0.0}

# ------------------------------------------------------------------ #
# MAIN AGENT LOOP
# ------------------------------------------------------------------ #

def run_agent(task):
    print("\n" + "="*50)
    print(f"Task: {task.upper()}")
    print("="*50)

    env = TradingEnv(task)
    state = env.reset().model_dump()
    strategy = get_strategy(task)

    step = 0
    done = False
    total_reward = 0.0
    entry_price = 0.0
    max_steps = env.max_steps

    while not done:
        step += 1

        action = decide_action(state, step, strategy, entry_price, max_steps)

        if action["action"] == "BUY" and entry_price == 0.0:
            entry_price = state.get("current_price", 0.0)
        elif action["action"] == "SELL":
            entry_price = 0.0

        result = env.step(Action(**action))
        state = result.observation.model_dump()
        done = result.done
        total_reward += result.reward

        if step % 50 == 0:
            print(f"Step {step} | {action['action']} | reward={result.reward:.4f}")

    score_data = env.final_score()

    print(f"\nScore  : {score_data['score']}")
    print(f"Profit : ${score_data['profit']:.2f}")
    print(f"Reward : {total_reward:.4f}")

    return score_data

# ------------------------------------------------------------------ #
# ENTRY POINT
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("FINAL AGENT — LOCAL MODE 🚀")

    results = {}

    for task in TASKS:
        results[task] = run_agent(task)

    print("\nFINAL RESULTS")
    for k, v in results.items():
        print(f"{k}: {v['score']}")

    avg = sum(v["score"] for v in results.values()) / len(results)
    print(f"AVG: {avg:.4f}")