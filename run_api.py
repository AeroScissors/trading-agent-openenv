import requests

BASE_URL = "http://localhost:8000"
TASKS = ["easy", "medium", "hard"]

MAX_STEPS = {"easy": 249, "medium": 249, "hard": 249}

SAFE_DEFAULTS = {
    "easy":   {"pos_size": 0.95, "stop_loss_pct": 0.85, "sell_end": False, "take_profit": 1.20},
    "medium": {"pos_size": 0.90, "stop_loss_pct": 0.05, "sell_end": True,  "take_profit": 999.0},  # ← stop_loss basically disabled, take_profit disabled
    "hard":   {"pos_size": 0.90, "stop_loss_pct": 0.80, "sell_end": False,  "take_profit": 999.0},
}

def get_strategy(task):
    return SAFE_DEFAULTS[task]

# ==============================
# HYBRID DECISION LOGIC 🧠🔥
# ==============================
def decide_action(state, step, strategy, agent_state, max_steps):
    price    = state.get("current_price", 0.0)
    cash     = state.get("cash", 0.0)
    position = state.get("position", 0.0)
    ma5      = state.get("ma5")
    ma10     = state.get("ma10")
    sharpe   = state.get("sharpe", 1.0)

    if price <= 0:
        return {"action": "HOLD", "quantity": 0.0}

    def buy():
        qty = (cash * strategy["pos_size"]) / price
        agent_state["entry_price"] = price
        agent_state["death_cnt"] = 0
        return {"action": "BUY", "quantity": round(qty, 4)}

    def sell():
        agent_state["entry_price"] = 0.0
        agent_state["death_cnt"] = 0
        return {"action": "SELL", "quantity": round(position, 4)}

    # =========================
    # CROSS DETECTION
    # =========================
    golden = False
    death  = False

    if (ma5 is not None and ma10 is not None
        and agent_state["prev_ma5"] is not None
        and agent_state["prev_ma10"] is not None):

        prev_above = agent_state["prev_ma5"] > agent_state["prev_ma10"]
        curr_above = ma5 > ma10

        if not prev_above and curr_above:
            golden = True
        if prev_above and not curr_above:
            death = True

    agent_state["prev_ma5"]  = ma5
    agent_state["prev_ma10"] = ma10

    # =========================
    # ENTRY (guaranteed + smart)
    # =========================
    if position == 0 and cash > 0:
        if step == 1:
            return buy()

        if golden:
            return buy()

        if step == 10:
            return buy()

    # =========================
    # EXIT (risk system)
    # =========================
    if position > 0:

        entry = agent_state["entry_price"]

        # TAKE PROFIT
        if entry > 0 and price > entry * strategy["take_profit"]:
            return sell()

        # STOP LOSS
        if entry > 0 and price < entry * strategy["stop_loss_pct"]:
            return sell()

        # VOLATILITY EXIT (NEW 🔥)
        if sharpe < 0.3 and entry > 0 and price < entry * 0.95:
            return sell()

        # DEATH CROSS CONFIRMATION
        if death or (ma5 is not None and ma10 is not None and ma5 < ma10):
            agent_state["death_cnt"] += 1
        else:
            agent_state["death_cnt"] = 0

        if agent_state["death_cnt"] >= 3:
            return sell()

        # END SELL
        if strategy["sell_end"] and step >= max_steps - 1:
            return sell()

    return {"action": "HOLD", "quantity": 0.0}

# ==============================
# RUN AGENT
# ==============================
def run_agent(task):
    print("\n" + "="*50)
    print(f"Task: {task.upper()}")
    print("="*50)

    r = requests.post(f"{BASE_URL}/reset", json={"task": task})
    state = r.json().get("initial_state")

    strategy = get_strategy(task)
    max_steps = MAX_STEPS[task]

    agent_state = {
        "entry_price": 0.0,
        "prev_ma5": None,
        "prev_ma10": None,
        "death_cnt": 0,
    }

    step = 0
    done = False
    total_reward = 0.0

    while not done:
        step += 1

        # MEDIUM: pure buy-and-hold, one clean trade
        if task == "medium":
            if step == 1:
                qty = (state.get("cash", 0) * 0.90) / state.get("current_price", 1)
                action = {"action": "BUY", "quantity": round(qty, 4)}
            elif step >= max_steps - 1:
                action = {"action": "SELL", "quantity": round(state.get("position", 0), 4)}
            else:
                action = {"action": "HOLD", "quantity": 0.0}
        else:
            action = decide_action(state, step, strategy, agent_state, max_steps)

        r = requests.post(f"{BASE_URL}/step", json={
            "task": task,
            "action": action["action"],
            "quantity": float(action["quantity"]),
        })

        result = r.json()
        state = result.get("observation")
        reward = result.get("reward", 0.0)
        done = result.get("done", False)

        total_reward += reward

        if step % 50 == 0 or done:
            print(f"Step {step} | {action['action']} | reward={reward:.4f}")

    score = requests.post(f"{BASE_URL}/grader", json={"task": task}).json()

    print(f"\nScore  : {score['score']}")
    print(f"Profit : ${score['profit']:.2f}")
    print(f"Reward : {total_reward:.4f}")

    return score

# ==============================
# ENTRY
# ==============================
if __name__ == "__main__":
    print("HYBRID AGENT — API MODE 🚀")

    results = {}

    for task in TASKS:
        results[task] = run_agent(task)

    print("\nFINAL RESULTS")
    for t, v in results.items():
        print(f"{t}: {v['score']}")

    avg = sum(v["score"] for v in results.values()) / len(results)
    print(f"AVG: {avg:.4f}")