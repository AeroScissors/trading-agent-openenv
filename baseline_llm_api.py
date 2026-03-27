import requests

BASE_URL = "http://localhost:8000"
TASKS = ["easy", "medium", "hard"]

SAFE_DEFAULTS = {
    "easy":   {"entry_step": 1, "pos_size": 0.95, "sell_end": False, "stop_loss_pct": 0.70},
    "medium": {"entry_step": 1, "pos_size": 0.90, "sell_end": True,  "stop_loss_pct": 0.80},
    "hard":   {"entry_step": 1, "pos_size": 0.90, "sell_end": False, "stop_loss_pct": 0.75},
}

# per-task known max_steps (from local env)
MAX_STEPS = {"easy": 249, "medium": 249, "hard": 249}

def get_strategy(task):
    return SAFE_DEFAULTS.get(task, SAFE_DEFAULTS["easy"])

def decide_action(state, step, strategy, entry_price, max_steps):
    price    = state.get("current_price", 0.0)
    cash     = state.get("cash", 0.0)
    position = state.get("position", 0.0)

    if price <= 0:
        return {"action": "HOLD", "quantity": 0.0}

    pos_size      = strategy["pos_size"]
    stop_loss_pct = strategy["stop_loss_pct"]
    sell_end      = strategy["sell_end"]

    # -------- ENTRY --------
    if position == 0 and cash > 0:
        if step == strategy["entry_step"] or step == 10:
            qty = (cash * pos_size) / price
            return {"action": "BUY", "quantity": round(qty, 4)}

    # -------- EXIT --------
    if position > 0:

        # medium: sell right before end to be safe
        if sell_end and step >= max_steps - 1:
            return {"action": "SELL", "quantity": round(position, 4)}

        # stop loss
        if entry_price > 0 and price < entry_price * stop_loss_pct:
            return {"action": "SELL", "quantity": round(position, 4)}

    return {"action": "HOLD", "quantity": 0.0}

def run_agent(task):
    print("\n" + "="*50)
    print(f"Task: {task.upper()}")
    print("="*50)

    r = requests.post(f"{BASE_URL}/reset", json={"task": task})
    reset_data = r.json()
    state = reset_data.get("initial_state") or reset_data.get("state")
    max_steps = MAX_STEPS[task]

    strategy = get_strategy(task)

    step = 0
    done = False
    total_reward = 0.0
    entry_price = 0.0

    while not done:
        step += 1

        action = decide_action(state, step, strategy, entry_price, max_steps)

        if action["action"] == "BUY" and entry_price == 0.0:
            entry_price = state.get("current_price", 0.0)
        elif action["action"] == "SELL":
            entry_price = 0.0

        r = requests.post(f"{BASE_URL}/step", json={
            "task": task,
            "action": action["action"],
            "quantity": float(action["quantity"]),
        })

        result = r.json()
        state = result.get("observation") or result.get("state")
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        total_reward += reward

        if step % 50 == 0 or done:
            print(f"Step {step} | {action['action']} | reward={reward:.4f} | done={done}")

    r = requests.post(f"{BASE_URL}/grader", json={"task": task})
    score_data = r.json()

    print(f"\nScore  : {score_data['score']}")
    print(f"Profit : ${score_data['profit']:.2f}")
    print(f"Reward : {total_reward:.4f}")

    return score_data

if __name__ == "__main__":
    print("FINAL AGENT — API MODE 🚀")

    results = {}

    for task in TASKS:
        results[task] = run_agent(task)

    print("\nFINAL RESULTS")
    for t, v in results.items():
        print(f"{t}: {v['score']}")

    avg = sum(v["score"] for v in results.values()) / len(results)
    print(f"AVG: {avg:.4f}")