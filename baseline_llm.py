import numpy as np
from env.core_env import TradingEnv
from env.models import Action

TASKS = ["easy", "medium", "hard"]

# ─────────────────────────────────────────────
# Per-task config
# ─────────────────────────────────────────────
TASK_CONFIG = {
    "easy": {
        "pos_size":         0.95,
        "stop_loss_pct":    0.85,
        "use_ma_signal":    False,  # pure buy-and-hold
        "death_cross_conf": 1,
        "sell_at_end":      False,
        "take_profit_pct":  1.20,
    },
    "medium": {
        "pos_size":         0.90,
        "stop_loss_pct":    0.88,
        "use_ma_signal":    False,  # ← was True, kill the noise entirely
        "death_cross_conf": 3,   # was 1, now 3 — filters out MSFT noise
        "sell_at_end":      True,
        "take_profit_pct":  1.20,
    },
    "hard": {
        "pos_size":         0.90,
        "stop_loss_pct":    0.80,
        "use_ma_signal":    True,
        "death_cross_conf": 2,      # need 2 consecutive steps below MA before exiting
        "sell_at_end":      False,
        "take_profit_pct":  999.0,
    },
}

# ─────────────────────────────────────────────
# Agent state (per episode)
# ─────────────────────────────────────────────
class AgentState:
    def __init__(self, config):
        self.config          = config
        self.entry_price     = 0.0
        self.prev_ma5        = None
        self.prev_ma10       = None
        self.death_cross_cnt = 0

    def reset(self):
        self.entry_price     = 0.0
        self.prev_ma5        = None
        self.prev_ma10       = None
        self.death_cross_cnt = 0

# ─────────────────────────────────────────────
# Decision logic
# ─────────────────────────────────────────────
def decide_action(state: dict, step: int, agent_state: AgentState, max_steps: int) -> dict:
    price    = state.get("current_price", 0.0)
    cash     = state.get("cash", 0.0)
    position = state.get("position", 0.0)
    ma5      = state.get("ma5")
    ma10     = state.get("ma10")
    cfg      = agent_state.config

    if price <= 0:
        return {"action": "HOLD", "quantity": 0.0}

    def buy_all():
        qty = (cash * cfg["pos_size"]) / price
        agent_state.entry_price     = price
        agent_state.death_cross_cnt = 0
        return {"action": "BUY", "quantity": round(qty, 4)}

    def sell_all():
        agent_state.entry_price     = 0.0
        agent_state.death_cross_cnt = 0
        return {"action": "SELL", "quantity": round(position, 4)}

    # ── crossover detection ───────────────────
    golden_cross = False
    death_cross  = False

    if (ma5 is not None and ma10 is not None
            and agent_state.prev_ma5 is not None
            and agent_state.prev_ma10 is not None):
        prev_above = agent_state.prev_ma5 > agent_state.prev_ma10
        curr_above = ma5 > ma10
        if not prev_above and curr_above:
            golden_cross = True
        if prev_above and not curr_above:
            death_cross = True

    if ma5  is not None: agent_state.prev_ma5  = ma5
    if ma10 is not None: agent_state.prev_ma10 = ma10

    # ══════════════════════════════════════════
    # EASY / MEDIUM — non-MA signal primary
    # ══════════════════════════════════════════
    if not cfg["use_ma_signal"]:
        if position == 0 and cash > 0:
            if step == 1 or golden_cross:
                if golden_cross and step > 1:
                    print(f"  [RE-ENTRY CROSS] step={step} MA5={ma5:.2f} MA10={ma10:.2f}")
                return buy_all()
                
        if position > 0 and agent_state.entry_price > 0:
            # take profit
            if price > agent_state.entry_price * cfg["take_profit_pct"]:
                print(f"  [TAKE PROFIT] step={step} entry={agent_state.entry_price:.2f} now={price:.2f}")
                return sell_all()

            if price < agent_state.entry_price * cfg["stop_loss_pct"]:
                print(f"  [STOP-LOSS] step={step} entry={agent_state.entry_price:.2f} now={price:.2f}")
                return sell_all()
            
            # end-of-episode liquidation
            if cfg["sell_at_end"] and step >= max_steps - 1:
                print(f"  [END SELL] step={step}")
                return sell_all()
                
        return {"action": "HOLD", "quantity": 0.0}

    # ══════════════════════════════════════════
    # HARD — signal-based
    # ══════════════════════════════════════════

    # ENTRY
    if position == 0 and cash > 0:
        if step == 1:                  # buy immediately before MAs are ready
            return buy_all()
        if golden_cross:
            print(f"  [GOLDEN CROSS] step={step} MA5={ma5:.2f} MA10={ma10:.2f}")
            return buy_all()

    # EXIT
    if position > 0:

        # take profit
        if agent_state.entry_price > 0 and price > agent_state.entry_price * cfg["take_profit_pct"]:
            print(f"  [TAKE PROFIT] step={step} entry={agent_state.entry_price:.2f} now={price:.2f}")
            return sell_all()

        # hard stop-loss
        if agent_state.entry_price > 0 and price < agent_state.entry_price * cfg["stop_loss_pct"]:
            print(f"  [STOP-LOSS] step={step} entry={agent_state.entry_price:.2f} now={price:.2f}")
            return sell_all()

        # death cross — with confirmation window
        if death_cross or (ma5 is not None and ma10 is not None and ma5 < ma10):
            agent_state.death_cross_cnt += 1
        else:
            agent_state.death_cross_cnt = 0

        if agent_state.death_cross_cnt >= cfg["death_cross_conf"]:
            print(f"  [DEATH CROSS] step={step} MA5={ma5:.2f} MA10={ma10:.2f} conf={agent_state.death_cross_cnt}")
            return sell_all()

        # end-of-episode liquidation
        if cfg["sell_at_end"] and step >= max_steps - 1:
            print(f"  [END SELL] step={step}")
            return sell_all()

    return {"action": "HOLD", "quantity": 0.0}

# ─────────────────────────────────────────────
# Run one task
# ─────────────────────────────────────────────
def run_agent(task: str) -> dict:
    print("\n" + "="*50)
    print(f"Task: {task.upper()}")
    print("="*50)

    env         = TradingEnv(task)
    state       = env.reset().model_dump()
    max_steps   = env.max_steps
    agent_state = AgentState(TASK_CONFIG[task])
    agent_state.reset()

    step         = 0
    done         = False
    total_reward = 0.0

    while not done:
        step += 1
        action = decide_action(state, step, agent_state, max_steps)

        result       = env.step(Action(**action))
        state        = result.observation.model_dump()
        done         = result.done
        total_reward += result.reward

        if step % 50 == 0 or done:
            print(f"  Step {step:>3} | {action['action']:<4} qty={action['quantity']:.4f} "
                  f"| reward={result.reward:.4f} | pos={state.get('position', 0):.4f} "
                  f"cash={state.get('cash', 0):.2f}")

    score_data = env.final_score()
    print(f"\n  Score  : {score_data['score']:.4f}")
    print(f"  Profit : ${score_data['profit']:.2f}")
    print(f"  Reward : {total_reward:.4f}")
    return score_data

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("UPGRADED LOCAL AGENT — MA CROSSOVER + RISK MGMT 🚀")

    results = {}
    for task in TASKS:
        results[task] = run_agent(task)

    print("\n" + "="*50)
    print("FINAL SCORES")
    print("="*50)
    for t, v in results.items():
        print(f"  {t:<8}: {v['score']:.4f}  (profit=${v['profit']:.2f})")

    avg = sum(v["score"] for v in results.values()) / len(results)
    print(f"  {'AVG':<8}: {avg:.4f}")