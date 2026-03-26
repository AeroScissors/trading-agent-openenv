import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.core_env import TradingEnv
from env.models import Action

# ------------------------------------------------------------------ #
#  Baseline Agent — MA5 / MA10 Crossover                               #
#  No API key needed. Runs entirely rule-based.                        #
# ------------------------------------------------------------------ #

TRADE_QUANTITY = 1.0       # units to buy/sell per signal


def run_baseline_agent(task: str) -> dict:
    """
    Run the MA crossover baseline agent on a given task.

    Strategy:
        BUY  when MA5 crosses above MA10 (golden cross)
        SELL when MA5 crosses below MA10 (death cross)
        HOLD otherwise

    Args:
        task : "easy", "medium", or "hard"

    Returns:
        dict with score, profit, breakdown, and episode stats
    """
    env   = TradingEnv(task=task)
    state = env.reset()

    total_reward = 0.0
    steps        = 0
    trades       = 0
    done         = False

    print(f"\n{'='*50}")
    print(f"  Baseline Agent — Task: {task.upper()}")
    print(f"{'='*50}")
    print(f"  Starting price : ${state.current_price:.2f}")
    print(f"  Starting cash  : ${state.cash:.2f}")
    print(f"  Total steps    : {env.max_steps}")
    print(f"{'─'*50}")

    while not done:
        # ---- Crossover signal -------------------------------------- #
        if state.ma5 > state.ma10:
            action_str = "BUY"
        elif state.ma5 < state.ma10:
            action_str = "SELL"
        else:
            action_str = "HOLD"

        action = Action(action=action_str, quantity=TRADE_QUANTITY)
        result = env.step(action)

        total_reward += result.reward
        steps        += 1
        done          = result.done
        state         = result.observation

        if action_str in ("BUY", "SELL"):
            trades += 1

        # Print every 50 steps
        if steps % 50 == 0 or done:
            print(
                f"  Step {steps:>3d} | {action_str:4s} | "
                f"price=${state.current_price:.2f} | "
                f"portfolio=${result.info['portfolio_value']:.2f} | "
                f"reward={result.reward:+.4f}"
            )

    # ---- Final score ----------------------------------------------- #
    score_info = env.final_score()

    print(f"{'─'*50}")
    print(f"  Final portfolio : ${result.info['portfolio_value']:.2f}")
    print(f"  Total profit    : ${score_info['profit']:.2f}")
    print(f"  Total reward    : {total_reward:.4f}")
    print(f"  Trades executed : {trades}")
    print(f"  Final score     : {score_info['score']:.4f}")
    print(f"  Breakdown       : {score_info['breakdown']}")
    print(f"{'='*50}")

    return {
        "task":          task,
        "score":         score_info["score"],
        "profit":        score_info["profit"],
        "breakdown":     score_info["breakdown"],
        "total_reward":  round(total_reward, 4),
        "steps":         steps,
        "trades":        trades,
        "final_portfolio": result.info["portfolio_value"],
    }


# ------------------------------------------------------------------ #
#  Run all three tasks                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("\n🤖  Trading Agent OpenEnv — Baseline Agent")
    print("    Strategy: Buy MA5 > MA10 | Sell MA5 < MA10\n")

    all_scores = {}

    for task in ("easy", "medium", "hard"):
        result           = run_baseline_agent(task)
        all_scores[task] = result["score"]

    # ---- Summary --------------------------------------------------- #
    print("\n" + "="*50)
    print("  BASELINE SUMMARY")
    print("="*50)
    for task, score in all_scores.items():
        bar   = "█" * int(score * 20)
        space = "░" * (20 - int(score * 20))
        print(f"  {task:6s} | {bar}{space} | {score:.4f}")

    avg = sum(all_scores.values()) / len(all_scores)
    print(f"{'─'*50}")
    print(f"  Average score : {avg:.4f}")
    print("="*50)