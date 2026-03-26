import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.tasks import easy, medium, hard

# ------------------------------------------------------------------ #
#  Master Grader                                                       #
#  Routes grading to the correct task grader and returns 0.0 – 1.0   #
# ------------------------------------------------------------------ #

SUPPORTED_TASKS = ("easy", "medium", "hard")


def grade(
    task: str,
    final_portfolio: float,
    initial_cash: float = 10_000.0,
    portfolio_history: list[float] = None,
    trade_log: list[dict] = None,
) -> dict:
    """
    Master grading function — delegates to per-task grader.

    Args:
        task              : "easy", "medium", or "hard"
        final_portfolio   : total portfolio value at episode end
        initial_cash      : starting capital (default 10,000)
        portfolio_history : list of portfolio values per step (needed for hard)
        trade_log         : list of trade dicts (needed for medium)

    Returns:
        dict with keys:
            score       → float in [0.0, 1.0]
            task        → task id string
            profit      → raw profit in USD
            breakdown   → task-specific sub-scores
    """

    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from {SUPPORTED_TASKS}")

    profit = final_portfolio - initial_cash

    # ---- Easy ---------------------------------------------------- #
    if task == "easy":
        score = easy.grade(final_portfolio, initial_cash)
        breakdown = {
            "profit_score": score,
            "profit_target": easy.PROFIT_TARGET,
        }

    # ---- Medium -------------------------------------------------- #
    elif task == "medium":
        score = medium.grade(final_portfolio, trade_log or [], initial_cash)
        profit_score = min(1.0, max(0.0, profit / medium.PROFIT_TARGET))
        efficiency   = medium.compute_trade_efficiency(trade_log or [])
        breakdown = {
            "profit_score":     round(profit_score, 4),
            "trade_efficiency": round(efficiency, 4),
            "profit_target":    medium.PROFIT_TARGET,
        }

    # ---- Hard ---------------------------------------------------- #
    elif task == "hard":
        score = hard.grade(final_portfolio, portfolio_history or [], initial_cash)
        profit_score = min(1.0, max(0.0, profit / hard.PROFIT_TARGET))
        sharpe       = hard.compute_sharpe(portfolio_history or [final_portfolio])
        sharpe_score = min(1.0, max(0.0, sharpe / hard.SHARPE_TARGET))
        breakdown = {
            "profit_score": round(profit_score, 4),
            "sharpe":       round(sharpe, 4),
            "sharpe_score": round(sharpe_score, 4),
            "profit_target": hard.PROFIT_TARGET,
            "sharpe_target": hard.SHARPE_TARGET,
        }

    return {
        "score":     score,
        "task":      task,
        "profit":    round(profit, 2),
        "breakdown": breakdown,
    }


def grade_all(results: dict) -> dict:
    """
    Grade all three tasks at once.

    Args:
        results: dict keyed by task id, each value is a dict with:
                 final_portfolio, initial_cash, portfolio_history, trade_log

    Returns:
        dict keyed by task id → grade result + overall average score
    """
    scores = {}
    for task in SUPPORTED_TASKS:
        if task not in results:
            continue
        r = results[task]
        scores[task] = grade(
            task              = task,
            final_portfolio   = r.get("final_portfolio", 10_000.0),
            initial_cash      = r.get("initial_cash", 10_000.0),
            portfolio_history = r.get("portfolio_history"),
            trade_log         = r.get("trade_log"),
        )

    if scores:
        avg = round(sum(v["score"] for v in scores.values()) / len(scores), 4)
    else:
        avg = 0.0

    return {"tasks": scores, "average_score": avg}


# ------------------------------------------------------------------ #
#  Quick test                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=== Grader self-test ===\n")

    # Easy
    result = grade("easy", final_portfolio=10_600.0)
    print(f"Easy   → score={result['score']}  profit=${result['profit']}  breakdown={result['breakdown']}")

    # Medium
    trades = [
        {"action": "BUY",  "price": 300.0, "quantity": 10},
        {"action": "SELL", "price": 320.0, "quantity": 10},
    ]
    result = grade("medium", final_portfolio=10_900.0, trade_log=trades)
    print(f"Medium → score={result['score']}  profit=${result['profit']}  breakdown={result['breakdown']}")

    # Hard
    history = [10_000 * (1 + 0.002 * i) for i in range(252)]
    result  = grade("hard", final_portfolio=history[-1], portfolio_history=history)
    print(f"Hard   → score={result['score']}  profit=${result['profit']}  breakdown={result['breakdown']}")