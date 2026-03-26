# medium.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.fetch_data import fetch_prices
import numpy as np

# ------------------------------------------------------------------ #
#  Task 2 — React to Signals (Medium)                                  #
# ------------------------------------------------------------------ #

TASK_ID          = "medium"
TASK_NAME        = "React to Signals"
TASK_DESCRIPTION = (
    "A sideways-to-moderate trend stock (MSFT, 1 year). "
    "MA5 and MA10 indicators are provided in state. "
    "Use crossover signals to time entries and exits efficiently."
)

INITIAL_CASH       = 10_000.0
MAX_STEPS          = None
PROFIT_TARGET      = 800.0    # harder to achieve than easy
EFFICIENCY_WINDOW  = 10       # trades within this window count as efficient


def load_data() -> list[float]:
    """Return closing price series for the medium task."""
    return fetch_prices("medium", period_days=365)


def compute_trade_efficiency(trade_log: list[dict]) -> float:
    """
    Measure how well the agent times its trades.

    Efficiency = ratio of profitable trades to total trades.
    Returns 0.0 if no trades were made.

    Args:
        trade_log: list of dicts with keys 'action', 'price', 'quantity'

    Returns:
        float in [0.0, 1.0]
    """
    if not trade_log:
        return 0.0

    profitable = 0
    total      = 0
    buy_price  = None

    for trade in trade_log:
        if trade["action"] == "BUY":
            buy_price = trade["price"]

        elif trade["action"] == "SELL" and buy_price is not None:
            total += 1
            if trade["price"] > buy_price:
                profitable += 1
            buy_price = None

    if total == 0:
        return 0.0

    return round(profitable / total, 4)


def grade(
    final_portfolio: float,
    trade_log: list[dict] = None,
    initial_cash: float = INITIAL_CASH
) -> float:
    """
    Score = 0.5 * profit_score + 0.5 * trade_efficiency

    Args:
        final_portfolio : total portfolio value at end of episode
        trade_log       : list of trade dicts (optional)
        initial_cash    : starting capital

    Returns:
        float in [0.0, 1.0]
    """
    profit        = final_portfolio - initial_cash
    profit_score  = min(1.0, max(0.0, profit / PROFIT_TARGET))

    efficiency    = compute_trade_efficiency(trade_log or [])

    score = 0.5 * profit_score + 0.5 * efficiency
    return round(score, 4)


# ------------------------------------------------------------------ #
#  Task metadata dict (consumed by routes.py /tasks endpoint)         #
# ------------------------------------------------------------------ #
TASK_INFO = {
    "id":            TASK_ID,
    "name":          TASK_NAME,
    "description":   TASK_DESCRIPTION,
    "initial_cash":  INITIAL_CASH,
    "profit_target": PROFIT_TARGET,
    "ticker":        "MSFT",
    "difficulty":    "medium",
    "indicators":    ["ma5", "ma10"],
    "grader":        "score = 0.5 * profit_score + 0.5 * trade_efficiency",
    "action_schema": {
        "action": "BUY | SELL | HOLD",
        "quantity": "float (units to trade, 0.0 for HOLD)"
    }
}


if __name__ == "__main__":
    prices = load_data()
    print(f"Medium task loaded: {len(prices)} price points")
    print(f"Price range: ${min(prices):.2f} — ${max(prices):.2f}")

    # Simulate a dummy trade log for grader test
    dummy_trades = [
        {"action": "BUY",  "price": prices[10], "quantity": 10},
        {"action": "SELL", "price": prices[30], "quantity": 10},
        {"action": "BUY",  "price": prices[50], "quantity": 10},
        {"action": "SELL", "price": prices[40], "quantity": 10},  # losing trade
    ]
    shares      = INITIAL_CASH / prices[0]
    final_value = shares * prices[-1]
    print(f"Buy-and-hold score: {grade(final_value, dummy_trades):.4f}")
    print(f"Trade efficiency  : {compute_trade_efficiency(dummy_trades):.4f}")