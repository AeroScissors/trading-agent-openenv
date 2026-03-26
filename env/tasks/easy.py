# easy.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.fetch_data import fetch_prices

# ------------------------------------------------------------------ #
#  Task 1 — Follow the Trend (Easy)                                    #
# ------------------------------------------------------------------ #

TASK_ID          = "easy"
TASK_NAME        = "Follow the Trend"
TASK_DESCRIPTION = (
    "A simple uptrending stock (AAPL, 1 year). "
    "Buy early, hold through the trend, sell near the peak. "
    "No indicators needed — just read price momentum."
)

INITIAL_CASH     = 10_000.0   # starting capital in USD
MAX_STEPS        = None        # use full dataset length
PROFIT_TARGET    = 500.0       # $500 profit → score of 1.0


def load_data() -> list[float]:
    """Return closing price series for the easy task."""
    return fetch_prices("easy", period_days=365)


def grade(final_portfolio: float, initial_cash: float = INITIAL_CASH) -> float:
    """
    Score = min(1.0, profit / PROFIT_TARGET)

    Args:
        final_portfolio : total portfolio value at end of episode
        initial_cash    : starting capital (default 10,000)

    Returns:
        float in [0.0, 1.0]
    """
    profit = final_portfolio - initial_cash
    score  = min(1.0, max(0.0, profit / PROFIT_TARGET))
    return round(score, 4)


# ------------------------------------------------------------------ #
#  Task metadata dict (consumed by routes.py /tasks endpoint)         #
# ------------------------------------------------------------------ #
TASK_INFO = {
    "id":           TASK_ID,
    "name":         TASK_NAME,
    "description":  TASK_DESCRIPTION,
    "initial_cash": INITIAL_CASH,
    "profit_target": PROFIT_TARGET,
    "ticker":       "AAPL",
    "difficulty":   "easy",
    "grader":       "score = min(1.0, profit / 500)",
    "action_schema": {
        "action": "BUY | SELL | HOLD",
        "quantity": "float (units to trade, 0.0 for HOLD)"
    }
}


if __name__ == "__main__":
    prices = load_data()
    print(f"Easy task loaded: {len(prices)} price points")
    print(f"Price range: ${min(prices):.2f} — ${max(prices):.2f}")

    # Simulate a perfect buy-and-hold for reference
    shares         = INITIAL_CASH / prices[0]
    final_value    = shares * prices[-1]
    print(f"Buy-and-hold score: {grade(final_value):.4f}")