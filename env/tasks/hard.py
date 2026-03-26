# hard.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.fetch_data import fetch_prices
import numpy as np

# ------------------------------------------------------------------ #
#  Task 3 — Maximize Profit with Risk (Hard)                           #
# ------------------------------------------------------------------ #

TASK_ID          = "hard"
TASK_NAME        = "Maximize Profit with Risk"
TASK_DESCRIPTION = (
    "A highly volatile asset (BTC-USD, 1 year). "
    "Transaction costs and drawdown penalties are active. "
    "Goal is to maximize the Sharpe ratio, not just raw profit. "
    "Risk management is critical — reckless trading will be penalized."
)

INITIAL_CASH     = 10_000.0
MAX_STEPS        = None
PROFIT_TARGET    = 2000.0      # higher bar due to BTC volatility
SHARPE_TARGET    = 1.5         # annualised Sharpe considered excellent
RISK_FREE_RATE   = 0.05        # 5% annual risk-free rate


def load_data() -> list[float]:
    """Return closing price series for the hard task."""
    return fetch_prices("hard", period_days=365)


def compute_sharpe(portfolio_values: list[float], risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    Compute annualised Sharpe ratio from a series of portfolio values.

    Args:
        portfolio_values : list of portfolio values over time
        risk_free_rate   : annual risk-free rate (default 5%)

    Returns:
        Sharpe ratio as a float (can be negative)
    """
    if len(portfolio_values) < 2:
        return 0.0

    values  = np.array(portfolio_values, dtype=float)
    returns = np.diff(values) / (values[:-1] + 1e-8)

    daily_rf  = risk_free_rate / 252
    excess    = returns - daily_rf
    mean_exc  = np.mean(excess)
    std_exc   = np.std(excess) + 1e-8

    sharpe = (mean_exc / std_exc) * np.sqrt(252)   # annualise
    return round(float(sharpe), 4)


def grade(
    final_portfolio: float,
    portfolio_history: list[float] = None,
    initial_cash: float = INITIAL_CASH
) -> float:
    """
    Score = 0.5 * sharpe_score + 0.5 * profit_score

    Args:
        final_portfolio  : total portfolio value at end of episode
        portfolio_history: list of portfolio values at each step
        initial_cash     : starting capital

    Returns:
        float in [0.0, 1.0]
    """
    # Profit score
    profit       = final_portfolio - initial_cash
    profit_score = min(1.0, max(0.0, profit / PROFIT_TARGET))

    # Sharpe score
    if portfolio_history and len(portfolio_history) >= 2:
        sharpe       = compute_sharpe(portfolio_history)
        sharpe_score = min(1.0, max(0.0, sharpe / SHARPE_TARGET))
    else:
        sharpe_score = 0.0

    score = 0.5 * sharpe_score + 0.5 * profit_score
    return round(score, 4)


# ------------------------------------------------------------------ #
#  Task metadata dict (consumed by routes.py /tasks endpoint)         #
# ------------------------------------------------------------------ #
TASK_INFO = {
    "id":              TASK_ID,
    "name":            TASK_NAME,
    "description":     TASK_DESCRIPTION,
    "initial_cash":    INITIAL_CASH,
    "profit_target":   PROFIT_TARGET,
    "sharpe_target":   SHARPE_TARGET,
    "ticker":          "BTC-USD",
    "difficulty":      "hard",
    "penalties":       ["transaction_costs", "drawdown_penalty"],
    "grader":          "score = 0.5 * sharpe_score + 0.5 * profit_score",
    "action_schema": {
        "action": "BUY | SELL | HOLD",
        "quantity": "float (units to trade, 0.0 for HOLD)"
    }
}


if __name__ == "__main__":
    prices = load_data()
    print(f"Hard task loaded: {len(prices)} price points")
    print(f"Price range: ${min(prices):,.2f} — ${max(prices):,.2f}")

    # Simulate buy-and-hold portfolio history
    shares  = INITIAL_CASH / prices[0]
    history = [shares * p for p in prices]

    final_value = history[-1]
    sharpe      = compute_sharpe(history)
    score       = grade(final_value, history)

    print(f"Buy-and-hold final value : ${final_value:,.2f}")
    print(f"Buy-and-hold Sharpe ratio: {sharpe:.4f}")
    print(f"Buy-and-hold score       : {score:.4f}")