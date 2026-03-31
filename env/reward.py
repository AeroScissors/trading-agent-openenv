"""
reward.py — Reward computation for Trading-Agent-OpenEnv

Design philosophy:
  - Reward should be positive when the agent is doing well (holding a winning position)
  - Reward should be zero or slightly negative for neutral actions (idle HOLDs)
  - Reward should be clearly negative only for genuinely bad behavior:
      * large drawdowns
      * overtrading (buy/sell with negligible price movement)
      * transaction costs on wasteful trades

Clipped to [-1.0, 1.0] to prevent gradient explosions in RL training.
"""

TRADE_COST_RATE = 0.001   # 0.1% per trade (realistic slippage simulation)

# Reward weights
ALPHA = 1.5    # profit change weight
DELTA = 0.3    # consistency bonus (reward positive momentum)
BETA  = 0.5    # drawdown penalty weight
GAMMA = 0.1    # trade cost weight

# Penalties — kept small so they don't dominate
HOLD_PENALTY    = 0.0001   # tiny friction for idle holds (was 0.0005 — too harsh)
OVERTRADE_PENALTY = 0.001  # penalty for trading with < 0.1% price movement


def compute_reward(
    prev_portfolio: float,
    current_portfolio: float,
    peak_portfolio: float,
    action: str,
    trade_value: float,
) -> dict:
    """
    Compute step reward from portfolio state and action taken.

    Args:
        prev_portfolio    : portfolio value at previous step
        current_portfolio : portfolio value at current step
        peak_portfolio    : highest portfolio value seen so far this episode
        action            : one of "BUY", "SELL", "HOLD"
        trade_value       : notional value of the trade (0 for HOLD)

    Returns:
        dict with 'reward' and breakdown components
    """

    # 1. Profit change as a fraction of previous portfolio
    profit_pct = (current_portfolio - prev_portfolio) / (prev_portfolio + 1e-8)

    # 2. Drawdown from peak — only penalise when it actually matters (>2%)
    drawdown = (peak_portfolio - current_portfolio) / (peak_portfolio + 1e-8)
    drawdown = max(0.0, drawdown)

    risk_penalty = 0.0
    if drawdown > 0.02:                       # ignore micro-drawdowns < 2%
        risk_penalty = (drawdown ** 1.1)

    # 3. Transaction cost — only on actual trades, scaled to trade size
    trade_cost = 0.0
    if action in ("BUY", "SELL"):
        trade_cost = TRADE_COST_RATE * (trade_value / (prev_portfolio + 1e-8))

    # 4. Consistency bonus — reward staying in a winning position
    #    Positive only when portfolio is genuinely growing
    consistency_bonus = profit_pct if profit_pct > 0.0 else 0.0

    # 5. Overtrade penalty — penalise churn with negligible price movement
    overtrade_penalty = 0.0
    if action in ("BUY", "SELL") and abs(profit_pct) < 0.001:
        overtrade_penalty = OVERTRADE_PENALTY

    # 6. Hold penalty — very small, just enough to discourage pure inaction
    hold_penalty = HOLD_PENALTY if action == "HOLD" else 0.0

    # Combine
    reward = (
        ALPHA * profit_pct
        + DELTA * consistency_bonus
        - BETA  * risk_penalty
        - GAMMA * trade_cost
        - overtrade_penalty
        - hold_penalty
    )

    # Clip to [-1, 1]
    reward = max(min(reward, 1.0), -1.0)

    return {
        "reward":             round(reward, 6),
        "profit_pct":         round(profit_pct, 6),
        "risk_penalty":       round(risk_penalty, 6),
        "trade_cost":         round(trade_cost, 6),
        "consistency_bonus":  round(consistency_bonus, 6),
        "overtrade_penalty":  round(overtrade_penalty, 6),
        "drawdown":           round(drawdown, 6),
    }