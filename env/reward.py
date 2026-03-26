TRADE_COST_RATE = 0.001       # 0.1% fee per trade (slippage simulation)
DRAWDOWN_THRESHOLD = 0.05    # 5% drop triggers risk penalty
DRAWDOWN_PENALTY_SCALE = 2.0 # multiplier for drawdown penalty


def compute_reward(
    prev_portfolio: float,
    current_portfolio: float,
    peak_portfolio: float,
    action: str,
    trade_value: float
) -> dict:
    """
    Compute the reward for a single step.

    Args:
        prev_portfolio   : portfolio value at previous step
        current_portfolio: portfolio value after action
        peak_portfolio   : highest portfolio value seen so far
        action           : "BUY", "SELL", or "HOLD"
        trade_value      : dollar value of shares traded this step

    Returns:
        dict with reward (float) and breakdown components
    """

    # 1. Profit change
    profit_change = current_portfolio - prev_portfolio

    # 2. Drawdown penalty
    drawdown = (peak_portfolio - current_portfolio) / (peak_portfolio + 1e-8)
    if drawdown > DRAWDOWN_THRESHOLD:
        risk_penalty = DRAWDOWN_PENALTY_SCALE * (drawdown - DRAWDOWN_THRESHOLD) * prev_portfolio
    else:
        risk_penalty = 0.0

    # 3. Trade cost (only on BUY or SELL)
    if action in ("BUY", "SELL"):
        trade_cost = TRADE_COST_RATE * trade_value
    else:
        trade_cost = 0.0

    # 4. Final reward
    reward = profit_change - risk_penalty - trade_cost

    return {
        "reward": reward,
        "profit_change": profit_change,
        "risk_penalty": risk_penalty,
        "trade_cost": trade_cost,
        "drawdown": drawdown,
    }