TRADE_COST_RATE = 0.001

ALPHA = 1.5
BETA  = 0.5
GAMMA = 0.1
DELTA = 0.2
HOLD_PENALTY = 0.0005


def compute_reward(
    prev_portfolio: float,
    current_portfolio: float,
    peak_portfolio: float,
    action: str,
    trade_value: float
) -> dict:

    profit_pct = (current_portfolio - prev_portfolio) / (prev_portfolio + 1e-8)

    drawdown = (peak_portfolio - current_portfolio) / (peak_portfolio + 1e-8)

    risk_penalty = 0.0
    if drawdown > 0:
        risk_penalty = (drawdown ** 1.1)

    trade_cost = 0.0
    if action in ("BUY", "SELL"):
        trade_cost = TRADE_COST_RATE * (trade_value / (prev_portfolio + 1e-8))

    consistency_bonus = profit_pct if profit_pct > 0 else 0.0

    overtrade_penalty = 0.0
    if action in ("BUY", "SELL") and abs(profit_pct) < 0.001:
        overtrade_penalty = 0.001

    reward = (
        ALPHA * profit_pct
        + DELTA * consistency_bonus
        - BETA * risk_penalty
        - GAMMA * trade_cost
        - overtrade_penalty
    )

    if action == "HOLD":
        reward -= HOLD_PENALTY

    reward = max(min(reward, 1.0), -1.0)

    return {
        "reward": reward,
        "profit_pct": profit_pct,
        "risk_penalty": risk_penalty,
        "trade_cost": trade_cost,
        "consistency_bonus": consistency_bonus,
        "overtrade_penalty": overtrade_penalty,
        "drawdown": drawdown,
    }