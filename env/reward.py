# File: env/reward.py

import numpy as np


def compute_sharpe(portfolio_history):
    """
    Compute annualized Sharpe ratio from portfolio value history.

    Args:
        portfolio_history: list[float]

    Returns:
        float (clipped to [-5.0, 5.0] range)
    """
    if len(portfolio_history) < 3:
        return 0.0

    values = np.array(portfolio_history, dtype=float)
    returns = np.diff(values) / (values[:-1] + 1e-8)

    if np.std(returns) < 1e-8:
        return 0.0

    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    
    # Clip to reasonable range to handle crypto volatility
    sharpe = np.clip(sharpe, -5.0, 5.0)
    
    return float(sharpe)