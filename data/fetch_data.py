import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ------------------------------------------------------------------ #
#  Ticker config per task                                              #
# ------------------------------------------------------------------ #
TASK_TICKERS = {
    "easy":   "AAPL",
    "medium": "MSFT",
    "hard":   "BTC-USD",
}


def fetch_prices(task: str, period_days: int = 365) -> list[float]:
    """
    Fetch closing prices for the given task ticker.
    Falls back to synthetic data if yfinance is unavailable or fails.

    Args:
        task        : "easy", "medium", or "hard"
        period_days : how many calendar days of history to pull

    Returns:
        List of closing prices (floats), oldest → newest
    """
    ticker = TASK_TICKERS.get(task, "AAPL")

    if YFINANCE_AVAILABLE:
        try:
            end   = datetime.today()
            start = end - timedelta(days=period_days)
            df    = yf.download(ticker, start=start, end=end, progress=False)

            if df.empty or "Close" not in df.columns:
                raise ValueError("Empty or malformed data returned")

            prices = df["Close"].dropna().tolist()

            if len(prices) < 50:
                raise ValueError(f"Too few data points: {len(prices)}")

            print(f"[fetch_data] ✅ Real data fetched for {ticker}: {len(prices)} days")
            return prices

        except Exception as e:
            print(f"[fetch_data] ⚠️  yfinance failed ({e}), using synthetic data")

    # ---------------------------------------------------------------- #
    #  Synthetic fallback                                                #
    # ---------------------------------------------------------------- #
    return _synthetic_prices(task, n=252)


def _synthetic_prices(task: str, n: int = 252) -> list[float]:
    """
    Generate realistic synthetic price series for each task difficulty.

    easy   → clean uptrend  (low volatility, positive drift)
    medium → sideways trend (moderate volatility, slight drift)
    hard   → volatile asset (high volatility, random drift)
    """
    np.random.seed(42)

    if task == "easy":
        drift      = 0.0008    # ~20 % annual
        volatility = 0.010
        start      = 150.0

    elif task == "medium":
        drift      = 0.0003
        volatility = 0.015
        start      = 300.0

    else:  # hard
        drift      = 0.0005
        volatility = 0.035
        start      = 30_000.0

    log_returns = np.random.normal(drift, volatility, n)
    prices      = start * np.exp(np.cumsum(log_returns))

    print(f"[fetch_data] 🔧 Synthetic data generated for task='{task}': {n} days")
    return prices.tolist()


# ------------------------------------------------------------------ #
#  Quick test                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    for t in ("easy", "medium", "hard"):
        p = fetch_prices(t)
        print(f"  {t:6s} → {len(p)} prices | first={p[0]:.2f}  last={p[-1]:.2f}")