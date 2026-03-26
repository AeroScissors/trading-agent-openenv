import numpy as np
from pathlib import Path

TASK_TICKERS = {
    "easy":   "aapl",
    "medium": "msft",
    "hard":   "btc",
}

# CSV files live at data/prices/ relative to project root
DATA_DIR = Path(__file__).parent / "prices"


def fetch_prices(task: str, period_days: int = 365) -> list[float]:
    """
    Load closing prices for the given task.

    Priority:
      1. Bundled CSV from data/prices/{ticker}.csv  (real historical data)
      2. Synthetic fallback                          (if CSV missing)

    Args:
        task        : "easy", "medium", or "hard"
        period_days : ignored when loading from CSV (full dataset used)

    Returns:
        List of closing prices (floats), oldest → newest
    """
    name = TASK_TICKERS.get(task, "aapl")
    csv_path = DATA_DIR / f"{name}.csv"

    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            prices = df["close"].dropna().tolist()
            if len(prices) >= 50:
                print(f"[fetch_data] ✅ Loaded real data for task='{task}': {len(prices)} days")
                return prices
        except Exception as e:
            print(f"[fetch_data] ⚠️  CSV load failed ({e}), using synthetic data")

    return _synthetic_prices(task)


def _synthetic_prices(task: str, n: int = 252) -> list[float]:
    """
    Realistic synthetic price series — only used if CSV is missing.
    Fixed seed ensures reproducibility.
    """
    np.random.seed(42)

    if task == "easy":
        drift, volatility, start = 0.0008, 0.010, 150.0
    elif task == "medium":
        drift, volatility, start = 0.0003, 0.015, 300.0
    else:
        drift, volatility, start = 0.0005, 0.035, 30_000.0

    log_returns = np.random.normal(drift, volatility, n)
    prices      = start * np.exp(np.cumsum(log_returns))

    print(f"[fetch_data] 🔧 Synthetic data for task='{task}': {n} days")
    return prices.tolist()


if __name__ == "__main__":
    for t in ("easy", "medium", "hard"):
        p = fetch_prices(t)
        print(f"  {t:6s} → {len(p)} prices | first={p[0]:.2f}  last={p[-1]:.2f}")