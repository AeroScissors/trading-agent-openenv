"""
generate_data.py
================
Run this ONCE from your project root to regenerate the price CSVs.

Priority:
  1. Downloads real 2023 historical data via yfinance (best)
  2. Falls back to calibrated synthetic data if download fails

Usage:
    python generate_data.py

Output:
    data/prices/aapl.csv   (AAPL 2023 — clean uptrend)
    data/prices/msft.csv   (MSFT 2023 — clean uptrend)
    data/prices/btc.csv    (BTC  2023 — volatile uptrend)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "prices"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# 1. Try real yfinance download (2023 — confirmed uptrends)
# ------------------------------------------------------------------ #

DOWNLOAD_CONFIGS = {
    "aapl": {"ticker": "AAPL",    "start": "2023-01-01", "end": "2023-12-31"},
    "msft": {"ticker": "MSFT",    "start": "2023-01-01", "end": "2023-12-31"},
    "btc":  {"ticker": "BTC-USD", "start": "2023-01-01", "end": "2023-12-31"},
}

def try_download_real(name: str, cfg: dict) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.download(
            cfg["ticker"],
            start=cfg["start"],
            end=cfg["end"],
            auto_adjust=True,
            progress=False,
        )
        if df.empty or len(df) < 50:
            return None
        out = pd.DataFrame({"close": df["Close"].values.flatten()})
        return out
    except Exception as e:
        print(f"  [yfinance] Download failed for {name}: {e}")
        return None


# ------------------------------------------------------------------ #
# 2. Calibrated synthetic fallback
#    Parameters tuned to match 2023 real-world performance
# ------------------------------------------------------------------ #

SYNTHETIC_CONFIGS = {
    # AAPL 2023: ~$130 → $195  (+50%)
    "aapl": {
        "start": 130.0,
        "drift": 0.0016,      # strong daily upward drift
        "volatility": 0.012,
        "n": 252,
        "seed": 42,
    },
    # MSFT 2023: ~$220 → $375  (+70%)
    "msft": {
        "start": 220.0,
        "drift": 0.0022,      # even stronger — MSFT had a great 2023
        "volatility": 0.014,
        "n": 252,
        "seed": 43,
    },
    # BTC 2023: ~$16k → $42k  (+160%), high volatility
    "btc": {
        "start": 16_500.0,
        "drift": 0.0045,      # massive drift for BTC bull run
        "volatility": 0.038,
        "n": 365,
        "seed": 44,
    },
}

def make_synthetic(name: str) -> pd.DataFrame:
    cfg = SYNTHETIC_CONFIGS[name]
    np.random.seed(cfg["seed"])
    log_returns = np.random.normal(cfg["drift"], cfg["volatility"], cfg["n"])
    prices = cfg["start"] * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({"close": prices})


# ------------------------------------------------------------------ #
# 3. Generate & save
# ------------------------------------------------------------------ #

def generate_all():
    print("Generating price CSVs...\n")

    for name, dl_cfg in DOWNLOAD_CONFIGS.items():
        csv_path = DATA_DIR / f"{name}.csv"
        print(f"  [{name.upper()}]")

        df = try_download_real(name, dl_cfg)
        source = "real yfinance"

        if df is None:
            df = make_synthetic(name)
            source = "calibrated synthetic"

        df.to_csv(csv_path, index=False)
        pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        print(f"    Source : {source}")
        print(f"    Days   : {len(df)}")
        print(f"    Range  : ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}  ({pct:+.1f}%)")
        print(f"    Saved  : {csv_path}\n")

    print("Done! Now restart your FastAPI server and re-run baseline_llm.py")


if __name__ == "__main__":
    generate_all()