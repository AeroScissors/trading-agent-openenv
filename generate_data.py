"""
generate_data.py
================
Run this ONCE from your project root to generate multi-asset portfolio CSVs.

Downloads real historical data via yfinance (2020-2024).

Usage:
    python generate_data.py

Output:
    data/easy.csv    → 3 assets:  AAPL, MSFT, GOOGL
    data/medium.csv  → 5 assets:  AAPL, MSFT, GOOGL, JNJ, JPM
    data/hard.csv    → 7 assets:  AAPL, MSFT, GOOGL, JNJ, JPM, BTC-USD, ETH-USD
"""

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TASKS = {
    "easy": ["AAPL", "MSFT", "GOOGL"],
    "medium": ["AAPL", "MSFT", "GOOGL", "JNJ", "JPM"],
    "hard": ["AAPL", "MSFT", "GOOGL", "JNJ", "JPM", "BTC-USD", "ETH-USD"]
}

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# ------------------------------------------------------------------ #
# Download & merge
# ------------------------------------------------------------------ #

def generate_task_data(task_name: str, symbols: list[str]):
    """
    Download multi-asset data and save as single CSV with columns per asset.
    
    Args:
        task_name: "easy", "medium", or "hard"
        symbols: list of ticker symbols
    """
    print(f"\n[{task_name.upper()}] Fetching {len(symbols)} assets...")
    
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        return
    
    combined = pd.DataFrame()
    
    for symbol in symbols:
        print(f"  - Downloading {symbol}...", end=" ")
        
        try:
            df = yf.download(
                symbol,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty or len(df) < 50:
                print(f"FAILED (insufficient data)")
                return
            
            # yfinance returns multi-index columns for single ticker
            # Flatten the columns first
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Extract Close prices
            close_series = df['Close']
            
            # Merge into combined DataFrame
            if combined.empty:
                combined = pd.DataFrame({symbol: close_series})
            else:
                combined[symbol] = close_series
            
            print(f"OK ({len(df)} rows)")
        
        except Exception as e:
            print(f"FAILED ({e})")
            return
    
    # Forward-fill missing values, then drop remaining NaN rows
    combined = combined.ffill().dropna()
    
    # Save to CSV
    csv_path = DATA_DIR / f"{task_name}.csv"
    combined.to_csv(csv_path)
    
    # Summary
    start_val = combined.iloc[0].mean()
    end_val = combined.iloc[-1].mean()
    pct_change = ((end_val / start_val) - 1) * 100
    
    print(f"\n  ✓ Saved: {csv_path}")
    print(f"    Rows: {len(combined)}")
    print(f"    Columns: {list(combined.columns)}")
    print(f"    Date range: {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"    Avg return: {pct_change:+.1f}%")

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def generate_all():
    print("=" * 70)
    print("GENERATING MULTI-ASSET PORTFOLIO DATA")
    print("=" * 70)
    
    for task_name, symbols in TASKS.items():
        generate_task_data(task_name, symbols)
    
    print("\n" + "=" * 70)
    print("DONE! CSVs ready in data/")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Verify CSVs: ls -lh data/")
    print("  2. Test locally: uvicorn server.app:app --reload --port 7860")
    print("  3. Push to HF: git push hf stable-v1:main --force")


if __name__ == "__main__":
    generate_all()