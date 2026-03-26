
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

TICKERS = {
    "aapl": "AAPL",
    "msft": "MSFT",
    "btc":  "BTC-USD",
}

out_dir = Path("data/prices")
out_dir.mkdir(parents=True, exist_ok=True)

end   = datetime.today()
start = end - timedelta(days=365)

for name, ticker in TICKERS.items():
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    prices = df["Close"].dropna().reset_index()
    prices.columns = ["date", "close"]
    path = out_dir / f"{name}.csv"
    prices.to_csv(path, index=False)
    print(f"  Saved {len(prices)} rows → {path}")

print("\nDone! Commit the data/prices/ folder to your repo.")