---
title: Trading Agent OpenEnv
sdk: docker
app_port: 7860
---

#  Trading Agent OpenEnv

An **OpenEnv-compatible reinforcement learning environment** for trading agents.
Three tasks of increasing difficulty using real market data (AAPL, MSFT, BTC-USD)
with automatic synthetic fallback when offline.

---

##  What It Does

Agents interact with a simulated trading environment via a REST API.
At each step, the agent receives a **state** (prices, indicators, portfolio),
sends an **action** (BUY / SELL / HOLD), and receives a **reward**.
At the end of the episode, a **grader** returns a score between 0.0 and 1.0.

---

##  State Space

| Field           | Type         | Description                          |
|----------------|--------------|--------------------------------------|
| `price_history` | float[20]    | Last 20 closing prices               |
| `current_price` | float        | Current asset price                  |
| `position`      | float        | Units currently held                 |
| `cash`          | float        | Available cash in USD                |
| `ma5`           | float        | 5-period moving average              |
| `ma10`          | float        | 10-period moving average             |
| `sharpe`        | float        | Rolling annualised Sharpe ratio      |
| `step`          | int          | Current timestep index               |

---

##  Action Space

| Field      | Type   | Values              |
|-----------|--------|---------------------|
| `action`   | string | BUY, SELL, HOLD     |
| `quantity` | float  | Units to trade ≥ 0  |

---

##  Tasks

### Task 1 — Easy: Follow the Trend
- **Ticker:** AAPL (1 year, uptrending)
- **Goal:** Buy early, hold, sell near peak
- **Grader:** `score = min(1.0, profit / $500)`
- **Starting cash:** $10,000

### Task 2 — Medium: React to Signals
- **Ticker:** MSFT (1 year, sideways + moderate trend)
- **Goal:** Use MA5/MA10 crossover signals to time trades
- **Grader:** `score = 0.5 × profit_score + 0.5 × trade_efficiency`
- **Starting cash:** $10,000

### Task 3 — Hard: Maximize Profit with Risk
- **Ticker:** BTC-USD (1 year, highly volatile)
- **Goal:** Maximize Sharpe ratio under transaction costs + drawdown penalties
- **Grader:** `score = 0.5 × sharpe_score + 0.5 × profit_score`
- **Starting cash:** $10,000

---

##  Reward Function
```
reward = profit_change - risk_penalty - trade_cost
```

| Component       | Description                                      |
|----------------|--------------------------------------------------|
| `profit_change` | Change in portfolio value this step              |
| `risk_penalty`  | Penalty when drawdown exceeds 5% of peak value   |
| `trade_cost`    | 0.1% fee on every BUY or SELL                    |

---

##  API Endpoints

| Method | Endpoint    | Description                              |
|--------|------------|------------------------------------------|
| GET    | `/`         | Health check + available endpoints       |
| GET    | `/tasks`    | List all tasks with metadata             |
| POST   | `/reset`    | Reset environment, returns initial state |
| GET    | `/state`    | Current state snapshot                   |
| POST   | `/step`     | Execute action, returns reward/done/info |
| POST   | `/grader`   | Return final score (0.0 – 1.0)           |
| POST   | `/baseline` | Run MA crossover baseline agent          |
| GET    | `/docs`     | Interactive Swagger UI                   |

### Example Usage
```python
import requests

BASE = "http://localhost:7860"

# Reset environment
state = requests.post(f"{BASE}/reset", json={"task": "easy"}).json()

# Step loop
done = False
while not done:
    result = requests.post(f"{BASE}/step", json={
        "task": "easy",
        "action": "BUY",
        "quantity": 1.0
    }).json()
    done = result["done"]

# Get final score
score = requests.post(f"{BASE}/grader", json={"task": "easy"}).json()
print(f"Final score: {score['score']}")
```

---

##  Setup

### Local
```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline agent (no API key needed)
python baseline.py

# Start API server
uvicorn main:app --reload --port 7860
```

### Docker
```bash
# Build image
docker build -t trading-env .

# Run container
docker run -p 7860:7860 trading-env
```

### HuggingFace Spaces

1. Create a new Space → select **Docker** SDK
2. Upload all project files
3. Space auto-builds and exposes the API at your Space URL
4. Visit `https://your-space.hf.space/docs` for Swagger UI

---

##  Baseline Scores

Scores from the rule-based MA5/MA10 crossover agent:

| Task   | Strategy         | Typical Score |
|--------|-----------------|---------------|
| Easy   | Buy MA5 > MA10  | ~0.55 – 0.70  |
| Medium | MA crossover    | ~0.40 – 0.55  |
| Hard   | MA crossover    | ~0.25 – 0.40  |

Run `python baseline.py` to reproduce these scores locally.

---

##  Project Structure
```
Trading-Agent-OpenEnv/
├── env/
│   ├── core_env.py          # step() / reset() / state()
│   ├── models.py            # Pydantic models
│   ├── reward.py            # Reward calculation
│   ├── tasks/
│   │   ├── easy.py          # AAPL trend task
│   │   ├── medium.py        # MSFT MA signal task
│   │   └── hard.py          # BTC-USD Sharpe task
│   └── graders/
│       └── grader.py        # Master grader (0.0 – 1.0)
├── api/
│   └── routes.py            # FastAPI endpoints
├── data/
│   └── fetch_data.py        # yfinance + synthetic fallback
├── baseline.py              # Rule-based MA crossover agent
├── main.py                  # FastAPI app entry point
├── openenv.yaml             # OpenEnv spec file
├── Dockerfile               # Docker build config
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

##  License

MIT License — free to use, modify, and distribute.