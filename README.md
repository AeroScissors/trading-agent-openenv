---
title: Trading Agent OpenEnv
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
sdk_version: "3.11"
python_version: "3.11"
app_file: main.py
pinned: false
tags:
  - openenv
---

# Trading Agent OpenEnv

A complete [OpenEnv](https://openenv.ai)-compatible reinforcement learning environment for training and evaluating trading agents on real market data. Agents learn to buy, sell, and hold assets across three tasks of increasing difficulty — from simple trend-following on AAPL to risk-adjusted trading on volatile BTC-USD.

---

## Why this environment?

Most RL environments use games or toy problems. Real-world trading is one of the clearest examples of sequential decision-making under uncertainty: the agent observes market state, takes actions with direct financial consequences, and receives a reward signal shaped by profit, risk, and transaction costs. This environment models that faithfully — with real price data, realistic trade costs, and graders that reflect what a human trader would actually care about.

---

## Environment description

The environment simulates a single-asset trading account. At each timestep the agent receives a market observation and chooses to BUY, SELL, or HOLD a quantity of the asset. The portfolio value changes accordingly, and the agent receives a reward signal composed of profit change, a drawdown penalty, and transaction costs.

Price data is fetched from Yahoo Finance (AAPL, MSFT, BTC-USD). If the network is unavailable, a realistic synthetic series is used as fallback so the environment always starts cleanly.

---

## Observation space

Each observation is a `State` object with the following fields:

| Field | Type | Description |
|---|---|---|
| `price_history` | `list[float]` (length 20) | Last 20 closing prices, oldest first |
| `current_price` | `float` | Current asset price |
| `position` | `float` | Units of the asset currently held |
| `cash` | `float` | Available cash in USD |
| `ma5` | `float` | 5-period moving average of closing price |
| `ma10` | `float` | 10-period moving average of closing price |
| `sharpe` | `float` | Rolling annualised Sharpe ratio |
| `step` | `int` | Current timestep index |

---

## Action space

Each action is an `Action` object:

| Field | Type | Values | Description |
|---|---|---|---|
| `action` | `string` | `BUY`, `SELL`, `HOLD` | Trading direction |
| `quantity` | `float` | `>= 0.0` | Units to trade (ignored for HOLD) |

BUY is capped at what the agent can afford. SELL is capped at the agent's current position. Invalid quantities are silently clamped rather than raising errors, so agents don't need to track portfolio state perfectly.

---

## Reward function

```
reward = profit_change - risk_penalty - trade_cost
```

| Component | Description |
|---|---|
| `profit_change` | Change in total portfolio value this step (cash + position × price) |
| `risk_penalty` | Activates when drawdown exceeds 5% — scales with severity |
| `trade_cost` | 0.1% fee on every BUY or SELL (slippage simulation) |

The reward is dense — every timestep provides signal. Agents that over-trade are penalised by accumulated costs; agents that let drawdowns grow are penalised by the risk term.

---

## Tasks

### Task 1 — Follow the Trend (Easy)

- **Ticker:** AAPL (1 year of daily closes)
- **Objective:** Buy early in an uptrend, hold through it, sell near the peak
- **Initial cash:** $10,000
- **Profit target:** $500
- **Grader:** `score = min(1.0, profit / 500)`
- **Difficulty:** No indicators needed — raw price momentum is sufficient
- **Baseline score:** ~0.10

### Task 2 — React to Signals (Medium)

- **Ticker:** MSFT (1 year of daily closes)
- **Objective:** Use MA5/MA10 crossover signals to time entries and exits
- **Initial cash:** $10,000
- **Profit target:** $800
- **Grader:** `score = 0.5 × profit_score + 0.5 × trade_efficiency`
- **Difficulty:** Agent must learn to act on indicator crossovers, not just price
- **Baseline score:** ~0.09

### Task 3 — Maximize Profit with Risk (Hard)

- **Ticker:** BTC-USD (1 year of daily closes)
- **Objective:** Maximize risk-adjusted returns on a highly volatile asset
- **Initial cash:** $10,000
- **Profit target:** $2,000 · Sharpe target: 1.5
- **Grader:** `score = 0.5 × sharpe_score + 0.5 × profit_score`
- **Penalties:** Transaction costs + drawdown penalty both active
- **Difficulty:** Raw profit-seeking is punished — risk management is required
- **Baseline score:** ~0.08

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/tasks` | List all tasks with metadata and action schema |
| `POST` | `/reset` | Reset environment, returns initial state |
| `GET` | `/state` | Get current state without advancing the timestep |
| `POST` | `/step` | Execute one action, returns observation/reward/done/info |
| `POST` | `/grader` | Return final score (0.0–1.0) for current episode |
| `POST` | `/baseline` | Run baseline agent, return scores for one or all tasks |

All endpoints accept and return JSON. The `/docs` path (FastAPI auto-docs) provides an interactive schema explorer.

### Example: full episode

```python
import requests

BASE = "http://localhost:7860"

# Reset
state = requests.post(f"{BASE}/reset", json={"task": "easy"}).json()

done = False
while not done:
    result = requests.post(f"{BASE}/step", json={
        "task": "easy",
        "action": "BUY",
        "quantity": 1.0
    }).json()
    done = result["done"]

score = requests.post(f"{BASE}/grader", json={"task": "easy"}).json()
print(f"Score: {score['score']}  Profit: ${score['profit']}")
```

---

## Setup and usage

### Run locally with Docker

```bash
docker build -t trading-openenv .
docker run -p 7860:7860 trading-openenv
```

The API will be available at `http://localhost:7860`. Visit `/docs` for the interactive Swagger UI.

### Run locally without Docker

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Run the baseline agent

```bash
# Rule-based MA crossover baseline (no API key needed)
python baseline.py

# LLM-powered baseline (requires OpenAI API key)
export OPENAI_API_KEY=your_key_here
python baseline_llm.py
```

---

## Project structure

```
.
├── main.py                  # FastAPI app entry point
├── baseline.py              # Rule-based MA crossover baseline
├── baseline_llm.py          # LLM baseline agent (OpenAI API)
├── openenv.yaml             # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
├── api/
│   └── routes.py            # All HTTP endpoints
└── env/
    ├── core_env.py          # TradingEnv class (reset/step/state)
    ├── models.py            # Pydantic models (State, Action, StepResult)
    ├── reward.py            # Reward computation
    ├── tasks/
    │   ├── easy.py          # AAPL task + grader
    │   ├── medium.py        # MSFT task + grader
    │   └── hard.py          # BTC-USD task + grader
    └── graders/
        └── grader.py        # Master grader routing
```

---

## OpenEnv compliance

This environment implements the full OpenEnv interface:

- `POST /reset` — resets state, returns typed `State` observation
- `POST /step` — accepts typed `Action`, returns `observation`, `reward`, `done`, `info`
- `GET /state` — returns current state snapshot without side effects
- `openenv.yaml` — complete metadata including state/action spaces, reward formula, task definitions

Validated with `openenv validate`.

---

## Baseline scores

Scores produced by the MA5/MA10 crossover rule-based agent on synthetic data (yfinance fallback):

| Task | Score | Profit |
|---|---|---|
| easy | 0.095 | $47.50 |
| medium | 0.088 | $32.10 |
| hard | 0.076 | $58.20 |

Scores vary slightly with market data — the synthetic fallback uses a fixed seed (`np.random.seed(42)`) for reproducibility.

---

## License

MIT