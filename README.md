---
title: Trading Agent OpenEnv
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_file: main.py
pinned: false
tags:
  - openenv
  - portfolio-management
  - reinforcement-learning
---

# Trading Agent OpenEnv v2.0

A complete [OpenEnv](https://openenv.ai)-compatible reinforcement learning environment for training and evaluating **multi-asset portfolio management** agents. Agents learn to allocate capital across diversified portfolios and are evaluated on **risk-adjusted returns** (Sharpe ratio) rather than raw profit.

**Live demo:** [HuggingFace Space](https://huggingface.co/spaces/AeroScissors/trading-agent-openenv)

---

## What's new in v2.0?

**Major upgrade from single-asset trading to multi-asset portfolio management:**

- **Multi-asset portfolios:** 3/5/7 asset tasks instead of single stocks
- **Risk-adjusted evaluation:** Sharpe ratio (returns / volatility) replaces profit-only scoring
- **Diversification strategies:** Agents learn to balance stocks, healthcare, finance, and crypto
- **Real market data:** 2020-2024 daily prices from Yahoo Finance (AAPL, MSFT, GOOGL, JNJ, JPM, BTC-USD, ETH-USD)
- **Interactive dashboard:** Live portfolio allocation visualization with <2min episodes

---

## Why this environment?

Most RL environments use games or toy problems. **Portfolio management** is one of the clearest examples of sequential decision-making under uncertainty in the real world:

- Agents observe market conditions across multiple assets
- Actions have direct financial consequences (capital allocation)
- Rewards reflect what professional investors care about: **risk-adjusted returns**

This environment models modern portfolio theory faithfully — with real correlation structures, realistic volatility, and evaluation metrics used by institutional investors.

---

## Environment description

The environment simulates a **multi-asset portfolio account**. At each timestep the agent receives market observations (prices, portfolio state) and outputs **target allocation weights** for each asset. The portfolio rebalances accordingly, and the agent receives a reward signal based on the **Sharpe ratio** of the portfolio's returns.

Price data spans **2020-2024** from Yahoo Finance. Assets are grouped into three difficulty tiers with increasing correlation complexity and volatility.

---

## Observation space

Each observation is a `State` object with the following fields:

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Current timestep index |
| `prices` | `dict[str, float]` | Current price per asset, e.g. `{"AAPL": 180.5, "MSFT": 320.2}` |
| `portfolio_value` | `float` | Total portfolio value (assets + cash) |
| `weights` | `dict[str, float]` | Current allocation per asset, e.g. `{"AAPL": 0.33, "MSFT": 0.33}` |
| `cash_fraction` | `float` | Fraction of portfolio held in cash (0.0 to 1.0) |

---

## Action space

Each action specifies **target portfolio weights**:

| Field | Type | Description |
|---|---|---|
| `weights` | `dict[str, float]` | Target allocation per asset, e.g. `{"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}` |

**Constraints:**
- All weights must be non-negative
- Sum of weights must be ≤ 1.0 (remaining is cash)
- Weights are auto-normalized if sum > 1.0

**Example:**
```python
action = {
    "weights": {
        "AAPL": 0.35,
        "MSFT": 0.30,
        "GOOGL": 0.25
    }
    # Remaining 10% stays in cash
}
```

---

## Reward function

**Sharpe ratio** (annualized, risk-adjusted returns):

```
reward = (mean_returns / std_returns) × sqrt(252)
```

Clipped to **[-5.0, 5.0]** to handle extreme crypto volatility.

| Component | Description |
|---|---|
| `mean_returns` | Average daily portfolio return |
| `std_returns` | Standard deviation of daily returns (volatility) |
| `sqrt(252)` | Annualization factor (252 trading days/year) |

**Why Sharpe ratio?**
- Penalizes volatility (encourages stable growth)
- Standard metric in finance (comparable across strategies)
- Encourages diversification (uncorrelated assets reduce std_returns)

---

## Tasks

### Task 1 — Balanced Portfolio (Easy)

- **Assets:** AAPL, MSFT, GOOGL (3 large-cap tech stocks)
- **Objective:** Manage a simple 3-asset portfolio with high correlation
- **Initial cash:** $10,000
- **Sharpe target:** 1.0
- **Difficulty:** Equal-weight baseline achieves ~0.95 Sharpe on clean uptrend data
- **Baseline strategy:** Equal weight (33.3% per asset)
- **Expected score:** ~1.0 (normalized)

### Task 2 — Mixed Sectors (Medium)

- **Assets:** AAPL, MSFT, GOOGL, JNJ, JPM (tech + healthcare + finance)
- **Objective:** Diversify across sectors with lower correlation
- **Initial cash:** $10,000
- **Sharpe target:** 0.9
- **Difficulty:** Lower correlation requires strategic rebalancing
- **Baseline strategy:** Equal weight (20% per asset)
- **Expected score:** ~1.0 (normalized)

### Task 3 — Stocks + Crypto (Hard)

- **Assets:** AAPL, MSFT, GOOGL, JNJ, JPM, BTC-USD, ETH-USD (5 stocks + 2 crypto)
- **Objective:** Balance stable stocks with high-volatility crypto
- **Initial cash:** $10,000
- **Sharpe target:** 1.2
- **Difficulty:** Crypto's 10x volatility requires risk-parity allocation
- **Baseline strategy:** 80% stocks, 20% crypto (risk-adjusted)
- **Expected score:** ~1.0 (normalized)
- **Note:** Crypto trades 24/7 with extreme volatility — naive equal-weight fails

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/tasks` | List all tasks with metadata |
| `POST` | `/reset` | Reset environment, returns initial state |
| `GET` | `/state` | Get current state without advancing timestep |
| `POST` | `/step` | Execute portfolio rebalancing, returns obs/reward/done |
| `POST` | `/grader` | Return final Sharpe and normalized score (0.0–1.0) |

All endpoints accept and return JSON. Visit `/docs` for interactive Swagger UI.

### Example: full episode

```python
import requests

BASE = "http://localhost:7860"

# Reset to Medium task (5 assets)
response = requests.post(f"{BASE}/reset", json={"task": "medium"}).json()
state = response["observation"]

done = False
while not done:
    # Agent decides portfolio allocation
    weights = {
        "AAPL": 0.25,
        "MSFT": 0.20,
        "GOOGL": 0.20,
        "JNJ": 0.20,
        "JPM": 0.15
    }
    
    result = requests.post(f"{BASE}/step", json={
        "task": "medium",
        "weights": weights
    }).json()
    
    done = result["done"]

# Get final score
grade = requests.post(f"{BASE}/grader", json={"task": "medium"}).json()
print(f"Sharpe: {grade['sharpe']:.4f}")
print(f"Score: {grade['score']:.4f}")
print(f"Final value: ${grade['portfolio_value']:,.0f}")
```

---

## Setup and usage

### Run locally with Docker

```bash
docker build -t trading-openenv .
docker run -p 7860:7860 trading-openenv
```

The dashboard will be available at `http://localhost:7860`.

### Run locally without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data (downloads real prices from Yahoo Finance)
python generate_data.py

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Interactive dashboard

Visit `http://localhost:7860` to see:
- Live portfolio value charts
- Real-time allocation bars per asset
- Sharpe ratio scores
- Episode replay with pan/zoom

All 3 tasks complete in **~6 minutes total** on HuggingFace Spaces.

---

## Project structure

```
.
├── server/
│   └── app.py               # FastAPI app entry point
├── api/
│   └── routes.py            # All HTTP endpoints
├── env/
│   ├── core_env.py          # TradingEnv class (multi-asset logic)
│   ├── models.py            # Pydantic models (State, Action, StepResult)
│   └── reward.py            # Sharpe ratio computation
├── data/
│   ├── easy.csv             # 3-asset price data (2020-2024)
│   ├── medium.csv           # 5-asset price data
│   └── hard.csv             # 7-asset price data
├── frontend/
│   ├── index.html           # Dashboard UI
│   └── static/
│       ├── style.css        # Dark terminal theme
│       └── dashboard.js     # Chart.js + Alpine.js
├── generate_data.py         # Download real prices from yfinance
├── openenv.yaml             # OpenEnv compliance manifest
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## OpenEnv compliance

This environment implements the full OpenEnv interface:

- ✅ `POST /reset` — returns typed `State` observation
- ✅ `POST /step` — accepts typed `Action`, returns obs/reward/done/info
- ✅ `GET /state` — current state snapshot without side effects
- ✅ `openenv.yaml` — complete metadata (state/action spaces, reward, tasks)
- ✅ Multi-task support (3 difficulty levels)
- ✅ Docker deployment
- ✅ Interactive dashboard

Validated with `openenv validate`.

---

## Baseline scores

Equal-weight allocation strategy on real 2020-2024 market data:

| Task | Assets | Sharpe | Normalized Score | Final Value |
|---|---|---|---|---|
| Easy | 3 (tech) | 1.45 | **1.0000** | $31,398 |
| Medium | 5 (diversified) | 1.24 | **1.0000** | $24,770 |
| Hard | 7 (+crypto) | 1.35 | **1.0000** | $45,818 |

**Average:** 1.0000 (all tasks exceed target Sharpe ratios)

---

## Data source

Real historical prices from Yahoo Finance (2020-01-01 to 2024-12-31):

- **Stocks:** AAPL, MSFT, GOOGL (tech), JNJ (healthcare), JPM (finance)
- **Crypto:** BTC-USD, ETH-USD (24/7 markets, high volatility)
- **Preprocessing:** Forward-fill missing values, align indices across assets
- **Frequency:** Daily closes (~1257 rows for stocks, ~1825 for crypto)

Data is bundled in the repository — no network access required at runtime.

---

## Research applications

This environment is designed for:

- **RL algorithm research:** Test policy gradient, DQN, PPO on real-world sequential decisions
- **Portfolio optimization:** Compare learned strategies vs. traditional approaches (Markowitz, risk-parity)
- **Transfer learning:** Train on Easy, test on Hard (different correlation structures)
- **Multi-objective RL:** Balance return vs. volatility vs. drawdown
- **LLM-based agents:** Natural language interface to portfolio management

---

## License

MIT

---

## Citation

```bibtex
@software{trading_agent_openenv_v2,
  title = {Trading Agent OpenEnv: Multi-Asset Portfolio Management Environment},
  author = {AeroScissors},
  year = {2026},
  url = {https://github.com/AeroScissors/trading-agent-openenv},
  version = {2.0.0}
}
```

---

## Acknowledgments

Built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology**.

Reference environments:
- [REPL Environment](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/repl_env)
- [Reasoning Gym](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/reasoning_gym_env)
- [Calendar Environment](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/calendar_env)