# File: env/core_env.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from env.models import State, Action, StepResult

SUPPORTED_TASKS = ("easy", "medium", "hard")
INITIAL_CASH = 10_000.0


class TradingEnv:
    def __init__(self, task: str = "easy"):
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unknown task '{task}'")

        self.task = task
        self.symbols = []
        self.data: pd.DataFrame = None

        self.current_step = 0
        self.portfolio_value = INITIAL_CASH
        self.weights = {}
        self.history = []

        self._load_task_data()

    # ------------------------------------------------------------ #

    def _load_task_data(self):
        if self.task == "easy":
            path = "data/easy.csv"
        elif self.task == "medium":
            path = "data/medium.csv"
        else:
            path = "data/hard.csv"

        self.data = pd.read_csv(path, index_col=0)
        self.symbols = list(self.data.columns)
        self.max_steps = len(self.data) - 1

    # ------------------------------------------------------------ #

    def reset(self) -> State:
        self.current_step = 0
        self.portfolio_value = INITIAL_CASH
        self.weights = {s: 0.0 for s in self.symbols}
        self.history = [INITIAL_CASH]

        return self._build_state()

    def _build_state(self) -> State:
        row = self.data.iloc[self.current_step]

        return State(
            step=self.current_step,
            prices={s: float(row[s]) for s in self.symbols},
            portfolio_value=float(self.portfolio_value),
            weights=self.weights.copy(),
            cash_fraction=1.0 - sum(self.weights.values()),
        )

    def state(self) -> State:
        """Public method to get current state (used by /state route)"""
        return self._build_state()

    # ------------------------------------------------------------ #

    def step(self, action: Action) -> StepResult:
        if self.current_step >= self.max_steps:
            return StepResult(
                observation=self._build_state(),
                reward=0.0,
                done=True,
                info={}
            )

        new_weights = action.weights.copy()

        # Normalize weights
        total = sum(new_weights.values())
        if total > 1.0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        prev_prices = self.data.iloc[self.current_step]
        self.current_step += 1
        curr_prices = self.data.iloc[self.current_step]

        # Portfolio return
        portfolio_return = 0.0
        for s in self.symbols:
            if prev_prices[s] > 0:
                r = (curr_prices[s] / prev_prices[s]) - 1.0
                portfolio_return += new_weights.get(s, 0.0) * r

        cash_weight = 1.0 - sum(new_weights.values())
        portfolio_return += cash_weight * 0.0

        self.portfolio_value *= (1.0 + portfolio_return)
        self.weights = new_weights

        self.history.append(self.portfolio_value)

        done = self.current_step >= self.max_steps
        from env.reward import compute_sharpe
        reward = compute_sharpe(self.history) if done else 0.0

        return StepResult(
            observation=self._build_state(),
            reward=float(reward),
            done=done,
            info={
                "portfolio_value": float(self.portfolio_value),
                "return": float(portfolio_return)
            }
        )

    # ------------------------------------------------------------ #

    def final_score(self) -> dict:
        from env.reward import compute_sharpe
        return {
            "portfolio_value": float(self.portfolio_value),
            "sharpe": float(compute_sharpe(self.history))
        }