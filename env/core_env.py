import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.models import State, Action, StepResult
from env.reward import compute_reward

# ------------------------------------------------------------------ #
#  Core Trading Environment                                          #
# ------------------------------------------------------------------ #

SUPPORTED_TASKS  = ("easy", "medium", "hard")
INITIAL_CASH     = 10_000.0
PRICE_HISTORY_LEN = 20          # how many past prices to keep in state


class TradingEnv:
    """
    OpenEnv-compatible trading environment.

    Supports three tasks: easy, medium, hard.
    Each task loads its own price series and grader.
    """

    def __init__(self, task: str = "easy"):
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from {SUPPORTED_TASKS}")

        self.task         = task
        self.prices       = []
        self.current_step = 0
        self.cash         = INITIAL_CASH
        self.position     = 0.0          # units held
        self.peak_portfolio = INITIAL_CASH
        self.portfolio_history = []
        self.trade_log    = []
        self._load_task_data()

    # ---------------------------------------------------------------- #
    #  Internal helpers                                                #
    # ---------------------------------------------------------------- #

    def _load_task_data(self):
        """Load price data for the selected task."""
        if self.task == "easy":
            from env.tasks.easy import load_data
        elif self.task == "medium":
            from env.tasks.medium import load_data
        else:
            from env.tasks.hard import load_data

        self.prices = load_data()
        self.max_steps = len(self.prices) - 1

    def _portfolio_value(self, step: int = None) -> float:
        """Total portfolio = cash + position * current_price."""
        idx = step if step is not None else self.current_step
        return self.cash + self.position * self.prices[idx]

    def _compute_ma(self, period: int) -> float:
        """Moving average of last `period` closing prices."""
        start = max(0, self.current_step - period + 1)
        window = self.prices[start : self.current_step + 1]
        return float(np.mean(window))

    def _compute_sharpe(self) -> float:
        """Rolling Sharpe ratio from portfolio history."""
        if len(self.portfolio_history) < 2:
            return 0.0
        values  = np.array(self.portfolio_history[-60:], dtype=float)  # last 60 steps
        returns = np.diff(values) / (values[:-1] + 1e-8)
        if np.std(returns) < 1e-8:
            return 0.0
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        return round(float(sharpe), 4)

    def _build_state(self) -> State:
        """Construct and return the current State object."""
        start = max(0, self.current_step - PRICE_HISTORY_LEN + 1)
        price_history = self.prices[start : self.current_step + 1]

        # Pad with first price if history is shorter than window
        if len(price_history) < PRICE_HISTORY_LEN:
            pad = [price_history[0]] * (PRICE_HISTORY_LEN - len(price_history))
            price_history = pad + list(price_history)

        return State(
            price_history  = price_history,
            current_price  = self.prices[self.current_step],
            position       = self.position,
            cash           = round(self.cash, 4),
            ma5            = round(self._compute_ma(5), 4),
            ma10           = round(self._compute_ma(10), 4),
            sharpe         = self._compute_sharpe(),
            step           = self.current_step,
        )

    # ---------------------------------------------------------------- #
    #  Public API                                                      #
    # ---------------------------------------------------------------- #

    def reset(self) -> State:
        """
        Reset environment to initial state.

        Returns:
            Initial State object
        """
        self.current_step      = 0
        self.cash              = INITIAL_CASH
        self.position          = 0.0
        self.peak_portfolio    = INITIAL_CASH
        self.portfolio_history = [INITIAL_CASH]
        self.trade_log         = []

        return self._build_state()

    def state(self) -> State:
        """Return current state snapshot without advancing the step."""
        return self._build_state()

    def step(self, action: Action) -> StepResult:
        """
        Execute one action and advance the environment by one timestep.

        Args:
            action : Action object with action ("BUY"/"SELL"/"HOLD") and quantity

        Returns:
            StepResult with observation, reward, done, info
        """
        if self.current_step >= self.max_steps:
            return StepResult(
                observation=self._build_state(),
                reward=0.0,
                done=True,
                info={"msg": "Episode already finished"}
            )

        current_price  = self.prices[self.current_step]
        prev_portfolio = self._portfolio_value()
        act            = action.action.upper()
        qty            = max(0.0, action.quantity)
        trade_value    = 0.0
        trade_info     = {}

        # ---- Execute action ---------------------------------------- #
        if act == "BUY":
            max_affordable = self.cash / (current_price + 1e-8)
            qty            = min(qty, max_affordable)
            cost           = qty * current_price
            if qty > 0 and cost <= self.cash:
                self.cash     -= cost
                self.position += qty
                trade_value    = cost
                trade_info     = {"action": "BUY", "price": current_price, "quantity": qty}
                self.trade_log.append(trade_info)

        elif act == "SELL":
            qty        = min(qty, self.position)
            proceeds   = qty * current_price
            if qty > 0:
                self.cash     += proceeds
                self.position -= qty
                trade_value    = proceeds
                trade_info     = {"action": "SELL", "price": current_price, "quantity": qty}
                self.trade_log.append(trade_info)

        # ---- Advance timestep -------------------------------------- #
        self.current_step += 1
        current_portfolio  = max(self._portfolio_value(), 0.0)

        # Update peak for drawdown calculation
        if current_portfolio > self.peak_portfolio:
            self.peak_portfolio = current_portfolio

        self.portfolio_history.append(current_portfolio)

        # ---- Compute reward ---------------------------------------- #
        reward_info = compute_reward(
            prev_portfolio    = prev_portfolio,
            current_portfolio = current_portfolio,
            peak_portfolio    = self.peak_portfolio,
            action            = act,
            trade_value       = trade_value,
        )

        done = self.current_step >= self.max_steps

        info = {
            "portfolio_value": round(current_portfolio, 2),
            "cash":            round(self.cash, 2),
            "position":        self.position,
            "step":            self.current_step,
            "max_steps":       self.max_steps,
            "trade":           trade_info,
            "return_pct":      (current_portfolio - INITIAL_CASH) / INITIAL_CASH,
            **reward_info,
        }

        return StepResult(
            observation = self._build_state(),
            reward      = round(reward_info["reward"], 6),
            done        = done,
            info        = info,
        )

    def final_score(self) -> dict:
        """
        Compute final grade after episode ends.

        Returns:
            dict from master grader with score, profit, breakdown
        """
        from env.graders.grader import grade

        final_portfolio = self._portfolio_value()
        return grade(
            task              = self.task,
            final_portfolio   = final_portfolio,
            initial_cash      = INITIAL_CASH,
            portfolio_history = self.portfolio_history,
            trade_log         = self.trade_log,
        )


# ------------------------------------------------------------------ #
#  Quick smoke test                                                  #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    for task in SUPPORTED_TASKS:
        print(f"\n=== Task: {task} ===")
        env   = TradingEnv(task=task)
        state = env.reset()
        print(f"  Initial price : ${state.current_price:.2f}")
        print(f"  Cash          : ${state.cash:.2f}")

        # Run 5 random steps
        import random
        for _ in range(5):
            act = random.choice(["BUY", "SELL", "HOLD"])
            qty = round(random.uniform(0.1, 2.0), 2)
            result = env.step(Action(action=act, quantity=qty))
            print(f"  {act:4s} {qty:.2f} → reward={result.reward:+.4f}  portfolio=${result.info['portfolio_value']:.2f}  done={result.done}")

        score = env.final_score()
        print(f"  Final score: {score['score']}  profit=${score['profit']}")