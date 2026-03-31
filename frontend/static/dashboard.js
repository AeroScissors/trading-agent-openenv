const BASE = '';

const MAX_STEPS = { easy: 249, medium: 249, hard: 249 };

const SAFE_DEFAULTS = {
  easy:   { pos_size: 0.95, stop_loss_pct: 0.85, sell_end: false, take_profit: 1.20 },
  medium: { pos_size: 0.90, stop_loss_pct: 0.05, sell_end: true,  take_profit: 999.0 },
  hard:   { pos_size: 0.90, stop_loss_pct: 0.80, sell_end: false, take_profit: 999.0 },
};

const agentState = {
  entry_price: 0.0,
  prev_ma5: null,
  prev_ma10: null,
  death_cnt: 0,
};

function resetAgentState() {
  agentState.entry_price = 0.0;
  agentState.prev_ma5    = null;
  agentState.prev_ma10   = null;
  agentState.death_cnt   = 0;
}

// ── per-task memory store ─────────────────────────────────────────────────────
const taskMemory = {
  easy:   { labels: [], prices: [], buys: [], sells: [], actionLog: [] },
  medium: { labels: [], prices: [], buys: [], sells: [], actionLog: [] },
  hard:   { labels: [], prices: [], buys: [], sells: [], actionLog: [] },
};

function clearMemory(taskId) {
  taskMemory[taskId] = { labels: [], prices: [], buys: [], sells: [], actionLog: [] };
}

// ── chart ─────────────────────────────────────────────────────────────────────
let _chart = null;

function destroyChart() {
  if (_chart) { _chart.destroy(); _chart = null; }
}

function createChart(color, memory) {
  destroyChart();
  const ctx = document.getElementById('priceChart');
  if (!ctx) return null;

  _chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: memory ? [...memory.labels] : [],
      datasets: [
        {
          label: 'price',
          data: memory ? [...memory.prices] : [],
          borderColor: color,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
        },
        {
          label: 'buy',
          data: memory ? [...memory.buys] : [],
          type: 'scatter',
          pointRadius: 8,
          pointStyle: 'triangle',
          backgroundColor: '#22c55e',
          borderColor: '#22c55e',
          showLine: false,
        },
        {
          label: 'sell',
          data: memory ? [...memory.sells] : [],
          type: 'scatter',
          pointRadius: 8,
          pointStyle: 'rectRot',
          backgroundColor: '#ef4444',
          borderColor: '#ef4444',
          showLine: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        zoom: {
          pan: {
            enabled: true,
            mode: 'x',
            cursor: 'grab',
          },
          zoom: {
            wheel: { enabled: true, speed: 0.1 },
            pinch: { enabled: true },
            mode: 'x',
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#5a5a7a', font: { size: 10 }, maxTicksLimit: 8 },
          grid:  { color: 'rgba(255,255,255,0.03)' },
          border: { color: '#1e1e30' },
        },
        y: {
          ticks: { color: '#5a5a7a', font: { size: 10 } },
          grid:  { color: 'rgba(255,255,255,0.03)' },
          border: { color: '#1e1e30' },
        },
      },
    },
  });
  return _chart;
}

// ── agent logic ───────────────────────────────────────────────────────────────
function getAction(task, step, state, maxSteps) {
  if (task === 'medium') {
    const cash     = state?.cash ?? 0;
    const position = state?.position ?? 0;
    if (step === 1)             return { action: 'BUY',  quantity: Math.round((cash * 0.90) / (state?.current_price ?? 1) * 10000) / 10000 };
    if (step === maxSteps - 1)  return { action: 'SELL', quantity: Math.round(position * 10000) / 10000 };  // exact step only, no double SELL
    return { action: 'HOLD', quantity: 0.0 };
  }

  const strategy = SAFE_DEFAULTS[task];
  const price    = state?.current_price ?? 0;
  const cash     = state?.cash ?? 0;
  const position = state?.position ?? 0;
  const ma5      = state?.ma5  ?? null;
  const ma10     = state?.ma10 ?? null;
  const sharpe   = state?.sharpe ?? 1.0;

  if (price <= 0) return { action: 'HOLD', quantity: 0.0 };

  function buy() {
    const qty = (cash * strategy.pos_size) / price;
    agentState.entry_price = price;
    agentState.death_cnt   = 0;
    return { action: 'BUY', quantity: Math.round(qty * 10000) / 10000 };
  }

  function sell() {
    agentState.entry_price = 0.0;
    agentState.death_cnt   = 0;
    return { action: 'SELL', quantity: Math.round(position * 10000) / 10000 };
  }

  let golden = false;
  let death  = false;

  if (ma5 !== null && ma10 !== null && agentState.prev_ma5 !== null && agentState.prev_ma10 !== null) {
    const prev_above = agentState.prev_ma5 > agentState.prev_ma10;
    const curr_above = ma5 > ma10;
    if (!prev_above && curr_above) golden = true;
    if (prev_above && !curr_above) death  = true;
  }

  agentState.prev_ma5  = ma5;
  agentState.prev_ma10 = ma10;

  if (position === 0 && cash > 0) {
    if (step === 1)  return buy();
    if (golden)      return buy();
    if (step === 10) return buy();
  }

  if (position > 0) {
    const entry = agentState.entry_price;
    if (entry > 0 && price > entry * strategy.take_profit)   return sell();
    if (entry > 0 && price < entry * strategy.stop_loss_pct) return sell();
    if (sharpe < 0.3 && entry > 0 && price < entry * 0.95)  return sell();

    if (death || (ma5 !== null && ma10 !== null && ma5 < ma10)) {
      agentState.death_cnt++;
    } else {
      agentState.death_cnt = 0;
    }

    if (agentState.death_cnt >= 3) return sell();
    if (strategy.sell_end && step >= maxSteps - 1) return sell();
  }

  return { action: 'HOLD', quantity: 0.0 };
}

// ── alpine component ──────────────────────────────────────────────────────────
function dashboard() {
  return {
    apiOnline: false,
    activeTask: 'easy',
    running: false,
    progress: 0,
    avgScore: null,
    actionLog: [],

    tasks: [
      { id: 'easy',   label: 'Easy · AAPL',   color: '#22c55e', score: null, profit: null, steps: null },
      { id: 'medium', label: 'Medium · MSFT',  color: '#f59e0b', score: null, profit: null, steps: null },
      { id: 'hard',   label: 'Hard · BTC-USD', color: '#ef4444', score: null, profit: null, steps: null },
    ],

    getTask(id) {
      return this.tasks.find(t => t.id === id);
    },

    async init() {
      try {
        const r = await fetch(`${BASE}/tasks`);
        this.apiOnline = r.ok;
      } catch {
        this.apiOnline = false;
      }
      const t = this.getTask(this.activeTask);
      createChart(t.color, taskMemory[this.activeTask]);
    },

    switchTask(id) {
      if (this.running) return;
      this.activeTask = id;
      this.actionLog = [...taskMemory[id].actionLog];
      const t = this.getTask(id);
      createChart(t.color, taskMemory[id]);
    },

    async runAgent() {
      if (this.running) return;
      this.running  = true;
      this.progress = 0;

      const taskId = this.activeTask;
      const t      = this.getTask(taskId);
      t.score  = null;
      t.profit = null;
      t.steps  = null;

      clearMemory(taskId);
      this.actionLog = [];
      taskMemory[taskId].actionLog = [];

      const mem   = taskMemory[taskId];
      const chart = createChart(t.color, mem);

      resetAgentState();

      try {
        // 1. RESET
        const resetRes = await fetch(`${BASE}/reset`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task: taskId }),
        });
        const resetJson = await resetRes.json();
        let state = resetJson.initial_state;

        const maxSteps = MAX_STEPS[taskId];
        let step = 0;
        let done = false;

        // 2. STEP LOOP
        while (!done) {
          step++;

          const act = getAction(taskId, step, state, maxSteps);

          const stepRes = await fetch(`${BASE}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task: taskId, action: act.action, quantity: act.quantity }),
          });

          const result = await stepRes.json();
          done  = result.done ?? false;
          state = result.observation ?? {};

          const price  = state.current_price ?? 0;
          const reward = result.reward ?? 0;

          // update chart + memory
          chart.data.labels.push(step);
          mem.labels.push(step);

          chart.data.datasets[0].data.push(price);
          mem.prices.push(price);

          if (act.action === 'BUY') {
            chart.data.datasets[1].data.push({ x: step, y: price });
            mem.buys.push({ x: step, y: price });
          } else if (act.action === 'SELL') {
            chart.data.datasets[2].data.push({ x: step, y: price });
            mem.sells.push({ x: step, y: price });
          }

          chart.update('none');

          // action log — medium logs every 5 steps, others every 10
          const logInterval = taskId === 'medium' ? 5 : 10;
          if (act.action !== 'HOLD' || step % logInterval === 0) {
            const entry = { step, action: act.action, reward };
            this.actionLog.unshift(entry);
            taskMemory[taskId].actionLog.unshift(entry);
            if (this.actionLog.length > 20) {
              this.actionLog.pop();
              taskMemory[taskId].actionLog.pop();
            }
          }

          this.progress = Math.min(99, Math.round((step / maxSteps) * 100));
          await new Promise(r => setTimeout(r, 15));
        }

        t.steps = step;

        // 3. GRADER
        const gradeRes = await fetch(`${BASE}/grader`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task: taskId }),
        });
        const grade = await gradeRes.json();
        t.score  = grade.score  ?? 0;
        t.profit = Math.round(grade.profit ?? 0);

        const scored = this.tasks.filter(x => x.score !== null);
        this.avgScore = scored.reduce((a, x) => a + x.score, 0) / scored.length;

      } catch (e) {
        console.error('Agent run failed:', e);
      }

      this.running  = false;
      this.progress = 100;
    },
  };
}