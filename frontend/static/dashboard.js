const BASE = '';

const MAX_STEPS = { easy: 1256, medium: 1256, hard: 1824 };

// ── Multi-asset portfolio allocation strategies ──────────────────────────────
const STRATEGIES = {
  easy: {
    // 3 assets: equal-weight with quarterly rebalancing
    getWeights: (step, state) => {
      const symbols = state.symbols || ['AAPL', 'MSFT', 'GOOGL'];
      const weights = {};
      symbols.forEach(s => weights[s] = 1.0 / symbols.length);
      return weights;
    }
  },
  medium: {
    // 5 assets: momentum-weighted (top 3 get more allocation)
    getWeights: (step, state) => {
      const symbols = state.symbols || ['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'JPM'];
      const prices = state.prices || {};
      
      // Simple momentum: allocate more to recent winners
      const momentum = {};
      symbols.forEach(s => {
        momentum[s] = Math.random(); // placeholder - in real case, use price history
      });
      
      // Top 60% to top 3, 40% to bottom 2
      const sorted = Object.entries(momentum).sort((a, b) => b[1] - a[1]);
      const weights = {};
      sorted.slice(0, 3).forEach(([s, _]) => weights[s] = 0.20);
      sorted.slice(3).forEach(([s, _]) => weights[s] = 0.20);
      
      return weights;
    }
  },
  hard: {
    // 7 assets: risk-parity (reduce crypto allocation due to volatility)
    getWeights: (step, state) => {
      const symbols = state.symbols || ['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'JPM', 'BTC-USD', 'ETH-USD'];
      const weights = {};
      
      symbols.forEach(s => {
        if (s.includes('BTC') || s.includes('ETH')) {
          weights[s] = 0.10; // Lower weight for crypto
        } else {
          weights[s] = 0.16; // Higher weight for stocks
        }
      });
      
      // Normalize to sum to 1.0
      const total = Object.values(weights).reduce((a, b) => a + b, 0);
      Object.keys(weights).forEach(s => weights[s] /= total);
      
      return weights;
    }
  }
};

// ── per-task memory store ─────────────────────────────────────────────────────
const taskMemory = {
  easy:   { labels: [], values: [], allocation: [] },
  medium: { labels: [], values: [], allocation: [] },
  hard:   { labels: [], values: [], allocation: [] },
};

function clearMemory(taskId) {
  taskMemory[taskId] = { labels: [], values: [], allocation: [] };
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
          label: 'Portfolio Value',
          data: memory ? [...memory.values] : [],
          borderColor: color,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
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

// ── alpine component ──────────────────────────────────────────────────────────
function dashboard() {
  return {
    apiOnline: false,
    activeTask: 'easy',
    running: false,
    progress: 0,
    avgScore: null,
    currentAllocation: [],

    tasks: [
      { id: 'easy',   label: 'Easy (3 assets)',   color: '#00ffe7', score: null, profit: null, steps: null },
      { id: 'medium', label: 'Medium (5 assets)', color: '#ffaa00', score: null, profit: null, steps: null },
      { id: 'hard',   label: 'Hard (7 assets)',   color: '#ff00cc', score: null, profit: null, steps: null },
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
      this.currentAllocation = [...taskMemory[id].allocation];
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
      this.currentAllocation = [];

      const mem   = taskMemory[taskId];
      const chart = createChart(t.color, mem);

      try {
        // 1. RESET
        const resetRes = await fetch(`${BASE}/reset`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task: taskId }),
        });
        const resetJson = await resetRes.json();
        const obs = resetJson.observation;

        const symbols = Object.keys(obs.prices || {});
        const maxSteps = MAX_STEPS[taskId];
        let step = 0;
        let done = false;

        // 2. STEP LOOP
        while (!done && step < maxSteps) {
          step++;

          // Get portfolio weights from strategy
          const weights = STRATEGIES[taskId].getWeights(step, { symbols, prices: obs.prices });

          const stepRes = await fetch(`${BASE}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task: taskId, weights }),
          });

          const result = await stepRes.json();
          done  = result.done ?? false;
          const nextObs = result.observation ?? {};

          const portfolioValue = nextObs.portfolio_value ?? 10000;

          // Update memory
          mem.labels.push(step);
          mem.values.push(portfolioValue);

          // Update chart every 50 steps (not every step - much faster!)
          if (step % 50 === 0 || done) {
            chart.data.labels = [...mem.labels];
            chart.data.datasets[0].data = [...mem.values];
            chart.update('none');
          }

          // Update current allocation display
          const allocArray = Object.entries(nextObs.weights || {})
            .map(([symbol, weight]) => ({ symbol, weight }))
            .sort((a, b) => b.weight - a.weight);
          
          this.currentAllocation = allocArray;
          mem.allocation = allocArray;

          this.progress = Math.min(99, Math.round((step / maxSteps) * 100));
          
          // Only delay every 10 steps to speed up
          if (step % 10 === 0) {
            await new Promise(r => setTimeout(r, 1));
          }
        }

        t.steps = step;

        // 3. GRADER
        const gradeRes = await fetch(`${BASE}/grader`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task: taskId }),
        });
        const grade = await gradeRes.json();
        t.score  = grade.sharpe ?? 0;
        t.profit = Math.round((grade.portfolio_value ?? 10000) - 10000);

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