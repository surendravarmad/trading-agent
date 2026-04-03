# Autonomous Options Credit Spread Trading Agent

An autonomous trading agent specialized in generating daily income through high-probability, risk-defined options credit spreads. The agent's primary goal is **capital preservation** — it only enters trades where the maximum loss is known and capped, prioritizing time decay (Theta) over directional speculation.

---

## Architecture Overview

The agent runs a **two-stage loop** on every cycle: first it manages existing open positions, then it evaluates new trade opportunities across all configured tickers.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENT CYCLE  (agent.py)                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 1 — Monitor Existing Positions                           │   │
│  │                                                                  │   │
│  │  Position Monitor ──▶ Exit Signal Check ──▶ Order Tracker       │   │
│  │  (fetch spreads)      Stop-Loss 50%         (fills / cancels)   │   │
│  │                       Profit-Target 75%                          │   │
│  │                       Regime Shift                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 2 — Open New Positions  (per ticker: SPY QQQ AAPL …)    │   │
│  │                                                                  │   │
│  │  I·Perceive ──▶ II·Classify ──▶ III·Plan ──▶ IV·Risk ──▶       │   │
│  │  yfinance 200d   SMA-50/200     Select        8 guardrails       │   │
│  │  Alpaca snap.    RSI-14         strikes        Liquidity check    │   │
│  │  Bid/ask check   Bollinger BB   nearest Fri    Buying power       │   │
│  │  (batch+cached)  3-std MR sig.  in DTE range   Daily DD CB        │   │
│  │                  RS vs SPY/QQQ                                    │   │
│  │              ──▶ V·LLM Analysis ──▶ VI·Execute                  │   │
│  │                 RAG context         mleg order                   │   │
│  │                 Approve/Skip        Alpaca API                   │   │
│  │                 Confidence          Journal entry                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

> 📊 An interactive HTML architecture diagram is included: open `architecture_diagram.html` in any browser.

---

## Market Regime → Strategy Matrix

Strategy selection follows a strict priority order:

| Priority | Regime | Detection Rule | Strategy |
|----------|--------|---------------|----------|
| **1 (highest)** | **Mean Reversion** | Price touches 3-std Bollinger Band | Mean Reversion Spread |
| 2 | **Bullish + RS** | Bullish AND ticker outperforming SPY/QQQ by >0.1% in 5-min window | Bull Put Spread (RS bias) |
| 3 | **Sideways + RS** | Sideways AND ticker outperforming SPY/QQQ by >0.1% | Bull Put Spread (RS bias) |
| 4 | **Bullish** | Price > SMA-200 AND SMA-50 slope > 0 | Bull Put Spread |
| 5 | **Bearish** | Price < SMA-200 AND SMA-50 slope < 0 | Bear Call Spread |
| 6 | **Sideways** | Between SMAs or narrow Bollinger Bands | Iron Condor |

### Mean Reversion Spreads

When price reaches a **3-standard-deviation Bollinger Band** (statistically extreme), the agent expects reversion toward the mean and sells a spread in the direction of the expected move:

| Band Touch | Expected Move | Strategy |
|------------|--------------|----------|
| Upper 3-std touch | Price extended to upside → expect reversion down | Bear Call Spread above current price |
| Lower 3-std touch | Price extended to downside → expect reversion up | Bull Put Spread below current price |

The strategy name is labelled `"Mean Reversion Spread"` in all outputs.

### Relative Strength Bias

On every cycle, the agent computes each ticker's **5-minute return** via Alpaca bars and compares it to SPY and QQQ. If the ticker outperforms by >0.1% in the 5-min window, the strategy selection is biased toward a Bull Put Spread even in a sideways regime — indicating short-term momentum that supports a bullish credit spread.

---

## 5-Minute Cycle Optimisations

The agent is designed to complete a full cycle well within a 5-minute window:

| Optimisation | Where | Detail |
|---|---|---|
| **Parallel price-history fetch** | `market_data.py` | All tickers' 200-day OHLCV fetched concurrently via `ThreadPoolExecutor` before the ticker loop starts |
| **Batch snapshot call** | `market_data.py` | All current prices retrieved in **one** Alpaca API call (`?symbols=SPY,QQQ,…`) |
| **TTL-based caches** | `market_data.py` | Historical prices (4 h), stock snapshots (60 s), option chains (3 min) — empty results are never cached |
| **Hard timeout guard** | `agent.py` | A daemon timer fires at 270 s (4 min 30 s): logs a `cycle_timeout` event and calls `os._exit(1)` so the scheduler cleanly starts the next run |

### Live Quote Refresh at Execution

Option bid/ask can move significantly between Phase III (planning) and Phase VI (order submission). The executor always fetches a **fresh, no-cache quote** for the two leg symbols immediately before sending the order:

```
Phase III  fetch_option_chain()   ← full chain, 3-min TTL cache (scanning)
              net_credit locked into SpreadPlan
              ↓
           (risk checks, optional LLM analysis…)
              ↓
Phase VI   fetch_option_quotes([sold_symbol, bought_symbol])  ← no cache, 2 symbols only
              live_credit = sold.bid − bought.ask
              limit_price = −live_credit                      ← current market price
```

- If the live credit deviates **> 10%** from the plan, a `WARNING` is logged
- If the quote fetch fails (API timeout), the planned credit is used as fallback with a warning — the order still goes out
- The `PRICE_DRIFT_WARN_PCT = 0.10` threshold is a class constant in `OrderExecutor` and can be tightened

### Expiration Date Selection

`strategy.py` selects the **nearest Friday** to `TARGET_DTE=44`, clamped inside `DTE_RANGE=(21,45)`. Local date (`datetime.now()`) is used — not UTC — so the calculation matches the trading day the cron runs on.

---

## Intelligence Layer

The agent includes an optional LLM-powered intelligence layer that learns from every trade and improves decisions over time.

```
Trade Executes ──▶ Journal Entry Opened ──▶ LLM Post-Trade Analysis
      ──▶ Lessons → Knowledge Base ──▶ Better Decisions Next Cycle
      ──▶ (after 20+ trades) Fine-Tune Local Model
```

### Always-On Signal Journal

Regardless of whether the LLM layer is enabled, **every trade signal and execution attempt is logged** to `trade_journal/`:

- `signals.jsonl` — one JSON record per line; LLM fine-tuning and RAG-ready
- `signals.md` — append-only Markdown table for human review

Each record captures: `timestamp`, `ticker`, `action`, `price`, `exec_status`, and a `raw_signal` dict with regime, Greeks, risk checks, LLM fields, and a structured `thesis` block.

### Components

**LLM Client** (`llm_client.py`) — OpenAI-compatible interface supporting Ollama, LM Studio, or the Claude API. Defaults to `mistral` on a local Ollama instance. Swappable via `.env` with zero code changes.

**Trade Journal** (`trade_journal.py`) — Logs the full lifecycle of every trade: entry context (regime, indicators, Greeks, LLM reasoning), execution status, exit signal, realized P&L, and post-trade lessons. Also hosts `signals.jsonl` / `signals.md` (LLM-independent).

**Knowledge Base / RAG** (`knowledge_base.py`) — File-based vector store using `nomic-embed-text` embeddings and cosine similarity search. Before each trade, the LLM queries the KB for similar past trades and relevant lessons. No external vector DB required. Key methods:
- `update_trade_outcome()` — back-fills win/loss label + P&L into the KB document when a trade closes, so future similarity searches return outcome-aware results
- `query_by_metadata(filters)` — filter documents by strategy, regime, outcome label etc. for targeted fine-tuning queries
- `outcome_stats()` — win/loss breakdown by strategy and regime (edge-tracking)

**LLM Analyst** (`llm_analyst.py`) — Returns one of three actions:
- `approve` — proceed as planned
- `modify` — adjust strikes or sizing (within risk limits)
- `skip` — pass on this trade with reasoning

**Safety invariant:** The LLM analyst is advisory only. It cannot approve a trade the risk manager has already rejected, and can only tighten constraints — never loosen them.

**Fine-Tuning Pipeline** (`fine_tuning.py`) — Exports trade data in three formats once you have 20+ closed trades:
- Chat JSONL (Ollama / LM Studio)
- Alpaca instruction format (LoRA / QLoRA)
- DPO preference pairs (wins vs losses, paired by strategy + regime + indicator distance)

### Setting Up Ollama (Recommended)

```bash
chmod +x setup_intelligence.sh
./setup_intelligence.sh

# Or manually:
brew install ollama
ollama pull mistral
ollama pull nomic-embed-text
ollama serve
```

Then enable in `.env`:
```
LLM_ENABLED=true
LLM_PROVIDER=ollama
LLM_MODEL=mistral
```

Other recommended models: `llama3`, `deepseek-r1:7b`, `qwen2.5:7b`.

---

## Risk Management Guardrails

Every trade must pass **all eight checks** before execution:

| # | Check | Rule |
|---|-------|------|
| 1 | **Plan Validity** | Strategy planner found valid strikes and contracts |
| 2 | **Credit-to-Width Ratio** | Credit ≥ `MIN_CREDIT_RATIO` of spread width (default 0.33) |
| 3 | **Sold Delta** | ≤ `MAX_DELTA` (default 0.20, ≈80% probability OTM) |
| 4 | **Max Loss** | ≤ `MAX_RISK_PCT` × account equity per trade (default 2%) |
| 5 | **Account Type** | Must be `paper` |
| 6 | **Market Hours** | Market must be open |
| 7 | **Underlying Liquidity** | Stock bid/ask spread < `LIQUIDITY_MAX_SPREAD` (default $0.05) |
| 8 | **Buying Power** | Available buying power ≥ (1 − `MAX_BUYING_POWER_PCT`) × equity |

```
Max Loss = (Spread Width − Credit Collected) × 100
```

### Daily Drawdown Circuit Breaker

The agent persists today's opening equity in `trade_plans/daily_state.json` and checks it on every cycle. If the current equity has fallen more than `DAILY_DRAWDOWN_LIMIT` (default 5%) from the day's opening value, the agent:
1. Logs a `daily_drawdown_circuit_breaker` event to `signals.jsonl`
2. Calls `os._exit(1)` to hard-terminate the process

The file resets automatically at the start of each calendar day.

### Liquidation Mode

If available buying power exceeds `MAX_BUYING_POWER_PCT` (default 80%) of account equity, the agent enters **Liquidation Mode**: Stage 2 (new trade opening) is skipped entirely for all tickers. Stage 1 (position monitoring and closing) continues normally. Every skipped ticker is logged with action `skipped_liquidation_mode`.

### Capital Retainment Guards (defense_first)

Two additional entry filters block new trades and log `strategy_mode: defense_first` to `signals.jsonl`:

| Guard | Trigger | Effect |
|-------|---------|--------|
| **Macro Guard** | `price < SMA-200` when regime would generate a Bull Put Spread | Skips the entry; Bull spreads against the macro trend are blocked |
| **High-IV Block** | Realized-volatility IV rank > 95th percentile | Skips ALL new entries; extreme vol spikes make credit spreads dangerously wide |

Both log `action: skipped_defense_first` to JournalKB so the LLM training pipeline can learn from defensive decisions.

### Position Exit Debouncing

Non-immediate exit signals require **3 consecutive cycles** (≈ 15 minutes at the default 5-min interval) of the same signal before a spread is closed. This prevents whipsaw closes on transient market moves.

Signals that **bypass** debounce and close immediately:

| Signal | Trigger |
|--------|---------|
| `HARD_STOP` | Spread has lost ≥ 3× the initial credit collected |
| `STRIKE_PROXIMITY` | Underlying within 1% of any short strike |
| `DTE_SAFETY` | Thursday after 15:30 ET and expiry is the next day (Friday) |

Debounce vote counts are persisted in `trade_plans/daily_state.json` and reset each calendar day.

### Position Sizing — 1% Risk Rule

When submitting live orders, the executor calculates contract quantity dynamically:

```
max_loss_per_contract = (spread_width − net_credit) × 100
qty = floor(1% × account_equity / max_loss_per_contract),  minimum 1
```

This ensures no single trade risks more than 1% of total equity regardless of spread width or premium level.

---

## Project Structure

```
trading-agent/
├── .env                          # API keys and config (not committed)
├── requirements.txt
├── README.md
├── setup_intelligence.sh         # Ollama setup helper
├── run_tests.py                  # Full test suite runner
├── architecture_diagram.html     # Interactive architecture diagram
│
├── trading_agent/
│   ├── agent.py                  # Orchestrator: two-stage cycle, timeout guard, CLI
│   ├── config.py                 # AppConfig, IntelligenceConfig loaders
│   ├── logger_setup.py
│   │
│   │   # ── Core Phases ──
│   ├── market_data.py            # Phase I   — yfinance + Alpaca (TTL cache, parallel)
│   ├── regime.py                 # Phase II  — SMA / RSI / Bollinger regime classifier
│   ├── strategy.py               # Phase III — strike selection, nearest-Friday DTE
│   ├── risk_manager.py           # Phase IV  — 8-guardrail validator
│   ├── executor.py               # Phase VI  — mleg order execution + HTML report
│   ├── trade_plan_report.py      # HTML report generator (auto-called by executor)
│   │
│   │   # ── Position Management ──
│   ├── position_monitor.py       # Stage 1 — monitor & close open spreads
│   ├── order_tracker.py          # Stage 1 — fill tracking
│   │
│   │   # ── Intelligence Layer ──
│   ├── journal_kb.py             # Always-on signal logger (JSONL + Markdown)
│   ├── trade_journal.py          # Full-lifecycle trade logging (TradeEntry)
│   ├── knowledge_base.py         # File-based RAG vector store
│   ├── llm_client.py             # OpenAI-compatible LLM client
│   ├── llm_analyst.py            # Pre/post trade LLM analysis & decisions
│   └── fine_tuning.py            # Training data export (JSONL / Alpaca / DPO)
│
├── trade_journal/                # Trade lifecycle logs + signal journal (auto-created)
│   ├── trades/                   #   Per-trade JSON files (LLM layer)
│   ├── index.json                #   Lightweight lookup index
│   ├── stats.json                #   Aggregate performance stats
│   ├── signals.jsonl             #   Always-on signal log (every cycle, LLM-independent)
│   └── signals.md                #   Human-readable Markdown table of signals
│
├── knowledge_base/               # RAG vector store (auto-created, LLM layer only)
├── trade_plans/                  # Per-ticker persistent trade plan files (auto-created)
│   ├── trade_plan_{TICKER}.json  #   Single file per ticker with state_history array
│   ├── trade_plan_{TICKER}.html  #   Self-contained HTML report (auto-generated each cycle)
│   └── daily_state.json          #   Daily drawdown circuit breaker state (auto-reset)
└── logs/                         # Runtime log files
```

---

## Trade Plan File Format

Each ticker has one persistent file (`trade_plans/{TICKER}.json`) that accumulates a `state_history` array — no per-run file sprawl, and you can roll back to any historical state:

```json
{
  "ticker": "SPY",
  "created": "2026-04-01T15:00:00+00:00",
  "last_updated": "2026-04-01T15:58:00+00:00",
  "state_history": [
    {
      "run_id": "20260401_155800",
      "timestamp": "2026-04-01T15:58:00+00:00",
      "trade_plan": {
        "ticker": "SPY",
        "strategy": "Bull Put Spread",
        "legs": [...],
        "net_credit": 1.70,
        "max_loss": 330.0,
        "credit_to_width_ratio": 0.34
      },
      "risk_verdict": {
        "approved": true,
        "checks_passed": ["plan_valid", "credit_ratio", "delta", "max_loss", "paper_account", "market_open"],
        "checks_failed": []
      },
      "mode": "dry_run",
      "order_result": { "status": "submitted", "order_id": "..." }
    }
  ]
}
```

History is capped at 200 entries per ticker. Old timestamped files (`trade_plan_{TICKER}_{TS}.json`) are loaded transparently for backward compatibility.

---

## HTML Trade Plan Report

Every time a trade plan is saved, `executor.py` automatically generates a self-contained HTML report alongside the JSON:

```
trade_plans/
├── trade_plan_IWM.json      ← machine-readable, authoritative
└── trade_plan_IWM.html      ← human-readable, auto-generated each cycle
```

Open any `.html` file in a browser — no server required, no dependencies to install.

### Report Sections

| Section | What it shows |
|---------|--------------|
| **Summary Cards** | Strategy, net credit, max loss, credit/width ratio, expiration + DTE, account equity, approval rate |
| **Spread Structure** | Visual strike ladder with colour-coded sell/buy bars, delta, bid/ask, mid for each leg |
| **Option Legs Table** | Full per-leg breakdown — symbol, strike, action, type, delta, bid, ask, mid |
| **Risk Checks** | Colour-coded ✅ / ❌ list of all 8 guardrails with verdict summary |
| **Trade Thesis** | Structured *Why this market? / Why now? / Exit plan* block |
| **Historical Trend Charts** | Chart.js line charts: net credit, credit/width ratio (vs 33% threshold line), account balance |
| **Order Submission** | Order ID, Alpaca status, limit price, fill status, per-leg fill detail |
| **Cycle History** | Collapsible entry for every run (newest first, auto-expanded) — plan + risk + legs + order |

### Generating Reports Manually

```bash
# Single ticker
python -m trading_agent.trade_plan_report trade_plans/trade_plan_IWM.json

# All tickers in the directory
python -m trading_agent.trade_plan_report trade_plans/
```

Reports are also regenerated automatically on every agent cycle — no manual step needed during live operation.

---

## Signal Journal Format (`signals.jsonl`)

Every trade attempt is logged as a single JSON line, regardless of LLM state:

```json
{
  "timestamp": "2026-04-01T15:58:00.123456+00:00",
  "ticker": "SPY",
  "action": "dry_run",
  "price": 655.44,
  "exec_status": "dry_run",
  "notes": "dry_run: Bull Put Spread, cr=1.70, ratio=0.34",
  "raw_signal": {
    "regime": "bullish",
    "strategy": "Bull Put Spread",
    "plan_valid": true,
    "risk_approved": true,
    "net_credit": 1.70,
    "max_loss": 330.0,
    "credit_to_width_ratio": 0.34,
    "spread_width": 5.0,
    "expiration": "2026-05-15",
    "sma_50": 675.93,
    "sma_200": 658.41,
    "rsi_14": 37.8,
    "mean_reversion_signal": false,
    "mean_reversion_direction": "",
    "rs_vs_spy": 0.0012,
    "rs_vs_qqq": -0.0003,
    "account_balance": 25000.0,
    "checks_passed": ["plan_valid", "credit_ratio", "delta", "max_loss",
                      "paper_account", "market_open", "liquidity", "buying_power"],
    "checks_failed": [],
    "llm_decision": null,
    "llm_confidence": null,
    "thesis": {
      "why": "BULLISH regime — price=655.44, SMA50=675.93, SMA200=658.41, RSI=37.8, BB_width=0.0821",
      "why_now": "Price (655.44) > SMA-200 (658.41) and SMA-50 slope is positive (0.4200).",
      "exit_plan": "Expiry 2026-05-15 | Profit target: 50% of credit ($85.00/contract) | Max loss: $330.00 | Close if regime shifts adversely"
    }
  }
}
```

Action values: `dry_run`, `submitted`, `rejected`, `skipped_by_llm`, `skipped_existing`, `skipped_liquidation_mode`, `error`, `cycle_timeout`, `daily_drawdown_circuit_breaker`.

---

## Data Sources

| Source | Purpose | What It Provides |
|--------|---------|-----------------|
| **Yahoo Finance** | Regime Detection | 200-day OHLCV for SMAs, RSI-14, Bollinger Bands |
| **Alpaca Market Data** | Snapshots + Options | Real-time prices (batch), Greeks (Delta/Theta/Vega), Bid/Ask |
| **Alpaca Paper API** | Order Execution | Account equity, market clock, paper trading sandbox |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```env
# ── Alpaca ──
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2

# ── Trading ──
TICKERS=SPY,QQQ,AAPL,MSFT,AMZN,IWM
DRY_RUN=false
MAX_RISK_PCT=0.02
MIN_CREDIT_RATIO=0.33
MAX_DELTA=0.20

# ── Intelligence (optional) ──
LLM_ENABLED=false
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=mistral
LLM_EMBEDDING_MODEL=nomic-embed-text
LLM_TEMPERATURE=0.3
TRADE_JOURNAL_DIR=trade_journal
KNOWLEDGE_BASE_DIR=knowledge_base
```

### 3. Run the agent

```bash
# Paper trading mode
python -m trading_agent.agent

# Dry-run (plans written, no orders sent)
python -m trading_agent.agent --dry-run

# Custom .env file
python -m trading_agent.agent --env /path/to/.env
```

### 4. Schedule (5-minute interval)

```bash
# crontab -e
*/5 9-16 * * 1-5 cd /path/to/trading-agent && python -m trading_agent.agent >> logs/cron.log 2>&1
```

The agent has a built-in 270-second timeout guard — if a cycle hangs, the process self-terminates so the next cron run starts cleanly.

### 5. Run tests

```bash
python run_tests.py        # Full suite
pytest tests/ -v           # Core modules only
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TICKERS` | `SPY,QQQ` | Comma-separated underlyings |
| `DRY_RUN` | `true` | Log plans but don't submit orders |
| `MODE` | `dry_run` | `live` or `dry_run` |
| `MAX_RISK_PCT` | `0.02` | Max loss per trade as % of equity |
| `MIN_CREDIT_RATIO` | `0.33` | Minimum credit / spread width |
| `MAX_DELTA` | `0.20` | Max absolute delta of sold strike |
| `SCHEDULE_INTERVAL` | `5m` | Cycle interval (informational — used in startup log) |
| **Risk Guardrails** | | |
| `DAILY_DRAWDOWN_LIMIT` | `0.05` | Kill process if account drops >N% in one day |
| `MAX_BUYING_POWER_PCT` | `0.80` | Enter Liquidation Mode if >N% of BP used |
| `LIQUIDITY_MAX_SPREAD` | `0.05` | Skip tickers where underlying bid/ask spread ≥ $N |
| **Logging** | | |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LOG_DIR` | `logs` | Log file directory |
| `TRADE_PLAN_DIR` | `trade_plans` | Per-ticker plan files + `daily_state.json` |
| **Intelligence Layer** | | |
| `LLM_ENABLED` | `false` | Enable the LLM intelligence layer |
| `LLM_PROVIDER` | `ollama` | `ollama`, `lmstudio`, `openai`, `anthropic` |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM API endpoint |
| `LLM_MODEL` | `mistral` | Reasoning model |
| `LLM_EMBEDDING_MODEL` | `nomic-embed-text` | Embeddings model for RAG |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `TRADE_JOURNAL_DIR` | `trade_journal` | Trade lifecycle logs + signal journal |
| `KNOWLEDGE_BASE_DIR` | `knowledge_base` | RAG vector store |
