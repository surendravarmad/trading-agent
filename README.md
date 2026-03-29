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
│  │  yfinance 200d   SMA-50/200     Select        6 guardrails       │   │
│  │  Alpaca snap.    RSI-14         strikes        Max loss 2%        │   │
│  │                  Bollinger BB   45 DTE target  Paper only         │   │
│  │                                                                  │   │
│  │              ──▶ V·LLM Analysis ──▶ VI·Execute                  │   │
│  │                 RAG context         mleg order                   │   │
│  │                 Approve/Skip        Alpaca API                   │   │
│  │                 Confidence          Journal entry                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

> 📊 An interactive HTML architecture diagram is included: open `architecture_diagram.html` in any browser — click any component for detail panels.

---

## Market Regime → Strategy Matrix

The agent classifies the current market regime before selecting any strategy. This prevents selling a bearish spread during a bull rally.

| Regime | Detection Rule | Strategy |
|--------|---------------|----------|
| **Bullish** | Price > SMA-200 AND SMA-50 slope > 0 | Bull Put Spread |
| **Bearish** | Price < SMA-200 AND SMA-50 slope < 0 | Bear Call Spread |
| **Sideways** | Between SMAs or narrow Bollinger Bands | Iron Condor |

---

## Intelligence Layer

The agent includes an optional LLM-powered intelligence layer that learns from every trade and improves decisions over time.

```
Trade Executes ──▶ Journal Entry Opened ──▶ LLM Post-Trade Analysis
      ──▶ Lessons → Knowledge Base ──▶ Better Decisions Next Cycle
      ──▶ (after 20+ trades) Fine-Tune Local Model
```

### Components

**LLM Client** (`llm_client.py`) — OpenAI-compatible interface that works with Ollama, LM Studio, or the Claude API. Defaults to `mistral` on a local Ollama instance. Swappable via `.env` with zero code changes.

**Trade Journal** (`trade_journal.py`) — Logs the full lifecycle of every trade: entry context (regime, indicators, Greeks, LLM reasoning), execution status, exit signal, realized P&L, and post-trade lessons. Every field needed for fine-tuning is captured.

**Knowledge Base / RAG** (`knowledge_base.py`) — File-based vector store using `nomic-embed-text` embeddings (768-dimensional) and cosine similarity search. Before each trade, the LLM queries the KB for similar past trades and relevant lessons. No external vector DB required.

**LLM Analyst** (`llm_analyst.py`) — The decision layer. Receives the full trade context, retrieves RAG results, and returns one of three actions:
- `approve` — proceed with the trade as planned
- `modify` — adjust strikes or sizing (within risk limits)
- `skip` — pass on this trade with reasoning

**Safety invariant:** The LLM analyst is advisory only. It can never approve a trade that the risk manager has already rejected, and it can only tighten constraints — never loosen them.

**Fine-Tuning Pipeline** (`fine_tuning.py`) — Exports accumulated trade data in three formats once you have 20+ closed trades:
- Chat JSONL (Ollama / LM Studio fine-tuning)
- Alpaca instruction format (LoRA / QLoRA)
- DPO preference pairs (win trades as "chosen", loss trades as "rejected")

### Setting Up Ollama (Recommended)

```bash
# Run the setup script
chmod +x setup_intelligence.sh
./setup_intelligence.sh

# Or manually:
brew install ollama          # macOS
ollama pull mistral          # reasoning model
ollama pull nomic-embed-text # embeddings model
ollama serve                 # start the server
```

Then enable in `.env`:
```
LLM_ENABLED=true
LLM_PROVIDER=ollama
LLM_MODEL=mistral
```

Other recommended models: `llama3`, `deepseek-r1:7b` (stronger reasoning), `qwen2.5:7b`.

---

## Risk Management Guardrails

Every trade must pass **all six checks** before execution:

| # | Check | Rule |
|---|-------|------|
| 1 | **Plan Validity** | The strategy planner found valid strikes and contracts |
| 2 | **Credit-to-Width Ratio** | Credit collected ≥ 1/3 of spread width (e.g. ≥ $1.65 on a $5 spread) |
| 3 | **Sold Delta** | ≤ 0.20 — high probability of expiring worthless (~80%+) |
| 4 | **Max Loss** | ≤ 2% of account equity per trade |
| 5 | **Account Type** | Must be `paper` — safety assertion against live trading |
| 6 | **Market Hours** | Market must currently be open |

**Max Loss Formula:**
```
Max Loss = (Spread Width − Credit Collected) × 100
```

---

## Project Structure

```
Trading Agent/
├── .env                          # API keys and config (not committed)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── setup_intelligence.sh         # Ollama setup helper script
├── run_tests.py                  # Full test suite runner (80 tests)
├── architecture_diagram.html     # Interactive architecture diagram
├── architecture_diagram.jsx      # React source for the diagram
│
├── trading_agent/
│   ├── __init__.py
│   ├── config.py                 # Config loader (AppConfig, IntelligenceConfig)
│   ├── logger_setup.py           # Logging configuration
│   │
│   │   # ── Core Agent Phases ──
│   ├── market_data.py            # Phase I   — yfinance + Alpaca data fetch
│   ├── regime.py                 # Phase II  — Regime classifier (SMA/RSI/BB)
│   ├── strategy.py               # Phase III — Strike selection & spread planning
│   ├── risk_manager.py           # Phase IV  — 6-guardrail risk validator
│   ├── executor.py               # Phase VI  — mleg order execution
│   ├── agent.py                  # Orchestrator: two-stage cycle + CLI entry
│   │
│   │   # ── Position Management ──
│   ├── position_monitor.py       # Stage 1 — Monitor & close open spreads
│   ├── order_tracker.py          # Stage 1 — Fill tracking & stale order cleanup
│   │
│   │   # ── Intelligence Layer ──
│   ├── llm_client.py             # OpenAI-compatible LLM client (Ollama/API)
│   ├── trade_journal.py          # Full-lifecycle trade logging (TradeEntry)
│   ├── knowledge_base.py         # File-based RAG vector store
│   ├── llm_analyst.py            # Pre/post trade LLM analysis & decisions
│   └── fine_tuning.py            # Training data export (JSONL / Alpaca / DPO)
│
├── trade_journal/                # Per-trade JSON lifecycle logs (auto-created)
├── knowledge_base/               # RAG vector store files (auto-created)
├── logs/                         # Runtime log files
└── trade_plans/                  # JSON trade plan audit trail
```

---

## Data Sources

| Source | Purpose | What It Provides |
|--------|---------|-----------------|
| **Yahoo Finance** | Regime Detection | 200-day historical OHLCV for SMAs, RSI-14, Bollinger Bands |
| **Alpaca Market Data** | Option Snapshots | Real-time Greeks (Delta, Theta, Vega), Bid/Ask spreads |
| **Alpaca Paper API** | Order Execution | Account equity, market clock, paper trading sandbox |

---

## Setup

### 1. Install dependencies

```bash
cd "Trading Agent"
pip install -r requirements.txt
```

### 2. Configure environment

Edit the `.env` file with your Alpaca Paper Trading credentials:

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
# Live paper trading mode
python -m trading_agent.agent

# Dry-run mode (calculates trades, writes plans, no orders sent)
python -m trading_agent.agent --dry-run

# Custom .env file
python -m trading_agent.agent --env /path/to/.env
```

### 4. Run tests

```bash
python run_tests.py        # Full suite (80 tests, 16 test classes)
pytest tests/ -v           # Core module tests only
```

---

## Trade Plan Audit Trail

Every trade cycle writes a JSON file to `trade_plans/` containing the full reasoning log:

```json
{
  "trade_plan": {
    "ticker": "SPY",
    "strategy": "Bull Put Spread",
    "regime": "bullish",
    "legs": [
      { "symbol": "SPY240119P00480000", "side": "sell", "ratio_qty": 1 },
      { "symbol": "SPY240119P00475000", "side": "buy",  "ratio_qty": 1 }
    ],
    "spread_width": 5.0,
    "net_credit": 1.70,
    "max_loss": 330.0,
    "credit_to_width_ratio": 0.34,
    "sold_delta": -0.15
  },
  "risk_verdict": {
    "approved": true,
    "checks_passed": ["plan_valid", "credit_ratio", "delta", "max_loss", "paper_account", "market_open"],
    "checks_failed": []
  },
  "llm_decision": {
    "action": "approve",
    "confidence": 0.82,
    "reasoning": "Similar trades in bullish SPY regimes with RSI < 60 have shown 78% win rate.",
    "similar_trades_found": 5
  },
  "mode": "live"
}
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TICKERS` | `SPY,QQQ` | Comma-separated underlyings to trade |
| `DRY_RUN` | `true` | If true, log plans but don't submit orders |
| `MAX_RISK_PCT` | `0.02` | Max loss per trade as % of account equity |
| `MIN_CREDIT_RATIO` | `0.33` | Minimum credit / spread width |
| `MAX_DELTA` | `0.20` | Max absolute delta of sold strike |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LLM_ENABLED` | `false` | Enable the intelligence layer |
| `LLM_PROVIDER` | `ollama` | LLM provider: `ollama`, `lmstudio`, `openai` |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM API base URL |
| `LLM_MODEL` | `mistral` | Model name for reasoning |
| `LLM_EMBEDDING_MODEL` | `nomic-embed-text` | Model name for RAG embeddings |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature (lower = more deterministic) |
| `TRADE_JOURNAL_DIR` | `trade_journal` | Directory for trade lifecycle logs |
| `KNOWLEDGE_BASE_DIR` | `knowledge_base` | Directory for RAG vector store |
