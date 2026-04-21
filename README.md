# Autonomous Options Credit Spread Trading Agent

An autonomous trading agent specialized in generating daily income through high-probability, risk-defined options credit spreads. The agent's primary goal is **capital preservation** — it only enters trades where the maximum loss is known and capped, prioritizing time decay (Theta) over directional speculation.

---

## Architecture Overview

The agent runs a **two-stage loop** on every cycle: first it manages existing open positions, then it evaluates new trade opportunities across all configured tickers.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           AGENT CYCLE  (agent.py)                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 1 — Monitor Existing Positions                                │   │
│  │  Position Monitor ──▶ Exit Signal Check ──▶ Order Tracker            │   │
│  │  (fetch spreads)      Stop-Loss 50%         (fills / cancels)        │   │
│  │                       Profit-Target 75%                              │   │
│  │                       Regime Shift / DTE                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 2 — Open New Positions  (per ticker: SPY QQQ AAPL …)         │   │
│  │                                                                      │   │
│  │  I·Perceive ──▶ II·Classify ──▶ III·Plan ──▶ IV·Risk ──▶            │   │
│  │  yfinance 200d   SMA-50/200     Select        8 guardrails           │   │
│  │  Alpaca snap.    RSI-14         strikes        Liquidity check       │   │
│  │  Bid/ask check   Bollinger BB   nearest Fri    Buying power          │   │
│  │  (batch+cached)  3-std MR sig.  in DTE range   Daily DD CB           │   │
│  │                  RS vs SPY/QQQ                                       │   │
│  │                       │                                              │   │
│  │             ┌─────────┘   ← submitted after Phase II, resolved by V  │   │
│  │             │                                                        │   │
│  │             ▼ cycle-scoped background pool  (sentiment_pipeline.py)  │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  TIERED SENTIMENT PIPELINE — no-hallucination, cached       │    │   │
│  │  │                                                             │    │   │
│  │  │  Tier 0 ── Earnings Calendar short-circuit                  │    │   │
│  │  │           earnings within 7d → event_risk=1.0 (no LLM)      │    │   │
│  │  │                                                             │    │   │
│  │  │  Tier 1 ── Content-hash cache (SHA-1 of evidence)           │    │   │
│  │  │           unchanged news → replay VerifiedSentimentReport   │    │   │
│  │  │                                                             │    │   │
│  │  │  Tier 2 ── Full chain                                       │    │   │
│  │  │    NewsAggregator ──▶ FinGPT Specialist ──▶ Verifier        │    │   │
│  │  │    Yahoo / SEC / Fed   source-weighted        claim-by-     │    │   │
│  │  │    Reddit / Twitter    SentimentReport        claim check;  │    │   │
│  │  │                        event_risk score       halluc. flags │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │             │       (Future cancelled if plan/risk reject)          │   │
│  │             ▼                                                        │   │
│  │         ──▶ V·LLM Analysis ──▶ VI·Execute                            │   │
│  │            RAG context          mleg order                           │   │
│  │            Verified sentiment   Alpaca API                           │   │
│  │            Approve/Modify/Skip  Journal entry                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
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

**SMA-50 slope units.** `MarketDataProvider.sma_slope()` returns the 5-day average **dollar change per day** of the SMA — a raw price delta, not a percentage. Every downstream consumer only reads the sign (`> 0` → bullish; `< 0` → bearish). Logs and the LLM analyst prompt annotate with `$/day` so a reader doesn't mistake the magnitude for a percentage.

### Mean Reversion Spreads

When price reaches a **3-standard-deviation Bollinger Band** (statistically extreme), the agent expects reversion toward the mean:

| Band Touch | Expected Move | Strategy |
|------------|--------------|----------|
| Upper 3-std touch | Price extended to upside → expect reversion down | Bear Call Spread above current price |
| Lower 3-std touch | Price extended to downside → expect reversion up | Bull Put Spread below current price |

### Relative Strength Bias

On every cycle, the agent computes each ticker's **5-minute return** via Alpaca bars and compares it to SPY and QQQ. If the ticker outperforms by >0.1% in the 5-min window, the strategy selection is biased toward a Bull Put Spread even in a sideways regime.

---

## 5-Minute Cycle Optimisations

| Optimisation | Where | Detail |
|---|---|---|
| **Parallel price-history fetch** | `market_data.py` | All tickers' 200-day OHLCV fetched concurrently via `ThreadPoolExecutor` before the ticker loop starts |
| **Batch snapshot call** | `market_data.py` | All current prices retrieved in **one** Alpaca API call |
| **TTL-based caches** | `market_data.py` | Historical prices (4 h), stock snapshots (60 s), option chains (3 min), 5-min intraday returns (60 s) |
| **Benchmark dedupe** | `market_data.py` | `get_5min_return("SPY")` / `("QQQ")` calls are collapsed by the 60 s cache — one fetch per cycle |
| **Cycle-scoped sentiment pool** | `sentiment_pipeline.py` | A single-worker `ThreadPoolExecutor` is created inside the cycle's `with pipeline:` block and shut down at the end — no stranded threads across cycles |
| **Concurrent sentiment submit** | `agent.py` | Sentiment work is submitted after Phase II and resolved at Phase V; plan/risk rejection cancels the Future so no LLM call is wasted on a trade that wouldn't execute anyway |
| **Tier-0 earnings short-circuit** | `earnings_calendar.py` | Authoritative `yfinance` earnings calendar — scheduled catalyst within 7d returns `event_risk=1.0` deterministically, skipping FinGPT and the verifier entirely |
| **Tier-1 content-hash cache** | `sentiment_cache.py` | SHA-1 fingerprint over `source \| form_type \| slug \| minute_timestamp` (sorted). Unchanged evidence replays the last `VerifiedSentimentReport`; the cache only ever holds post-verifier results |
| **Hard timeout guard** | `agent.py` | Daemon timer at 270 s: logs `cycle_timeout` and calls `os._exit(1)` so the scheduler cleanly starts the next run |

### Live Quote Refresh at Execution

Option bid/ask can move significantly between Phase III (planning) and Phase VI (order submission). The executor fetches a **fresh, no-cache quote** for the two leg symbols immediately before sending the order and re-validates economics-bearing guardrails (credit ratio, max loss) against the live credit.

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

### Core Intelligence Components

**LLM Client** (`llm_client.py`) — OpenAI-compatible interface supporting Ollama, LM Studio, or the Claude API.

**Trade Journal** (`trade_journal.py`) — Logs the full lifecycle of every trade: entry context, execution status, exit signal, realized P&L, and post-trade lessons.

**Knowledge Base / RAG** (`knowledge_base.py`) — File-based vector store using `nomic-embed-text` embeddings and cosine similarity search. No external vector DB required.

**LLM Analyst** (`llm_analyst.py`) — Returns `approve`, `modify`, or `skip`. Advisory only — cannot override risk manager rejections.

**Fine-Tuning Pipeline** (`fine_tuning.py`) — Exports trade data in Chat JSONL, Alpaca, and DPO preference-pair formats once you have 20+ closed trades.

---

## Multi-Source Sentiment Pipeline

An optional news-intelligence layer wrapped in the `SentimentPipeline` facade (`sentiment_pipeline.py`). It runs **concurrently in a background thread** during every cycle, delivering a `VerifiedSentimentReport` to the LLM Analyst at Phase V with near-zero added latency — and applies three tiers of gating before spending an LLM call so 5-minute cycles don't redundantly invoke local LLMs.

### Why it exists

Pure technical analysis (SMA, RSI, Bollinger Bands) cannot see:
- Earnings announcements within 7 days → binary risk, never sell premium into these
- Fed rate decisions / FOMC meetings → IV spike risk
- SEC 8-K material event filings → authoritative, market-moving facts
- Post-earnings IV crush → favorable window for premium sellers
- Reddit retail positioning → momentum signal, not fundamental

### Tiered Gating

The pipeline never weakens the no-hallucination guarantee — each tier is **stricter** than the next, not looser. The cache only replays reports that already passed the verifier.

| Tier | Gate | When it fires | LLM calls |
|------|------|---------------|-----------|
| **0** | Earnings calendar short-circuit | Authoritative `yfinance` calendar reports a scheduled announcement within `EARNINGS_CALENDAR_LOOKAHEAD_DAYS` (default 7) | None — deterministic `event_risk=1.0`, recommendation `avoid` |
| **1** | Content-hash cache | SHA-1 fingerprint over the normalised news evidence matches a previously-produced `VerifiedSentimentReport` within `SENTIMENT_HASH_CACHE_SIZE` / TTL | None — replays the cached verified report |
| **2** | Full chain | Evidence changed (or no prior report) — runs NewsAggregator → FinGPT specialist → reasoning verifier | FinGPT + verifier; verifier always runs if FinGPT did |

The hash is tolerant of trivial re-orderings and sub-minute timestamp jitter, but flips on any genuine new item, new source, new SEC form type, or new minute-level timestamp.

### Lifecycle

The pipeline owns its own single-worker `ThreadPoolExecutor` and is used as a context manager from `agent._run_cycle`:

```python
with SentimentPipeline.from_config(cfg.intelligence) as pipeline:
    for ticker in tickers:
        fut = pipeline.submit(ticker, regime, price, rsi, iv_rank, strategy)
        ...  # Phase III + IV
        if not (plan.valid and verdict.approved):
            fut.cancel()       # skip LLM work for rejected tickers
        sentiment = fut.result() if fut else None
        ...  # Phase V + VI
# pool.shutdown(wait=True) runs here — all in-flight cache inserts complete
```

Cycle-scoping eliminates the prior regression where an instance-lifetime pool never drained before SIGTERM. Every sentiment Future belongs to exactly one cycle.

### Stage 1 — NewsAggregator (`news_aggregator.py`)

Pulls from every layer of the information stack in parallel, normalises into `NewsItem` objects, deduplicates by slug, and TTL-caches per `(ticker, source)` (default `NEWS_CACHE_TTL=240s`, tuned to hit cache despite 5-minute cycle jitter).

| Source | Authority Weight | Auth Required | What it captures |
|--------|-----------------|---------------|-----------------|
| **SEC EDGAR 8-K / 10-Q** | 1.00 | None (free REST API) | Earnings filings, material events, insider changes |
| **Federal Reserve RSS** | 0.95 | None (public RSS) | FOMC statements, rate decisions, speeches |
| **Yahoo Finance** | 0.70 | None (yfinance) | General financial news, analyst notes |
| **Twitter / X cashtag** | 0.50 | Bearer token (tweepy) | Breaking news, retail momentum |
| **Reddit r/options, r/stocks** | 0.45 | PRAW credentials | Options-specific sentiment, positioning |
| **Reddit r/wallstreetbets** | 0.35 | PRAW credentials | High-noise retail sentiment, meme-stock signal |

The defaults above live in `news_aggregator.DEFAULT_SOURCE_WEIGHTS`; operators can override any subset without code changes by setting `NEWS_SOURCE_WEIGHTS_JSON='{"yahoo": 0.6, "sec_edgar": 1.0}'`. Each source degrades gracefully — a missing credential or API timeout skips that source without blocking the pipeline.

### Stage 2 — FinGPT Specialist (`fingpt_analyser.py`)

A locally-hosted finance LLM (via Ollama) analyzes the aggregated news with source authority context baked into the prompt. Items are grouped by source and sorted by weight so the model knows a SEC 8-K outranks a Reddit post before scoring.

**Output — `SentimentReport`:**

| Field | Range | Meaning |
|-------|-------|---------|
| `sentiment_score` | −1.0 to +1.0 | Bearish to bullish |
| `event_risk` | 0.0 to 1.0 | Binary catalyst risk (>0.7 → avoid selling premium) |
| `confidence` | 0.0 to 1.0 | Model's self-assessed certainty |
| `recommendation` | favorable / neutral / caution / avoid | Action signal for the LLM Analyst |
| `key_themes` | list | e.g. `["earnings_beat", "fed_hawkish"]` |
| `reasoning` | string | Chain-of-thought explanation |

### Stage 3 — Reasoning Verifier (`sentiment_verifier.py`)

A stronger reasoning model independently cross-checks every claim in FinGPT's output against the raw evidence. This is the **anti-hallucination gate** — FinGPT is fast and finance-tuned but can state confident facts not present in the headlines.

**What the verifier checks:**
- Maps each claim in FinGPT's reasoning to a specific evidence item
- Flags any claim **not supported** by the evidence (hallucination)
- Adjusts confidence scores based on evidence quality
- Confirms or revises the recommendation if claims are materially unsupported

**Output — `VerifiedSentimentReport`:**

| Field | What it adds |
|-------|-------------|
| `verified_sentiment_score` | Adjusted score after evidence cross-check |
| `verified_event_risk` | Verified event risk (only flags real catalysts found in evidence) |
| `verified_recommendation` | Final recommendation after verification |
| `hallucination_flags` | Claims found in FinGPT reasoning not supported by any evidence item |
| `agreement_score` | 0.0 = full disagreement with FinGPT, 1.0 = fully supported |
| `evidence_mapping` | Per-claim: `supported` / `partially_supported` / `unsupported` |

Supported verifier backends:

| Provider | Model examples | Use case |
|----------|---------------|---------|
| `ollama` (local) | `qwq:32b`, `deepseek-r1:32b` | Privacy-first; M5 Pro Max runs these comfortably alongside FinGPT |
| `anthropic` (cloud) | `claude-sonnet-4-6`, `claude-opus-4-7` | Best reasoning quality; requires API key |

**Fallback:** If the verifier is unavailable (model not loaded, API error), the original `SentimentReport` is wrapped and passed through unchanged — the pipeline never blocks.

### Sentiment in the LLM Analyst Prompt

The `VerifiedSentimentReport` is injected into the Phase V analysis prompt as a structured section, giving the LLM Analyst:
- Verified sentiment score and event risk
- Key themes and top headlines
- Any hallucination flags discovered during verification
- Agreement score showing verifier confidence in FinGPT's output
- A `⛔ HIGH EVENT RISK` warning when `event_risk > 0.7`

### Setting Up the Sentiment Pipeline

```bash
# ── Minimum — free sources only, no credentials needed ─────────────
FINGPT_ENABLED=true
FINGPT_MODEL=qwen2.5-trading        # already installed if using Ollama
FINGPT_TEMPERATURE=0.1              # deterministic scoring
FINGPT_MAX_TOKENS=512
FINGPT_TIMEOUT=45
NEWS_SOURCES=yahoo,sec_edgar,fed_rss

# ── Add Reddit (r/wsb, r/stocks, r/options, r/investing auto-enabled)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
# Register at: https://www.reddit.com/prefs/apps

# ── Add Twitter / X ────────────────────────────────────────────────
TWITTER_BEARER_TOKEN=your_bearer_token
# Requires Twitter API Basic or higher

# ── Add reasoning verifier — local (recommended on M-series Macs) ──
VERIFIER_ENABLED=true
VERIFIER_PROVIDER=ollama
VERIFIER_MODEL=qwq:32b              # pull with: ollama pull qwq:32b
VERIFIER_TEMPERATURE=0.15           # low but non-zero — reasoning models benefit
VERIFIER_MAX_TOKENS=2048
VERIFIER_TIMEOUT=90                 # reasoning is slower

# ── Add reasoning verifier — cloud (best quality) ──────────────────
VERIFIER_ENABLED=true
VERIFIER_PROVIDER=anthropic
VERIFIER_MODEL=claude-sonnet-4-6
VERIFIER_API_KEY=sk-ant-...

# ── Tier-0 / Tier-1 efficiency gates ───────────────────────────────
EARNINGS_CALENDAR_ENABLED=true
EARNINGS_CALENDAR_LOOKAHEAD_DAYS=7
EARNINGS_CALENDAR_REFRESH_HOURS=12
SENTIMENT_HASH_CACHE_SIZE=32

# ── Optional: override default source authority weights ────────────
NEWS_SOURCE_WEIGHTS_JSON={"yahoo": 0.65, "sec_edgar": 1.0}

# Optional pip installs for Reddit / Twitter / Anthropic verifier:
# pip install praw>=7.7.0 tweepy>=4.14.0 anthropic>=0.40.0
```

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

**Safety invariant:** The sentiment pipeline is advisory only. FinGPT and the verifier cannot approve a trade the risk manager has already rejected, and can only tighten constraints — never loosen them.

### Daily Drawdown Circuit Breaker

If current equity falls more than `DAILY_DRAWDOWN_LIMIT` (default 5%) from the day's opening value, the agent logs a `daily_drawdown_circuit_breaker` event and calls `os._exit(1)`.

### Liquidation Mode

If available buying power exceeds `MAX_BUYING_POWER_PCT` (default 80%), Stage 2 (new trade opening) is skipped entirely. Stage 1 (position monitoring) continues normally.

### Capital Retainment Guards (defense_first)

| Guard | Trigger | Effect |
|-------|---------|--------|
| **Macro Guard** | `price < SMA-200` when regime would generate a Bull Put Spread | Skips the entry |
| **High-IV Block** | Realized-volatility IV rank > 95th percentile | Skips ALL new entries |

### Position Exit Debouncing

Non-immediate exit signals require **3 consecutive cycles** (≈ 15 minutes) of the same signal before closing. Signals that bypass debounce and close immediately:

| Signal | Trigger |
|--------|---------|
| `HARD_STOP` | Spread has lost ≥ 3× the initial credit |
| `STRIKE_PROXIMITY` | Underlying within 1% of any short strike |
| `DTE_SAFETY` | Thursday after 15:30 ET and expiry is next day |

---

## Project Structure

```
trading-agent/
├── .env                              # API keys and config (not committed)
├── requirements.txt
├── README.md
├── setup_intelligence.sh             # Ollama setup helper
├── run_tests.py
├── architecture_diagram.html
│
├── trading_agent/
│   ├── agent.py                      # Orchestrator: two-stage cycle, timeout guard, sentiment pipeline
│   ├── config.py                     # AppConfig + IntelligenceConfig (LLM + FinGPT + verifier + cache + calendar)
│   ├── ports.py                      # Hexagonal protocols: MarketDataPort, BrokerPort, SentimentReadout
│   ├── market_profile.py             # MarketProfile (timezone, session bounds, trading-day oracle)
│   ├── logger_setup.py
│   │
│   │   # ── Core Phases ──
│   ├── market_data.py                # Phase I   — yfinance + Alpaca (TTL cache, parallel)
│   ├── regime.py                     # Phase II  — SMA / RSI / Bollinger regime classifier
│   ├── strategy.py                   # Phase III — strike selection, nearest-Friday DTE
│   ├── risk_manager.py               # Phase IV  — 8-guardrail validator
│   ├── executor.py                   # Phase VI  — mleg order execution + HTML report
│   ├── trade_plan_report.py
│   │
│   │   # ── Position Management ──
│   ├── position_monitor.py           # Stage 1 — monitor & close open spreads
│   ├── order_tracker.py              # Stage 1 — fill tracking
│   │
│   │   # ── Core Intelligence Layer ──
│   ├── journal_kb.py                 # Always-on signal logger (JSONL + Markdown)
│   ├── trade_journal.py              # Full-lifecycle trade logging (TradeEntry)
│   ├── knowledge_base.py             # File-based RAG vector store
│   ├── llm_client.py                 # OpenAI-compatible LLM client + make_llm_client(role) factory
│   ├── llm_analyst.py                # Pre/post trade LLM analysis — consumes SentimentReadout
│   ├── fine_tuning.py                # Training data export (JSONL / Alpaca / DPO)
│   │
│   │   # ── Multi-Source Sentiment Pipeline ──
│   ├── sentiment_pipeline.py         # SentimentPipeline facade — Tier-0/1/2 gating, cycle-scoped pool
│   ├── earnings_calendar.py          # Tier-0 — yfinance-backed authoritative event_risk short-circuit
│   ├── sentiment_cache.py            # Tier-1 — SHA-1 content-hash gate (TTL + LRU)
│   ├── news_aggregator.py            # Tier-2 — NewsItem + NewsAggregator (Yahoo/SEC/Fed/Reddit/Twitter)
│   ├── fingpt_analyser.py            # Tier-2 — FinGPT specialist (SentimentReport)
│   └── sentiment_verifier.py         # Tier-2 — Reasoning verifier (VerifiedSentimentReport)
│
├── trade_journal/                    # Trade lifecycle logs + signal journal (auto-created)
│   ├── trades/
│   ├── index.json
│   ├── stats.json
│   ├── signals.jsonl                 # Always-on signal log (every cycle, LLM-independent)
│   └── signals.md
│
├── knowledge_base/                   # RAG vector store (auto-created, LLM layer only)
├── trade_plans/                      # Per-ticker persistent trade plan files
│   ├── trade_plan_{TICKER}.json
│   ├── trade_plan_{TICKER}.html
│   └── daily_state.json
└── logs/
```

---

## Data Sources

| Source | Purpose | Auth |
|--------|---------|------|
| **Yahoo Finance** | Regime detection (SMA/RSI/BB), backtest history | None |
| **Alpaca Market Data** | Real-time snapshots, option chains, Greeks | API key |
| **Alpaca Paper API** | Order execution, account equity, market clock | API key |
| **SEC EDGAR** | 8-K material events, 10-Q filings (sentiment pipeline) | None |
| **Federal Reserve RSS** | FOMC statements, rate decisions (sentiment pipeline) | None |
| **Reddit** | Retail sentiment — r/wsb, r/stocks, r/options (sentiment pipeline) | PRAW credentials |
| **Twitter / X** | Breaking news, cashtag stream (sentiment pipeline) | Bearer token |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Optional — install only what you need for the sentiment pipeline:
pip install praw>=7.7.0        # Reddit
pip install tweepy>=4.14.0     # Twitter / X
pip install anthropic>=0.40.0  # Claude verifier
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

# ── Core Intelligence (optional) ──
LLM_ENABLED=false
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5-trading
LLM_EMBEDDING_MODEL=nomic-embed-text
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2048
LLM_TIMEOUT=60
TRADE_JOURNAL_DIR=trade_journal
KNOWLEDGE_BASE_DIR=knowledge_base

# ── FinGPT Specialist (optional — stage 2/3) ──
FINGPT_ENABLED=false
FINGPT_MODEL=qwen2.5-trading
FINGPT_NEWS_LIMIT=10
FINGPT_CACHE_TTL=300
FINGPT_TEMPERATURE=0.1            # deterministic scoring
FINGPT_MAX_TOKENS=512
FINGPT_TIMEOUT=45

# ── News Aggregator (stage 1/3) ──
NEWS_SOURCES=yahoo,sec_edgar,fed_rss
NEWS_LOOKBACK_HOURS=24
NEWS_MAX_ITEMS_PER_SOURCE=20
NEWS_CACHE_TTL=240                # 4 min — survives 5-min cycle jitter
NEWS_SOURCE_WEIGHTS_JSON=         # optional: JSON overrides

# ── Reddit (optional — auto-adds wsb/stocks/options/investing) ──
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=TradingAgent/1.0

# ── Twitter / X (optional) ──
TWITTER_BEARER_TOKEN=

# ── Reasoning Verifier (optional — stage 3/3) ──
VERIFIER_ENABLED=false
VERIFIER_PROVIDER=ollama          # "ollama" or "anthropic"
VERIFIER_MODEL=qwq:32b
VERIFIER_API_KEY=                 # Anthropic key if VERIFIER_PROVIDER=anthropic
VERIFIER_TEMPERATURE=0.15
VERIFIER_MAX_TOKENS=2048
VERIFIER_TIMEOUT=90

# ── Pipeline Efficiency Gates ──
EARNINGS_CALENDAR_ENABLED=true
EARNINGS_CALENDAR_LOOKAHEAD_DAYS=7
EARNINGS_CALENDAR_REFRESH_HOURS=12
SENTIMENT_HASH_CACHE_SIZE=32
```

### 3. Run the agent

```bash
# Paper trading
python -m trading_agent.agent

# Dry-run (no orders sent)
python -m trading_agent.agent --dry-run

# Custom .env
python -m trading_agent.agent --env /path/to/.env
```

### 4. Schedule (5-minute interval)

```bash
# crontab -e
*/5 9-16 * * 1-5 cd /path/to/trading-agent && python -m trading_agent.agent >> logs/cron.log 2>&1
```

#### After-hours automatic shutdown

| Condition | Action |
|-----------|--------|
| Before 9:25 AM ET (Mon–Fri) | Exit 0 — too early |
| After 4:05 PM ET (Mon–Fri) | Exit 0 — market closed |
| Saturday / Sunday | Exit 0 — weekend |

Override for after-hours paper testing:
```bash
FORCE_MARKET_OPEN=true python -m trading_agent.agent
```

### 5. Run tests

```bash
python run_tests.py
pytest tests/ -v
```

---

## Configuration Reference

### Core Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `TICKERS` | `SPY,QQQ` | Comma-separated underlyings |
| `DRY_RUN` | `true` | Log plans but don't submit orders |
| `MODE` | `dry_run` | `live` or `dry_run` |
| `MAX_RISK_PCT` | `0.02` | Max loss per trade as % of equity |
| `MIN_CREDIT_RATIO` | `0.33` | Minimum credit / spread width |
| `MAX_DELTA` | `0.20` | Max absolute delta of sold strike |
| `DAILY_DRAWDOWN_LIMIT` | `0.05` | Kill process if account drops >N% in one day |
| `MAX_BUYING_POWER_PCT` | `0.80` | Enter Liquidation Mode if >N% of BP used |
| `LIQUIDITY_MAX_SPREAD` | `0.05` | Skip tickers where underlying bid/ask ≥ $N |
| `FORCE_MARKET_OPEN` | `false` | Bypass market-hours check (paper testing) |

### Core Intelligence Layer (Analyst)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENABLED` | `false` | Enable the LLM intelligence layer |
| `LLM_PROVIDER` | `ollama` | `ollama`, `lmstudio`, `openai`, `anthropic` |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM API endpoint |
| `LLM_MODEL` | `mistral` | Primary reasoning model |
| `LLM_EMBEDDING_MODEL` | `nomic-embed-text` | Embeddings model for RAG |
| `LLM_TEMPERATURE` | `0.3` | Analyst sampling temperature |
| `LLM_MAX_TOKENS` | `2048` | Analyst response cap |
| `LLM_TIMEOUT` | `60` | Analyst HTTP timeout (s) |
| `TRADE_JOURNAL_DIR` | `trade_journal` | Trade lifecycle logs |
| `KNOWLEDGE_BASE_DIR` | `knowledge_base` | RAG vector store |

All three LLM callers (analyst, FinGPT, verifier) share the same `make_llm_client(role, cfg)` factory so their parameters live in one place — there is no more per-module hard-coded `LLMConfig`.

### FinGPT Specialist (Sentiment Stage 2/3)

| Variable | Default | Description |
|----------|---------|-------------|
| `FINGPT_ENABLED` | `false` | Enable sentiment pipeline |
| `FINGPT_MODEL` | `qwen2.5-trading` | Ollama model for FinGPT analysis |
| `FINGPT_NEWS_LIMIT` | `10` | Max headlines from yfinance fallback |
| `FINGPT_CACHE_TTL` | `300` | FinGPT in-process cache TTL (s) |
| `FINGPT_TEMPERATURE` | `0.1` | Keep deterministic — sentiment scoring should be repeatable |
| `FINGPT_MAX_TOKENS` | `512` | Short JSON response cap |
| `FINGPT_TIMEOUT` | `45` | HTTP timeout (s) — kept tight so a stuck model won't blow the 5-min budget |

### News Aggregator (Sentiment Stage 1/3)

| Variable | Default | Description |
|----------|---------|-------------|
| `NEWS_SOURCES` | `yahoo,sec_edgar,fed_rss` | Comma-separated source keys |
| `NEWS_LOOKBACK_HOURS` | `24` | How far back to fetch news |
| `NEWS_MAX_ITEMS_PER_SOURCE` | `20` | Max items fetched per source per cycle |
| `NEWS_CACHE_TTL` | `240` | Per-`(ticker, source)` cache TTL (s) — 4 min survives 5-min cycle jitter |
| `NEWS_SOURCE_WEIGHTS_JSON` | _(empty)_ | JSON object overriding `DEFAULT_SOURCE_WEIGHTS` (e.g. `{"yahoo": 0.6}`) |

### News Source Credentials

| Variable | Default | Description |
|----------|---------|-------------|
| `REDDIT_CLIENT_ID` | _(empty)_ | PRAW client ID — enables all Reddit sources |
| `REDDIT_CLIENT_SECRET` | _(empty)_ | PRAW client secret |
| `REDDIT_USER_AGENT` | `TradingAgent/1.0` | PRAW user agent string |
| `TWITTER_BEARER_TOKEN` | _(empty)_ | Twitter API v2 Bearer token — enables Twitter source |

### Reasoning Verifier (Sentiment Stage 3/3)

| Variable | Default | Description |
|----------|---------|-------------|
| `VERIFIER_ENABLED` | `false` | Enable reasoning-model verification |
| `VERIFIER_PROVIDER` | `ollama` | `ollama` (local) or `anthropic` (cloud) |
| `VERIFIER_MODEL` | `qwq:32b` | Verifier model (`qwq:32b`, `deepseek-r1:32b`, `claude-sonnet-4-6`) |
| `VERIFIER_API_KEY` | _(empty)_ | Anthropic API key when `VERIFIER_PROVIDER=anthropic` |
| `VERIFIER_TEMPERATURE` | `0.15` | Low but non-zero — reasoning models benefit |
| `VERIFIER_MAX_TOKENS` | `2048` | Response cap |
| `VERIFIER_TIMEOUT` | `90` | HTTP timeout (s) — reasoning is slower than sentiment scoring |

### Pipeline Efficiency Gates

| Variable | Default | Description |
|----------|---------|-------------|
| `EARNINGS_CALENDAR_ENABLED` | `true` | Tier-0 short-circuit: set `event_risk=1.0` when yfinance reports a scheduled earnings date within the lookahead |
| `EARNINGS_CALENDAR_LOOKAHEAD_DAYS` | `7` | Window for Tier-0 firing |
| `EARNINGS_CALENDAR_REFRESH_HOURS` | `12` | Per-ticker cache freshness — refresh twice per trading day |
| `SENTIMENT_HASH_CACHE_SIZE` | `32` | Tier-1 LRU cap. Cache TTL auto-scales to `max(NEWS_CACHE_TTL, FINGPT_CACHE_TTL)` so evidence-level and sentiment-level reuse windows align |

---

## Agent Performance Dashboard

`visualize_logs.py` parses the signal journal and per-ticker trade-plan files to generate a self-contained interactive HTML report.

```bash
python visualize_logs.py
python visualize_logs.py --date 2026-04-03 --tickers SPY QQQ
python visualize_logs.py --all-dates --output reports/full_history.html
```

### Visual components

| # | Chart | Description |
|---|-------|-------------|
| 1 | **Heartbeat Timeline** | One dot per 5-min cycle, colour-coded by outcome |
| 2 | **Safety Buffer Chart** | Underlying price with short-strike danger lines for open positions |
| 3 | **Logic Distribution** | Pie: Active Trade / SMA Filter / RSI Filter / High-IV / Risk Rejected / Defense First / Error |

---

## Streamlit Live Dashboard

```bash
pip install "streamlit>=1.42.0" "plotly>=6.0.0"
streamlit run trading_agent/streamlit/app.py
```

### Tabs

| Tab | Features |
|-----|---------|
| **📡 Live Monitoring** | Agent Start/Stop/Dry Run controls · cycle PID · equity · P&L · regime badge · open positions · equity curve · 8-guardrail status · market status · agent log · journal expander · auto-refresh 30 s |
| **📊 Backtesting** | Date range · multi-ticker · timeframe (1Day/5Min) · simulated P&L · metric cards · per-regime bar chart · equity + drawdown charts · trade log · CSV/JSON/Journal export |
| **🤖 LLM Extension** | Chat with local Ollama model (RAG over journal) · Optimize Strategy → one-click `.env` update |

---

## Signal Journal Format (`signals.jsonl`)

Every trade attempt is logged as a single JSON line including `fingpt_sentiment`, `fingpt_event_risk`, `fingpt_recommendation`, `fingpt_agreement`, `fingpt_hallucination_flags`, and `fingpt_verified_by` when the sentiment pipeline is active.

Action values: `dry_run`, `submitted`, `rejected`, `skipped_by_llm`, `skipped_existing`, `skipped_liquidation_mode`, `skipped`, `error`, `cycle_timeout`, `daily_drawdown_circuit_breaker`.
