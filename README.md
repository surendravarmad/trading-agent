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
│  │                  Lead-z vs anchor                                    │   │
│  │                  VIX-z gate (^VIX)                                   │   │
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
| 2 | **VIX inhibit** | `vix_zscore > +2σ` AND regime ∈ {Bullish, Sideways} | Bear Call Spread (demoted) |
| 3 | **Bullish + Lead z** | Bullish AND `leadership_zscore > +1.5σ` vs anchor | Bull Put Spread (leadership bias) |
| 4 | **Sideways + Lead z** | Sideways AND `leadership_zscore > +1.5σ` vs anchor | Bull Put Spread (leadership bias) |
| 5 | **Bullish** | Price > SMA-200 AND SMA-50 slope > 0 | Bull Put Spread |
| 6 | **Bearish** | Price < SMA-200 AND SMA-50 slope < 0 | Bear Call Spread |
| 7 | **Sideways** | Between SMAs or narrow Bollinger Bands | Iron Condor |

**SMA-50 slope units.** `MarketDataProvider.sma_slope()` returns the 5-day average **dollar change per day** of the SMA — a raw price delta, not a percentage. Every downstream consumer only reads the sign (`> 0` → bullish; `< 0` → bearish). Logs and the LLM analyst prompt annotate with `$/day` so a reader doesn't mistake the magnitude for a percentage.

### Mean Reversion Spreads

When price reaches a **3-standard-deviation Bollinger Band** (statistically extreme), the agent expects reversion toward the mean:

| Band Touch | Expected Move | Strategy |
|------------|--------------|----------|
| Upper 3-std touch | Price extended to upside → expect reversion down | Bear Call Spread above current price |
| Lower 3-std touch | Price extended to downside → expect reversion up | Bull Put Spread below current price |

### Z-Scored Leadership Bias (ETF macro patch — Items 1 & 2)

The legacy "5-min return ticker vs SPY+QQQ, threshold 0.1%" was useless on an ETF-only deployment: SPY-vs-SPY is degenerately zero and a flat threshold ignores per-ticker volatility. The new path:

1. **Per-ticker leadership anchor** — `leadership_anchors` in `trading_rules.yaml` (under the `regime:` section) maps each ticker to a sibling benchmark (e.g. `SPY → QQQ`, `QQQ → SPY`, sector ETFs → SPY, `IWM → SPY`). To extend coverage, add a row there; no code change required.
2. **Z-scored differential** — `MarketDataProvider.get_leadership_zscore(ticker, anchor)` returns `(raw_diff, z)` where `z` is the latest 5-min return differential normalised against its own rolling intraday distribution (population stdev over the last ~20 bars, with the first 2 open bars dropped to suppress the open-print spike).
3. **Bias trigger** — `StrategyPlanner.RS_ZSCORE_THRESHOLD = 1.5`. When `leadership_zscore > 1.5σ` and the regime is Bullish or Sideways, the planner picks a **Bull Put Spread** instead of the default mapping. 1.5σ ≈ 13th-percentile two-tailed move — strong enough to filter noise, loose enough to actually fire.

### Inter-Market Gate — VIX (ETF macro patch — Item 3)

When inter-market fear spikes, short-DTE bullish premium is the worst position to hold. The classifier samples ^VIX (via `yfinance` — Alpaca doesn't carry the index) and Z-scores its 5-min level change. If `vix_zscore > +2.0σ`, `analysis.inter_market_inhibit_bullish = True` and the planner **demotes** Bull Put / Iron Condor to a **Bear Call Spread** for that cycle. Mean-reversion trades bypass the gate (the band touch already encodes the volatility condition).

### Adaptive Spread Width

Spread width is no longer a flat `$5`. The planner infers each chain's strike grid step (modal gap between sorted strikes — `$1` for IWM-style names, `$5` for SPY/QQQ far-dated wings) and picks:

```
width = max(SPREAD_WIDTH_FLOOR, 3 × strike_grid_step, 0.025 × spot)
        snapped UP to the strike grid
```

This produces a `$5` spread on a `$80` ticker (floor wins) and a `$15-20` spread on SPY/QQQ at `$700` (the spot-percentage term wins) — the wider strike distance is what lets the credit clear the `1/3 × width` `MIN_CREDIT_RATIO` gate at `0.20`-delta short / 30-DTE. The legacy `SPREAD_WIDTH = $5` constant is now a hard floor, never a target. Applied to all four spread types (Bull Put, Bear Call, both Iron Condor wings).

### DTE Targeting

Theta capture is concentrated in the **25-40 DTE** band, so the planner targets `TARGET_DTE = 35` (was 45) and accepts any expiration in `DTE_RANGE = (28, 45)` (was `(35, 50)`). When several adjacent Fridays fall in range, the highest-DTE one is preferred (more theta runway, less gamma risk near expiry).

---

## 5-Minute Cycle Optimisations

| Optimisation | Where | Detail |
|---|---|---|
| **Parallel price-history fetch** | `market_data.py` | All tickers' 200-day OHLCV fetched concurrently via `ThreadPoolExecutor` before the ticker loop starts |
| **Batch snapshot call** | `market_data.py` | All current prices retrieved in **one** Alpaca API call |
| **TTL-based caches** | `market_data.py` | Historical prices (4 h), stock snapshots (60 s), option chains (3 min), 5-min intraday returns (60 s) |
| **Benchmark dedupe** | `market_data.py` | `get_5min_return("SPY")` / `("QQQ")` calls are collapsed by the 60 s cache — one fetch per cycle |
| **Leadership series cache** | `market_data.py` | `get_5min_return_series` caches the rolling 21-bar window per ticker for 60 s so the Z-scored leadership signal reuses one fetch across all anchors |
| **VIX z-score cache** | `market_data.py` | Single global yfinance fetch per cycle (60 s TTL) for the inter-market gate |
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
| 7 | **Underlying Liquidity** | Stock bid/ask spread < `max(LIQUIDITY_MAX_SPREAD, LIQUIDITY_BPS_OF_MID × mid)` — scales with spot so high-priced names (SPY, GOOG) aren't false-rejected. If `(spread / mid) > STALE_SPREAD_PCT` the quote is treated as stale and soft-passed with a `WARNING` instead of hard-failing. |
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
├── visualize_logs.py                 # Root-level entry point (delegates to reporting/)
├── architecture_diagram.html
│
├── trading_agent/
│   ├── __main__.py                   # python -m trading_agent entry point
│   │
│   │   # ── Configuration ──
│   ├── config/
│   │   ├── __init__.py               # AppConfig + IntelligenceConfig (load_config)
│   │   ├── loader.py                 # TradingRulesConfig dataclasses + load_trading_rules()
│   │   └── trading_rules.yaml        # Trader-tunable algorithm parameters (single source of truth)
│   │
│   │   # ── Orchestrator ──
│   ├── core/
│   │   ├── agent.py                  # TradingAgent: two-stage cycle, timeout guard, sentiment pipeline
│   │   ├── ports.py                  # Hexagonal protocols: MarketDataPort, BrokerPort, SentimentReadout
│   │   ├── stage_monitor.py          # Stage 1 helpers (position exit orchestration)
│   │   └── stage_plan.py             # Stage 2 helpers (per-ticker plan/risk/execute flow)
│   │
│   │   # ── Market Data (Phase I) ──
│   ├── market/
│   │   ├── market_data.py            # yfinance + Alpaca (TTL cache, parallel fetch, leadership z-score)
│   │   ├── market_profile.py         # MarketProfile (timezone, session bounds, trading-day oracle)
│   │   ├── market_hours.py           # Market session gating (open/closed/after-hours)
│   │   └── calendar_utils.py         # Trading-calendar helpers
│   │
│   │   # ── Strategy Pipeline ──
│   ├── strategy/
│   │   ├── regime.py                 # Phase II  — SMA / RSI / Bollinger / VIX / leadership regime classifier
│   │   ├── strategy.py               # Phase III — strike selection, adaptive spread width, nearest-Friday DTE
│   │   └── risk_manager.py           # Phase IV  — 8-guardrail validator
│   │
│   │   # ── Execution ──
│   ├── execution/
│   │   ├── executor.py               # Phase VI  — mleg order execution + HTML report
│   │   ├── position_monitor.py       # Stage 1   — monitor & close open spreads
│   │   └── order_tracker.py          # Stage 1   — fill tracking
│   │
│   │   # ── Intelligence Layer ──
│   ├── intelligence/
│   │   ├── journal_kb.py             # Always-on signal logger (JSONL + Markdown)
│   │   ├── trade_journal.py          # Full-lifecycle trade logging (TradeEntry)
│   │   ├── knowledge_base.py         # File-based RAG vector store
│   │   ├── llm_client.py             # OpenAI-compatible LLM client + make_llm_client(role) factory
│   │   ├── llm_analyst.py            # Pre/post trade LLM analysis — consumes SentimentReadout
│   │   └── fine_tuning.py            # Training data export (JSONL / Alpaca / DPO)
│   │
│   │   # ── Sentiment Pipeline ──
│   ├── sentiment/
│   │   ├── sentiment_pipeline.py     # SentimentPipeline facade — Tier-0/1/2 gating, cycle-scoped pool
│   │   ├── earnings_calendar.py      # Tier-0 — yfinance-backed authoritative event_risk short-circuit
│   │   ├── sentiment_cache.py        # Tier-1 — SHA-1 content-hash gate (TTL + LRU)
│   │   ├── news_aggregator.py        # Tier-2 — NewsItem + NewsAggregator (Yahoo/SEC/Fed/Reddit/Twitter)
│   │   ├── fingpt_analyser.py        # Tier-2 — FinGPT specialist (SentimentReport)
│   │   └── sentiment_verifier.py     # Tier-2 — Reasoning verifier (VerifiedSentimentReport)
│   │
│   │   # ── Reporting ──
│   ├── reporting/
│   │   ├── visualize_logs.py         # Agent Performance Dashboard generator (HTML report)
│   │   └── trade_plan_report.py      # Per-ticker HTML trade plan report
│   │
│   │   # ── Utilities ──
│   ├── utils/
│   │   ├── logger_setup.py           # Structured logging setup
│   │   ├── daily_state.py            # Daily equity / drawdown state persistence
│   │   ├── file_locks.py             # Cross-process file locking
│   │   ├── shutdown.py               # Graceful shutdown signal handling
│   │   └── thesis_builder.py         # LLM prompt context builder
│   │
│   │   # ── Streamlit Dashboard ──
│   └── streamlit/
│       ├── app.py                    # Dashboard entry point (tab router + logging setup)
│       ├── live_monitor.py           # Live Monitoring tab
│       ├── backtest_ui.py            # Backtesting tab + Backtester engine
│       ├── llm_extension.py          # LLM Extension tab (RAG chat + strategy optimizer)
│       └── components.py             # Shared Plotly charts and Streamlit UI primitives
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
python -m trading_agent

# Dry-run (no orders sent)
python -m trading_agent --dry-run

# Custom .env
python -m trading_agent --env /path/to/.env
```

### 4. Schedule (5-minute interval)

```bash
# crontab -e
*/5 9-16 * * 1-5 cd /path/to/trading-agent && python -m trading_agent >> logs/cron.log 2>&1
```

#### After-hours automatic shutdown

| Condition | Action |
|-----------|--------|
| Before 9:25 AM ET (Mon–Fri) | Exit 0 — too early |
| After 4:05 PM ET (Mon–Fri) | Exit 0 — market closed |
| Saturday / Sunday | Exit 0 — weekend |

Override for after-hours paper testing:
```bash
FORCE_MARKET_OPEN=true python -m trading_agent
```

### 5. Run tests

```bash
python run_tests.py
pytest tests/ -v
```

---

## Configuration Reference

### Algorithm Parameters (`trading_rules.yaml`)

Trader-tunable algorithm constants live in `trading_agent/config/trading_rules.yaml` — separate from `.env` so they can be version-controlled and reviewed without touching secrets. The file is **required**; `load_trading_rules()` raises `FileNotFoundError` if it is missing. Any key absent from the file falls back to the dataclass default for that field (partial YAML is fine; missing file is not).

Override the path via the `TRADING_RULES_YAML_PATH` environment variable.

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `strategy` | `min_delta` | `0.15` | Minimum absolute delta for short strike |
| `strategy` | `target_dte` | `35` | Target days-to-expiry |
| `strategy` | `dte_range_min` | `28` | Minimum acceptable DTE |
| `strategy` | `dte_range_max` | `45` | Maximum acceptable DTE |
| `strategy` | `spread_width_floor` | `5.0` | Minimum spread width ($) — adaptive width can exceed this |
| `strategy` | `rs_zscore_threshold` | `1.5` | Leadership z-score needed to trigger Bull Put bias |
| `regime` | `vix_inhibit_zscore` | `2.0` | VIX z-score threshold to demote bullish strategies |
| `regime` | `bollinger_narrow_threshold` | `0.04` | Bollinger Band width below which regime is Sideways |
| `regime` | `leadership_anchors` | _(ETF map)_ | Dict mapping each ticker to its benchmark anchor |
| `position_monitor` | `profit_target_pct` | `0.50` | Close when credit decays to this fraction of max profit |
| `position_monitor` | `hard_stop_multiplier` | `3.0` | Close when loss reaches N× initial credit |
| `position_monitor` | `strike_proximity_pct` | `0.01` | Close when underlying is within N% of short strike |
| `position_monitor` | `dte_safety_hour` | `15` | DTE safety check hour (24h ET) |
| `position_monitor` | `dte_safety_minute` | `30` | DTE safety check minute |
| `agent` | `cycle_timeout_seconds` | `270` | Hard timeout before `os._exit(1)` |
| `agent` | `exit_debounce_required` | `3` | Consecutive exit signals before closing |
| `execution` | `max_history` | `200` | Max order history records to keep |
| `execution` | `price_drift_warn_pct` | `0.10` | Warn when credit drifts >N% from planning values |
| `cache` | `price_history_ttl` | `14400` | Historical OHLCV cache TTL (s) |
| `cache` | `snapshot_ttl` | `60` | Stock snapshot cache TTL (s) |
| `cache` | `option_chain_ttl` | `180` | Option chain cache TTL (s) |
| `cache` | `intraday_return_ttl` | `60` | 5-min return series cache TTL (s) |
| `cache` | `max_prefetch_workers` | `5` | Thread-pool size for parallel history prefetch |
| `sentiment` | `source_weights` | _(dict)_ | Authority weight overrides per news source |
| `backtest` | `starting_equity` | `100000.0` | Backtester initial equity ($) |
| `backtest` | `commission_round_trip` | `2.60` | Round-trip commission per spread ($) |
| `backtest` | `daily_otm_pct` | `0.03` | Daily-bar OTM % for synthetic strike placement |
| `backtest` | `intraday_otm_pct` | `0.005` | Intraday-bar OTM % for synthetic strike placement |

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
| `LIQUIDITY_MAX_SPREAD` | `0.05` | Absolute floor of the underlying bid/ask gate ($) |
| `LIQUIDITY_BPS_OF_MID` | `0.0005` | Slope of the bid/ask gate (5 bps × mid). Effective threshold per ticker = `max(LIQUIDITY_MAX_SPREAD, LIQUIDITY_BPS_OF_MID × mid)`. Prevents a flat 5-cent cap from over-rejecting high-priced names (SPY ≈ $500 → $0.25 gate, GOOG ≈ $170 → $0.085 gate). |
| `STALE_SPREAD_PCT` | `0.01` | When `(spread / mid)` exceeds this, the underlying quote is treated as stale (common on the free IEX feed outside RTH or right at the open). The check soft-passes with a `WARNING` instead of hard-failing the trade. |
| `FORCE_MARKET_OPEN` | `false` | Bypass market-hours check (paper testing) |
| `ALPACA_STOCKS_FEED` | `iex` | Stock snapshot / bar feed. Free/basic Alpaca accounts cannot read `sip` and 403 without an explicit feed; `iex` is the correct free-tier choice. Set to `sip` if you have a paid SIP subscription. |
| `ALPACA_OPTIONS_FEED` | `indicative` | Option snapshot feed. `indicative` is free (15-min delayed); set to `opra` for real-time on a paid OPRA subscription. |

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

`trading_agent/reporting/visualize_logs.py` parses the signal journal and per-ticker trade-plan files to generate a self-contained interactive HTML report. A convenience entry point at the repo root delegates to it.

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
| **📊 Backtesting** | Date range · multi-ticker · timeframe (1Day/5Min) · **Live Quote Refresh** (Alpaca API) · simulated P&L · metric cards · per-regime bar chart · equity + drawdown charts · trade log · CSV/JSON/Journal export |
| **🤖 LLM Extension** | Chat with local Ollama model (RAG over journal) · Optimize Strategy → one-click `.env` update |

#### Backtesting Live Quote Refresh

The backtester mirrors the live agent's **executor._refresh_limit_price()** pattern:

- **When**: Immediately before simulating each trade (Phase VI)
- **What**: Fetches fresh option chain from Alpaca API for the trade's underlying
- **Why**: Prevents simulating trades on stale quotes that would be rejected in live trading
- **Guardrails**: Re-validates credit-to-width ratio and max-loss per contract
- **Drift Detection**: Logs warnings when credit drifts >10% from planning values
- **Rejection**: Skips trades that fail live quote guardrails (same as live agent)

**Two gates govern when the refresh actually runs** (changes here must be paired with `tests/test_streamlit/test_backtest_ui.py::TestRefreshGating`):

1. **Bypassed in `use_alpaca_historical` mode.** When historical mode is enabled, the planning stage already fetched the REAL option chain for the actual entry date. Refreshing against today's snapshot would *replace* that honest credit with a stale-vs-actual-entry quote and produce nonsense drift warnings (e.g. $6 credits on $5-wide spreads when the underlying has moved since entry). In historical mode the refresh stage is a no-op by design — the historical plan IS the truthful quote.
2. **Gated by `_SNAPSHOT_FRESH_DAYS` (default 3).** Outside snapshot mode and the entry date is older than 3 days, refresh is skipped because today's quote is structurally meaningless as a proxy for the quote at that bar's timestamp. Same threshold the snapshot *planning* path uses.

**When refresh fires**: snapshot mode (i.e. `use_alpaca_historical=False`) AND `(today - entry_date) <= _SNAPSHOT_FRESH_DAYS`. In practice this means it only fires on the most recent few bars of the backtest, which is exactly when "today's quote" is a defensible approximation of "the quote at entry".

**Configuration**: Enable by setting `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in your `.env` file. The backtester will automatically use live quotes when available, falling back to simulated quotes when the API is unavailable.

#### Backtester ETF Macro Signals (parity with live Items 1-3)

The backtester now applies the **same z-scored leadership and VIX inter-market gates** the live `RegimeClassifier` uses, closing the largest pre-patch live-vs-backtest drift item.

| Live agent | Backtester equivalent |
|---|---|
| `MarketDataProvider.get_leadership_zscore(ticker, anchor)` (5-min Alpaca bars, last 21 bars, OPEN_BAR_SKIP=2) | `Backtester._leadership_zscore_at(ticker_prices, anchor_prices, idx)` (per-bar slice of pre-loaded yfinance close series) |
| `MarketDataProvider.get_vix_zscore()` (5-min ^VIX from yfinance, last 21 bars) | `Backtester._vix_zscore_at(vix_series, ts)` (per-bar slice of pre-loaded yfinance ^VIX series) |
| `StrategyPlanner.plan()` Priority 2 — VIX inhibit demotes Bull Put / Iron Condor → Bear Call | Same gate fires in `Backtester.run()` per-bar loop (counted in `Backtester.vix_inhibited`) |
| `StrategyPlanner.plan()` Priority 3 — Leadership-z bias picks Bull Put when `z > 1.5σ` | Same gate fires per-bar (counted in `Backtester.leadership_biased`); applied only when the regime would otherwise have been SIDEWAYS to keep the legacy bullish/bearish paths unchanged |

**Enabling**: pass `use_macro_signals=True` to `Backtester(...)`. Default is `False` to keep the existing test suite green; the Streamlit UI default-enables for interactive runs.

**Constants reused** (single source of truth — tunable via `trading_rules.yaml`, consumed by both the live agent and the backtester):

- `leadership_anchors` (anchor map) — `trading_rules.yaml` → `regime.leadership_anchors`
- `VIX_INHIBIT_ZSCORE = 2.0` — `trading_rules.yaml` → `regime.vix_inhibit_zscore`
- `RS_ZSCORE_THRESHOLD = 1.5` — `trading_rules.yaml` → `strategy.rs_zscore_threshold`
- `LEADERSHIP_WINDOW_BARS = 21`
- `VIX_WINDOW_BARS = 21`

**Documented residual drift** (parity caveats not yet closed):

1. **Bar timescale** — live z-scores fire on 5-minute bars. The backtester runs the same arithmetic on whatever bar interval `timeframe` selects; for `timeframe="1Day"` the gates compare daily return diffs (directionally aligned with the live signal but at a different timescale). For true 5-min parity, use `timeframe="5Min"` (yfinance caps this at the last ~30 days).
2. **Open-bar skip** — the live `MarketDataProvider` drops the first 2 bars after the 9:30 ET open to suppress the open-print spike. The backtester does not currently re-implement this trim per session day; rolling z-scores on intraday data therefore include those bars. Impact: marginally higher stdev → slightly conservative z-scores (gate fires less, not more, vs live).
3. **Anchor data source** — live agent fetches from Alpaca; backtester fetches from yfinance. Close-vs-close they align within fractions of a basis point on liquid ETFs, but corporate-action edge cases differ.

#### Backtester Rules vs. Live Agent — Parity Matrix

The backtester runs three different credit-pricing paths depending on configuration. **Only one of them produces a per-bar dynamic credit derived from real option-market data**; the other two are heuristics. This section is the source of truth for what is and isn't replicated from the live `StrategyPlanner` / `RegimeClassifier` / `RiskManager` pipeline.

**Three credit-pricing modes** (`backtest_ui.py::Backtester.run()` lines ~2204-2346):

| Mode | Trigger | Credit formula | Dynamic? | Honest window |
|---|---|---|---|---|
| **Alpaca historical** | `use_alpaca_historical=True` | `short_close − long_close` from real `/v1beta1/options/bars` for the entry day | **Yes — truly per-bar** | Last ~30 calendar days (Alpaca options retention) |
| **σ-credit (synthetic)** | `sigma_mult` set, no Alpaca data | `spread_width × clip(0.45 − 0.15·σ_mult, 0.05, 0.45)` | Quasi — varies with σ-distance, not real IV | Any range, but it's a model not a quote |
| **Legacy fixed-% OTM** | `sigma_mult=None` | `spread_width × credit_pct` (default `5 × 0.30 = $1.50`) | **No — constant on every trade** | Calibration sanity-check only |

**Rule-by-rule comparison** with the live agent:

| Live rule (`strategy.py` / `regime.py` / `risk_manager.py`) | Backtester behavior | Status |
|---|---|---|
| Adaptive spread width: `_pick_spread_width = max(SPREAD_WIDTH=5, 3×grid, 2.5%×spot)`, snapped UP to grid | Fixed `self.spread_width` constructor arg; even Alpaca-historical mode uses the constant as the long-leg target distance | ❌ **Drift** — live SPY/QQQ at $500+ trades $15-20-wide spreads; backtester stays at $5 |
| Strike picker: filter listed greeks for `MIN_DELTA(0.15) ≤ |Δ| ≤ max_delta(0.20)`, pick widest gap | σ-projection (`sigma_mult × σ × √(hold_bars/bars_per_year)`) → snap to closest OTM listed strike. δ is *estimated* from the σ-distance map (`_delta_from_sigma_distance`), not read from the chain. **Hold-horizon σ projection** ensures intraday backtests pick strikes ~1% OTM (|Δ|≈0.20) instead of the ~13% OTM (|Δ|≈0.02) the legacy DTE-horizon math produced — see `TestAlpacaHistoricalSigmaHorizon` | ⚠️ **Reduced drift** (Apr 2026) — averages line up long-run; bar-by-bar diverge in vol regimes where realized σ ≠ implied σ. Strike-distance is now apples-to-apples for both daily AND intraday timeframes |
| Net credit: `sold.bid − bought.ask` (live limit-style fill) | Synthetic uses heuristic curve; Alpaca-historical uses `close − close` (no bid/ask spread modelled) | ❌ **Drift** — backtester systematically over-estimates fillable credit relative to a real mid-or-better limit |
| Priority 1 — Mean Reversion (3-std BB touch overrides everything) | Not implemented in backtest run-loop | ❌ **Missing** |
| Priority 2 — VIX inter-market inhibit (`vix_z > +2σ` demotes Bull Put / Iron Condor → Bear Call) | Wired via `use_macro_signals=True` (`Backtester.vix_inhibited` counter) | ✅ |
| Priority 3 — Leadership Z-score bias (`leadership_z > +1.5σ` → Bull Put) | Wired via `use_macro_signals=True` (`Backtester.leadership_biased` counter); applied only on SIDEWAYS bars (preserves legacy bullish/bearish paths) | ✅ |
| `min_credit_ratio = 0.33` (StrategyPlanner gate) | Wired via `min_credit_ratio` param. Skip is now **scoped to pure synthetic-σ mode only** (`use_sigma_path AND NOT use_alpaca_historical`); when σ-strike-picking runs alongside Alpaca-historical bars the credit comes from real `close − close` arithmetic and the gate is fully active. Bug history (Apr 2026): skip used to fire on `use_sigma_path` alone, which silently disabled the gate in alpaca-historical+σ runs and let trades with C/W < 0.10 reach the trade journal — see `TestCreditRatioGateAlpacaHistorical` | ✅ Active in fixed-% / Alpaca-historical modes; intentionally skipped in pure synthetic-σ |
| `max_delta = 0.20` (short-leg cap) | Wired via `max_delta` param, but compared against the σ-derived approximation, not a chain-sourced delta | ⚠️ Same number, different signal |
| `RiskManager.max_risk_pct × equity` (per-trade max-loss cap) | Wired via `max_risk_pct` param | ✅ |
| IV-rank high-vol guard (`RegimeClassifier.high_iv_warning`) | Wired via `use_iv_gate=True` + `iv_high_threshold` | ✅ |
| Earnings Tier-0 short-circuit (`EarningsCalendar`, default 7-day lookahead) | Wired via `use_earnings_gate=True` + `earnings_lookahead_days` | ✅ |
| Stop-loss / profit-target exits (50% / 75%) | Wired via `stop_loss_pct` + `profit_target_pct` | ✅ |
| `executor._refresh_limit_price()` parity (re-fetch quote immediately before fill, re-validate guardrails) | Wired in snapshot mode within `_SNAPSHOT_FRESH_DAYS=3`; intentionally **skipped** in Alpaca-historical mode (see "Backtesting Live Quote Refresh" above) | ✅ — see `TestRefreshGating` |

**Recommended live-parity configuration.** For the closest the codebase can get to apples-to-apples right now:

```python
Backtester(
    use_alpaca_historical=True,   # real chain + real entry-day bars
    use_macro_signals=True,       # VIX inhibit + leadership-z bias
    use_iv_gate=True,
    use_earnings_gate=True,
    min_credit_ratio=0.33,
    max_delta=0.20,
    max_risk_pct=0.02,            # match RiskManager default
    stop_loss_pct=0.50,
    profit_target_pct=0.75,
    # Keep the date range inside Alpaca's ~30-day options retention window
)
```

**Known residual drift sources** (not yet wired — track here so they don't get rediscovered as "bugs" in production):

1. **Adaptive spread width** — backtester uses the fixed `spread_width` arg everywhere; live scales with spot/grid. Material on $500+ underlyings.
2. **Chain-sourced δ picker (residual drift, reduced Apr 2026)** — As of the σ-horizon fix, the backtester now projects σ over the *hold horizon* (`hold_bars / bars_per_year`) rather than the full DTE, so intraday strikes land where live's `MIN_DELTA(0.15) ≤ |Δ| ≤ 0.20` band would put them (~1% OTM on 1-hour holds) instead of the ~13% OTM the legacy DTE-horizon math produced. **What still drifts:** δ is *labelled* via the analytic `_delta_from_sigma_distance` mapping, not read from the chain's listed `delta` column. In quiet-vol regimes where realized σ ≈ implied σ, this is a near-no-op; in vol regimes where they diverge, the backtester and live can pick neighboring strikes. See `TestAlpacaHistoricalSigmaHorizon` for the regression bound.
3. **Mean-Reversion priority** — Priority 1 of the live plan() is absent from the backtest run-loop.
4. **Bid/ask spread modelling** — Alpaca-historical uses `close − close`; live fills against bid/ask. Over-estimates real credit.

**Recently closed gaps (Apr 2026)** — fixes applied; verify in your next backtest run:

- **Credit-ratio gate now fires in alpaca-historical mode.** The skip-condition was widened too far in an earlier change and silently bypassed `min_credit_ratio` for any run with σ-strike-picking on, including alpaca-historical mode where credits come from real bars. Skip is now `use_sigma_path AND NOT use_alpaca_historical`. Expect to see `credit_ratio<floor` rejections appear in the diagnostics funnel for alpaca-historical runs whose real bars produce thin premium (C/W < 0.33). Regression: `TestCreditRatioGateAlpacaHistorical`.
- **Structured rejection reasons for `_alpaca_historical_plan`.** The historical-plan builder now returns `(plan, reason)` instead of `Optional[Dict]`. On failure, `reason` is a `<token>: <context>` string naming the specific upstream cause (`no_expiration_in_window`, `no_bars_on_entry_day`, `long_leg_off_grid`, `non_positive_credit`, …). The token before the colon becomes a sub-gate suffix in `Backtester.rejections` so per-cause counters appear separately in the decision-path diagnostics instead of collapsing into one generic "Alpaca data unavailable" bucket.
- **Trade-journal `expiry_date` reflects the actual option expiration.** The CSV journal column used to record `entry_date + 1 day` for every intraday trade, making it look like the agent was trading 0/1-DTE options when the planner actually picks ~30-DTE contracts (`TARGET_DTE=35`). The column now uses the OCC contract's expiration string from the alpaca-historical plan when available, falls back to `entry + target_dte` for synthetic-σ intraday runs, and keeps `entry + hold_count` for synthetic-σ daily runs. Verify in your trade log: `expiry_date − entry_date` should match the planner's target DTE, not 1 day.
- **Friday-weekly preference in `_pick_alpaca_expiration`** — the picker used to select strict-nearest-DTE, which on most weekday entry-dates landed on a Mon/Wed/Thu weekly. Mon/Wed weeklies on QQQ/SPY/IWM trade at a fraction of Friday-weekly volume, so Alpaca's options-bars endpoint returned empty results for ~80% of intraday backtest candidates. The picker now adds a +4-day penalty to non-Friday expirations so Friday weeklies within 4d of the target_dte beat ties; tiebreaker prefers Friday over non-Friday and earlier expirations over later. Math: `target = entry+35d` (Mon) → Mon@35d effective-diff 0+4=4, Wed@37d 2+4=6, Fri@32d 3+0=**3 (wins)**, Fri@39d 4+0=4. Regression: `TestPickAlpacaExpirationFridayPreference`.
- **Expiration-fallback loop on data-availability failures.** When the first-choice expiry returns empty bars (or no contracts, or non-positive credit), `_alpaca_historical_plan` now retries with the next-best expiration up to `MAX_FALLBACK_ATTEMPTS=3` times before giving up. Strike-grid failures (`long_leg_off_grid`, `no_otm_near_target`) do NOT retry because they're deterministic given the catalogue. New rejection token: `no_bars_after_fallbacks` (caught only when all retries are exhausted). Internal helper `_build_alpaca_plan_for_expiration` was extracted to make this loopable. Regression: `TestAlpacaHistoricalPlanFallback`.

---

## Signal Journal Format (`signals.jsonl`)

Every trade attempt is logged as a single JSON line including `fingpt_sentiment`, `fingpt_event_risk`, `fingpt_recommendation`, `fingpt_agreement`, `fingpt_hallucination_flags`, and `fingpt_verified_by` when the sentiment pipeline is active.

Action values: `dry_run`, `submitted`, `rejected`, `skipped_by_llm`, `skipped_existing`, `skipped_liquidation_mode`, `skipped`, `error`, `cycle_timeout`, `daily_drawdown_circuit_breaker`.
