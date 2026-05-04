# Autonomous Options Credit Spread Trading Agent

An autonomous trading agent that generates daily income through high-probability, risk-defined options credit spreads. Primary goal: **capital preservation** — every trade has a known, capped maximum loss.

## Contents

- [Quickstart](#quickstart)
- [Architecture Overview](#architecture-overview)
- [Strategy Selection](#strategy-selection)
- [Adaptive Chain Scanner](#adaptive-chain-scanner)
- [Risk Management Guardrails](#risk-management-guardrails)
- [Live ↔ Backtest Unified Decision Engine](#live--backtest-unified-decision-engine)
- [Intelligence & Sentiment Layers (Optional)](#intelligence--sentiment-layers-optional)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Setup & Configuration](#setup--configuration)
- [Project Structure](#project-structure)
- [Signal Journal Format](#signal-journal-format)

---

## Quickstart

```bash
pip install -r requirements.txt

# minimum .env
echo "ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
TICKERS=SPY,QQQ
DRY_RUN=true" > .env

# run a single cycle (paper / dry-run)
python -m trading_agent.agent --dry-run

# launch the dashboard (Live + Backtest + LLM tabs)
streamlit run trading_agent/streamlit/app.py

# 5-minute cron schedule
*/5 9-16 * * 1-5 cd /path/to/trading-agent && python -m trading_agent.agent >> logs/cron.log 2>&1
```

After-hours behaviour: exits cleanly before 9:25 AM ET, after 4:05 PM ET, and on weekends. Override with `FORCE_MARKET_OPEN=true` for paper testing.

---

## Architecture Overview

Each cycle runs two stages sequentially: monitor existing positions first, then evaluate new opportunities per ticker.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       AGENT CYCLE  (agent.py)                       │
│                                                                     │
│  STAGE 1 — Monitor Open Positions                                   │
│    Position Monitor → Exit Signal Check → Order Tracker             │
│      (50 % stop-loss, 75 % profit, regime shift, DTE safety)        │
│                                                                     │
│  STAGE 2 — Open New Positions  (per ticker)                         │
│    I·Perceive → II·Classify → III·Plan → IV·Risk → V·LLM → VI·Exec  │
│    yfinance     SMA / RSI     ChainScanner   8 guardrails           │
│    Alpaca       Bollinger     decision_      buying-power           │
│    snapshots    VIX-z gate    engine.decide  daily-DD breaker       │
│                                                                     │
│    Sentiment Pipeline runs concurrently (Tier 0/1/2 gating) and     │
│    delivers a VerifiedSentimentReport to Phase V (LLM Analyst).     │
│    Future is cancelled if Phase III/IV reject the trade.            │
└─────────────────────────────────────────────────────────────────────┘
```

> An interactive HTML diagram lives at `architecture_diagram.html`.

**SMA-50 slope units.** `MarketDataProvider.sma_slope()` returns the 5-day average **dollar change per day** of the SMA. Downstream consumers only read the sign; logs and the LLM prompt annotate `$/day` so the magnitude isn't mistaken for a percentage.

---

## Strategy Selection

Strategy choice follows a strict priority order:

| Priority | Regime | Detection | Strategy |
|---|---|---|---|
| **1** | **Mean Reversion** | Price touches 3-σ Bollinger Band | Mean-Reversion Spread |
| 2 | **VIX inhibit** | `vix_z > +2 σ` AND regime ∈ {Bullish, Sideways} | Bear Call (demoted) |
| 3 | **Bullish + Lead-z** | Bullish AND `leadership_z > +1.5 σ` vs anchor | Bull Put (leadership bias) |
| 4 | **Sideways + Lead-z** | Sideways AND `leadership_z > +1.5 σ` vs anchor | Bull Put (leadership bias) |
| 5 | **Bullish** | `Price > SMA-200` AND `SMA-50 slope > 0` | Bull Put |
| 6 | **Bearish** | `Price < SMA-200` AND `SMA-50 slope < 0` | Bear Call |
| 7 | **Sideways** | Between SMAs / narrow Bollinger | Iron Condor |

**Mean Reversion.** A 3-σ band touch is statistically extreme; the agent expects reversion. Upper touch → Bear Call above price; lower touch → Bull Put below price.

**Z-scored leadership bias.** `LEADERSHIP_ANCHORS` in `regime.py` maps each ticker to a sibling benchmark (`SPY → QQQ`, sector ETFs → SPY, …). `MarketDataProvider.get_leadership_zscore(ticker, anchor)` normalises the latest 5-min return differential against its own ~20-bar rolling distribution (first 2 open bars dropped to suppress the open-print spike). When `z > 1.5 σ` and regime is Bullish/Sideways, the planner picks Bull Put instead of the default mapping.

**VIX inter-market gate.** `^VIX` is fetched via yfinance (Alpaca doesn't carry the index) and z-scored over the last ~20 5-min bars. `vix_z > +2 σ` flips `inter_market_inhibit_bullish=True` and demotes Bull Put / Iron Condor → Bear Call for that cycle. Mean-reversion bypasses the gate.

**Adaptive spread width.** No longer flat `$5`. Per chain the planner computes `width = max(SPREAD_WIDTH_FLOOR, 3 × strike_grid_step, 0.025 × spot)`, snapped UP to grid. Result: `$5` on an `$80` ticker (floor wins), `$15-20` on SPY/QQQ at `$700` (spot-percentage wins). The legacy `SPREAD_WIDTH = $5` is now a hard floor, never a target.

**DTE targeting.** Theta capture is concentrated 25-40 DTE. Default `TARGET_DTE = 35`, accepted range `(28, 45)`. Highest-DTE Friday in range wins ties.

---

## Adaptive Chain Scanner

`chain_scanner.py` replaces the legacy "single point in chain space" planner with a scored sweep. For every `(DTE, target Δshort, width)` tuple in the configured grid the scanner fetches the relevant put/call chain, picks the contract closest to target Δ, picks the protective leg `width × spot` strikes away (snapped to grid), prices the spread off NBBO mids, and scores it.

**Credit pricing — `_quote_credit`.** `short_mid − long_mid − fill_haircut`, where each leg's mid is `(bid+ask)/2` when both are positive and the conservative side (short→bid, long→ask) when a quote is missing. Default `fill_haircut = $0.02` matches the executor's per-leg slippage budget so scored credit and targeted-fill limit price stay in sync. Worst-case `short_bid − long_ask` is no longer used for scoring.

**Score formula:**

```
POP         ≈ 1 − |Δshort|
C/W         = credit / width
EV/$risked  = (POP × C/W − (1 − POP) × (1 − C/W)) / (1 − C/W)
annualized  = EV/$risked × (365 / DTE)
```

**Hard filters before scoring:** `POP ≥ min_pop` (default 0.55), `C/W ≥ |Δshort| × (1 + edge_buffer)` — the **edge floor** (default `edge_buffer = 0.10` → 10 % over breakeven), positive net credit, both legs quoting non-zero bid/ask.

The scanner returns `[]` when nothing passes — the agent treats this as `skipped: no edge` and journals it. Annualized score breaks ties.

**Single-source-of-truth invariant.** The same `cw_floor = |Δshort| × (1 + edge_buffer)` formula appears in `chain_scanner.py`, `risk_manager.py`, and `executor.py`. `scan_invariant_check.py` AST-walks all three sites every release; the static `MIN_CREDIT_RATIO=0.33` floor is now only used when adaptive mode is disabled.

---

## Risk Management Guardrails

Every trade must pass **all eight checks** before execution:

| # | Check | Rule |
|---|---|---|
| 1 | Plan Validity | Strategy planner found valid strikes and contracts |
| 2 | Credit-to-Width | **Adaptive**: `C/W ≥ |Δshort| × (1 + edge_buffer)`. **Static**: `C/W ≥ MIN_CREDIT_RATIO` (0.33) |
| 3 | Sold Delta | `≤ MAX_DELTA` (default 0.20) |
| 4 | Max Loss | `≤ MAX_RISK_PCT × equity` (default 2 %) |
| 5 | Account Type | Must be `paper` |
| 6 | Market Hours | Market must be open |
| 7 | Underlying Liquidity | `bid/ask spread < max(LIQUIDITY_MAX_SPREAD, LIQUIDITY_BPS_OF_MID × mid)`; stale quotes (`spread/mid > STALE_SPREAD_PCT`) soft-pass with a WARNING |
| 8 | Buying Power | `available BP ≥ (1 − MAX_BUYING_POWER_PCT) × equity` |

`Max Loss = (Width − Credit) × 100`. The sentiment pipeline is advisory only — it can tighten constraints, never loosen them.

**Daily Drawdown Circuit Breaker.** Equity drop > `DAILY_DRAWDOWN_LIMIT` (default 5 %) from the day's open → log + `os._exit(1)`.

**Liquidation Mode.** Available BP > `MAX_BUYING_POWER_PCT` (default 80 %) → Stage 2 skipped; Stage 1 continues.

**Capital Retainment Guards.** Macro Guard (skips Bull Put when `price < SMA-200`); High-IV Block (skips ALL new entries when realized-vol IV rank > 95th percentile).

**Position Exit Debouncing.** Non-immediate exit signals require **3 consecutive cycles** (~15 min). Bypassed by `HARD_STOP` (lost ≥ 3× initial credit), `STRIKE_PROXIMITY` (within 1 % of any short strike), `DTE_SAFETY` (Thursday after 15:30 ET, expiry next day).

**Live Quote Refresh at Execution.** The executor fetches a fresh, no-cache quote for both leg symbols immediately before order submission and re-validates economics-bearing guardrails (credit ratio, max loss) against the live credit.

---

## Live ↔ Backtest Unified Decision Engine

The previous biggest source of drift was that the live agent ran a `(DTE × Δ × width)` adaptive sweep through `ChainScanner.scan()` while the backtester ran a homegrown σ-distance heuristic in `_alpaca_historical_plan` — different strike picker, different credit pricing, different EV/POP logic, no shared code. Any change on one side could silently diverge the other.

The May 2026 unification reshapes this around a single pure function:

```
chain_scanner.py            ← pure helpers (_quote_credit, _score_candidate_with_reason)
        │
        ▼
decision_engine.decide()    ← pure scoring entrypoint (no I/O)
        │
        ├──── ChainScanner.scan()                          (live)
        └──── Backtester._build_alpaca_plan_via_decide()   (backtest)
```

`decision_engine.decide(DecisionInput) -> DecisionOutput` is a pure function with no I/O, no calendar lookups, no broker calls. It takes `ChainSlice`s in (one expiration's `{strike, delta, bid, ask, symbol}` dicts plus the DTE), runs the full `(Δ × width)` sweep, and returns ranked `SpreadCandidate`s plus a `ScanDiagnostics` block. Both clients (`ChainScanner.scan` and `Backtester._build_alpaca_plan_via_decide`) delegate to it — neither owns the scoring logic.

**Backtester opt-in.** Pass `use_unified_engine=True` and `preset=<PresetConfig>` to `Backtester(...)`, or flip the **Unified Decision Engine** toggle in the Streamlit Backtesting tab. When set, `_build_alpaca_plan_for_expiration` synthesises a `ChainSlice` from Alpaca-historical contracts + bars (bar close stands in for both bid and ask; |Δ| approximated from σ_hold via Black-Scholes since Alpaca's historical endpoints don't return Greeks) and delegates to `decide()`. When unset, the legacy σ-distance heuristic still runs — preserved for the existing test suite.

**Drift-prevention enforcement** lives in three places:

1. **`scan_invariant_check.py`** — AST walker (CI). Asserts (a) the `|Δ|×(1+edge_buffer)` C/W floor formula appears in `chain_scanner.py`, `risk_manager.py`, and `executor.py`; (b) no module *outside* `chain_scanner` and `decision_engine` defines `_score_candidate`, `_score_candidate_with_reason`, or `_quote_credit`; (c) `streamlit/backtest_ui.py` contains a `decide(...)` call.
2. **`run_live_vs_backtest_parity_check.py`** — end-to-end smoke driver that builds a synthetic chain, feeds it to *both* `ChainScanner.scan()` and `Backtester._build_alpaca_plan_via_decide()`, and asserts identical strike picks + matching credit (Δ ≤ $0.01).
3. **Separate journal files.** Live writes `trade_journal/signals_live.jsonl`; backtest writes `trade_journal/signals_backtest.jsonl`. The LLM analyst corpus and the live-monitor diagnostics panel deliberately read only the live file so synthetic backtest counterfactuals can't bias guardrail recommendations. `JournalKB.__init__` accepts `run_mode={"live","backtest"}`; an unknown value raises `ValueError`.

**Smoke checks** (run in CI; see `scripts/checks/README.md`):

```bash
python3 scripts/checks/scan_invariant_check.py                # AST invariants
python3 scripts/checks/run_scan_diagnostics_check.py          # ChainScanner + decide() integration
python3 scripts/checks/run_unified_backtest_check.py          # Backtester unified-path smoke
python3 scripts/checks/run_journal_split_check.py             # JournalKB run_mode split
python3 scripts/checks/run_live_vs_backtest_parity_check.py   # end-to-end parity
```

All five must pass before any change to scoring, pricing, or floor logic ships.

### Backtester Parity Matrix (residual drift)

The backtester runs three credit-pricing paths depending on configuration. Only Alpaca-historical produces a per-bar dynamic credit derived from real option-market data; the others are heuristics.

| Mode | Trigger | Credit formula | Honest window |
|---|---|---|---|
| **Alpaca historical** | `use_alpaca_historical=True` | `short_close − long_close` from real `/v1beta1/options/bars` | Last ~30 calendar days |
| **σ-credit synthetic** | `sigma_mult` set, no Alpaca | `width × clip(0.45 − 0.15·σ_mult, 0.05, 0.45)` | Any range; model not quote |
| **Legacy fixed-% OTM** | `sigma_mult=None` | `width × credit_pct` (default `5 × 0.30 = $1.50`) | Calibration only |

**Recommended live-parity configuration:**

```python
Backtester(
    use_alpaca_historical=True,
    use_macro_signals=True,        # VIX inhibit + leadership-z bias
    use_iv_gate=True,
    use_earnings_gate=True,
    use_unified_engine=True,       # delegate strike/credit to decision_engine.decide()
    preset=PRESETS["balanced"],
    min_credit_ratio=0.33,
    max_delta=0.20,
    max_risk_pct=0.02,
    stop_loss_pct=0.50,
    profit_target_pct=0.75,
)
```

**Known residual drift sources (track here so they don't get rediscovered as bugs):**

1. **Adaptive spread width** — backtester uses the fixed `spread_width` arg; live scales with spot/grid. Material on $500+ underlyings.
2. **Chain-sourced δ picker** — δ is *labelled* via the analytic `_delta_from_sigma_distance` mapping, not read from the chain's listed `delta` column. Near-no-op when realized σ ≈ implied σ; can pick neighbouring strikes when they diverge. Regression: `TestAlpacaHistoricalSigmaHorizon`.
3. **Mean-Reversion priority** — Priority 1 of live `plan()` is absent from the backtest run-loop.
4. **Bid/ask spread modelling** — Alpaca-historical uses `close − close`; live fills against bid/ask. Slightly over-estimates real credit.

**Recently closed gaps (Apr 2026)** — Friday-weekly preference in `_pick_alpaca_expiration` (penalty for non-Friday expirations); credit-ratio gate now fires in alpaca-historical mode (was silently bypassed); structured rejection reasons in `_alpaca_historical_plan` (`no_expiration_in_window`, `no_bars_on_entry_day`, `long_leg_off_grid`, …); trade-journal `expiry_date` reflects the actual OCC contract expiration; expiration-fallback loop on data-availability failures (up to 3 retries; new token `no_bars_after_fallbacks`).

---

## Intelligence & Sentiment Layers (Optional)

Both layers are off by default. Each degrades gracefully if its dependencies aren't installed.

### Core Intelligence Layer

`Trade Executes → Journal Entry → LLM Post-Trade Analysis → Lessons → KB → Better Decisions Next Cycle → (after 20+ trades) Fine-Tune Local Model`

| Component | File | Role |
|---|---|---|
| LLM Client | `llm_client.py` | OpenAI-compatible — Ollama, LM Studio, Claude API |
| Trade Journal | `trade_journal.py` | Full lifecycle per trade: entry, execution, exit, P&L, lessons |
| Knowledge Base / RAG | `knowledge_base.py` | File-based vector store via `nomic-embed-text`; cosine search; no external DB |
| LLM Analyst | `llm_analyst.py` | Returns approve/modify/skip — **advisory only**, can't override the risk manager |
| Fine-Tuning Pipeline | `fine_tuning.py` | Exports Chat JSONL / Alpaca / DPO formats once 20+ closed trades exist |

### Multi-Source Sentiment Pipeline

A `SentimentPipeline` facade (`sentiment_pipeline.py`) runs concurrently in a background thread during every cycle and delivers a `VerifiedSentimentReport` to the LLM Analyst at Phase V. Three tiers of gating prevent redundant local-LLM calls within the 5-minute budget.

| Tier | Gate | When it fires | LLM calls |
|---|---|---|---|
| **0** | Earnings calendar short-circuit | yfinance reports a scheduled earnings date within `EARNINGS_CALENDAR_LOOKAHEAD_DAYS` (default 7) | None — deterministic `event_risk=1.0` |
| **1** | Content-hash cache | SHA-1 fingerprint over normalised evidence matches a previously-produced `VerifiedSentimentReport` | None — replays cached verified report |
| **2** | Full chain | Evidence changed — runs NewsAggregator → FinGPT specialist → reasoning verifier | FinGPT + verifier |

The cache only ever holds **post-verifier** results, so Tier 1 never weakens the no-hallucination guarantee.

**Source authority weights** (overridable via `NEWS_SOURCE_WEIGHTS_JSON`):

| Source | Weight | Auth | Captures |
|---|---|---|---|
| SEC EDGAR 8-K / 10-Q | 1.00 | None (free REST) | Material events, earnings, insider changes |
| Federal Reserve RSS | 0.95 | None | FOMC statements, rate decisions |
| Yahoo Finance | 0.70 | None | General financial news |
| Twitter / X cashtag | 0.50 | Bearer token | Breaking news, retail momentum |
| Reddit r/options, r/stocks | 0.45 | PRAW | Options-specific sentiment |
| Reddit r/wallstreetbets | 0.35 | PRAW | High-noise retail sentiment |

**FinGPT specialist** — local Ollama model returns a `SentimentReport` with `sentiment_score`, `event_risk`, `confidence`, `recommendation`, `key_themes`, `reasoning`.

**Reasoning verifier** — independent stronger model (`qwq:32b` / `deepseek-r1:32b` locally, or Claude via `VERIFIER_PROVIDER=anthropic`) cross-checks every claim in FinGPT's reasoning against raw evidence. Outputs `verified_sentiment_score`, `verified_event_risk`, `hallucination_flags`, `agreement_score`, `evidence_mapping`. Falls back to passing through the original report unchanged when unavailable.

**Lifecycle.** The pipeline owns its own single-worker `ThreadPoolExecutor` and is used as a context manager from `agent._run_cycle`:

```python
with SentimentPipeline.from_config(cfg.intelligence) as pipeline:
    for ticker in tickers:
        fut = pipeline.submit(ticker, regime, price, rsi, iv_rank, strategy)
        # ... Phase III + IV
        if not (plan.valid and verdict.approved):
            fut.cancel()             # don't waste a local LLM call
        sentiment = fut.result() if fut else None
        # ... Phase V + VI
# pool.shutdown(wait=True) here — every Future belongs to exactly one cycle
```

---

## Streamlit Dashboard

```bash
pip install "streamlit>=1.42.0" "plotly>=6.0.0" "watchdog>=3.0,<5"
streamlit run trading_agent/streamlit/app.py
```

| Tab | Features |
|---|---|
| **📡 Live Monitoring** | Agent Start / Stop / Dry-Run · cycle PID · equity · P&L · regime badge · open positions · equity curve · 8-guardrail status · agent log · Strategy Profile applier · journal expander |
| **📊 Backtesting** | Date range · multi-ticker · timeframe (1Day / 5Min) · Live Quote Refresh · **Unified Decision Engine** toggle (preset selector) · simulated P&L · per-regime bars · equity + drawdown · trade log · CSV / JSON / Journal export |
| **🤖 LLM Extension** | Chat with local Ollama model (RAG over `signals_live.jsonl`) · Optimize Strategy → one-click `.env` update |
| **📊 Watchlist** | Persistent ticker watchlist (`knowledge_base/watchlist.json`) · multi-timeframe regime table (1d / 4h / 1h / 15m / 5m) · ADX strength badge · VIX-z macro strip · 4-row Plotly chart (Price + overlays / Volume / Oscillators / Trend) with 6 timeframes (5m → 1d) and indicator toggles. **Read-only — never imports `decision_engine`, `chain_scanner`, `executor`, or `risk_manager`.** |

**Refresh model.** Refresh is event-driven via `watchdog`. A single per-process `Observer` (wrapped in `@st.cache_resource`) watches `trade_journal/` and `trade_plans/`; loaders cache by `(version, mtime, size)` so unrelated reruns hit the cache and only real journal writes invalidate. Default tick is `LIVE_MONITOR_REFRESH_SECS=3` (was 30). Kill switches: `WATCHDOG_DISABLE=1` (fall back to mtime polling) and `WATCHDOG_FORCE_POLLING=1` (NFS / network mounts where inotify is unavailable).

**Broker-state gating.** Alpaca account / positions / clock fetches in the Live Monitor are TTL-cached (`BROKER_STATE_TTL_SECS=30`) and **only run when the agent loop is started** — opening Streamlit alone makes zero broker calls. A manual `↻ Refresh broker state` button is shown when the loop is stopped, for ad-hoc inspection.

### Strategy Profile

The Live Monitoring tab has a Strategy Profile expander that controls the four knobs that meaningfully change credit-spread economics: short-leg |Δ|, DTE per strategy, spread-width policy, C/W floor. Picking a profile + a directional bias and clicking **Apply** writes `STRATEGY_PRESET.json`; the agent subprocess re-reads it at the start of every cycle.

| Profile | Δ-short | DTE (Vert / IC / MR) | Width | C/W floor | Risk/trade | Approx. POP |
|---|---|---|---|---|---|---|
| Conservative | 0.15 | 35 / 45 / 21 | 2.5 % × spot | 0.20 | 1 % | ~85 % |
| **Balanced** (default) | 0.25 | 21 / 35 / 14 | 1.5 % × spot | 0.30 | 2 % | ~75 % |
| Aggressive | 0.35 | 10 / 21 / 7  | $5 fixed     | 0.40 | 3 % | ~65 % |
| Custom | sliders | sliders | sliders | sliders | sliders | — |

**Directional-bias filter.** Restricts which classifier outputs the agent will trade. Fires immediately after Phase II so disallowed regimes short-circuit before sentiment / chain fetch. Mean-reversion is always allowed (the 3-σ touch override is a fear-spike signal, not a directional view).

**Where the preset is applied.** `agent.__init__` calls `load_active_preset()` and forwards the knobs into `StrategyPlanner` + `RiskManager`; `strategy._pick_expiration(kind=...)` honours the per-strategy DTE override; `strategy._pick_spread_width` honours `width_mode`; `agent._process_ticker` calls `regime_is_allowed(regime, bias)` after classify and writes `action="skipped_bias"` to the live journal on rejection.

`STRATEGY_PRESET.json` is written atomically (temp + rename). Missing / malformed JSON falls back to **Balanced** (logged) so a fresh install is always operational without touching the dashboard.

### Backtesting Live Quote Refresh

The backtester mirrors the live `executor._refresh_limit_price()` pattern: immediately before simulating each trade it fetches a fresh option chain from Alpaca and re-validates `min_credit_ratio` and per-contract max-loss. Two gates govern when refresh actually runs:

1. **Bypassed in `use_alpaca_historical` mode.** Historical mode already fetched the real chain for the actual entry date; refreshing against today's snapshot would produce nonsense drift warnings.
2. **Gated by `_SNAPSHOT_FRESH_DAYS=3`.** In snapshot mode, an entry older than 3 days is skipped because today's quote is structurally meaningless as a proxy.

Regression: `tests/test_streamlit/test_backtest_ui.py::TestRefreshGating`.

### Watchlist Tab

A read-only analyst surface for multi-timeframe regime monitoring. Add tickers via the input row, persist them across restarts (`knowledge_base/watchlist.json`, atomic temp+rename writes), and view each ticker's regime / ADX / IV-rank across five timeframes simultaneously.

**Multi-timeframe regime parity.** `multi_tf_regime.classify_multi_tf` reuses `RegimeClassifier._determine_regime` — the same pure rule the live agent uses on daily bars — fed intraday bars at `1d / 4h / 1h / 15m / 5m`. There is no fork of regime logic. SMA windows scale per timeframe (`(50, 200)` for 1d, `(20, 50)` for intraday). The "TF agree" column is the share of timeframes whose trend matches each ticker's longest interval — 100% means fully aligned across the stack.

**Hybrid intraday data path.** `MarketDataProvider.fetch_intraday_bars(ticker, interval)` pulls history from yfinance (chart depth: 5m/15m/30m capped at 60 days; 4h synthesised via `df.resample("4h")` from 60m) and overlays the right-most live tick from the Alpaca snapshot when available. Cached for 60s by `(ticker, interval)` so a Streamlit rerun within the refresh window is free.

**Chart panel.** A 4-row Plotly subplot stack (Price · Volume · Oscillators · Trend) with collapsible rows. Indicators include SMA-50/200, Bollinger Bands, ATR bands, full Ichimoku Kinkō Hyō (Tenkan / Kijun / Cloud), RSI(14), Stoch RSI, MACD(12,26,9), and ADX(14). All indicators are pure pandas/numpy in `watchlist_chart.py` — no `pandas-ta`, no `TA-Lib`, no `numba` (the latter has no Python 3.13+ wheels yet, which broke an earlier prototype). The ADX line on the chart and the strength badge in the table use the same `_adx_series` math so they cannot drift.

**Architectural safety.** `watchlist_ui.py` and `watchlist_chart.py` import only `market_data`, `multi_tf_regime`, `regime`, `watchlist_store`, plus `streamlit` / `plotly`. They explicitly **do not import** `decision_engine`, `chain_scanner`, `executor`, or `risk_manager` — the watchlist is a display surface and cannot influence trade decisions even by accident.

**Refresh model.** `@st.cache_data(ttl=WATCHLIST_REFRESH_SECS)` (default 60s) keyed on `(ticker, intervals_tuple, refresh_token)`. The `↻ Refresh` button bumps the token for immediate invalidation; otherwise the cache self-expires every minute so yfinance doesn't get rate-limited.

Regression: `tests/test_market_data.py::TestFetchIntradayBars`, `tests/test_multi_tf_regime.py`, `tests/test_watchlist_store.py`, `tests/test_watchlist_chart.py`.

---

## Setup & Configuration

### 1. Install

```bash
pip install -r requirements.txt

# optional sentiment-pipeline deps (each degrades gracefully if absent)
pip install praw>=7.7.0        # Reddit
pip install tweepy>=4.14.0     # Twitter / X
pip install anthropic>=0.40.0  # Claude verifier
```

### 2. Configure `.env`

A minimal `.env` is shown in [Quickstart](#quickstart). The full reference follows.

#### Core trading

| Variable | Default | Description |
|---|---|---|
| `TICKERS` | `SPY,QQQ` | Comma-separated underlyings |
| `DRY_RUN` | `true` | Log plans; don't submit orders |
| `MODE` | `dry_run` | `live` or `dry_run` |
| `MAX_RISK_PCT` | `0.02` | Max loss per trade as fraction of equity |
| `MIN_CREDIT_RATIO` | `0.33` | Minimum credit / spread width (static mode only) |
| `MAX_DELTA` | `0.20` | Max abs delta of sold strike |
| `EDGE_BUFFER` | `0.10` | Adaptive C/W margin over breakeven; required `C/W = |Δshort| × (1 + edge_buffer)` |
| `SCAN_MODE` | `adaptive` | `adaptive` activates the scanner + Δ-aware C/W floor; `static` falls back to legacy planner |
| `DAILY_DRAWDOWN_LIMIT` | `0.05` | Kill process if equity drops > N % in one day |
| `MAX_BUYING_POWER_PCT` | `0.80` | Enter Liquidation Mode at > N % BP used |
| `LIQUIDITY_MAX_SPREAD` | `0.05` | Absolute floor of underlying bid/ask gate ($) |
| `LIQUIDITY_BPS_OF_MID` | `0.0005` | Slope of bid/ask gate (5 bps × mid). Effective threshold = `max(LIQUIDITY_MAX_SPREAD, LIQUIDITY_BPS_OF_MID × mid)` |
| `STALE_SPREAD_PCT` | `0.01` | Stale-quote threshold; soft-passes with WARNING |
| `FORCE_MARKET_OPEN` | `false` | Bypass market-hours check (paper testing) |
| `ALPACA_STOCKS_FEED` | `iex` | `iex` (free) or `sip` (paid SIP) |
| `ALPACA_OPTIONS_FEED` | `indicative` | `indicative` (free, 15-min delayed) or `opra` (paid real-time) |
| `LOG_MAX_BYTES` | `10485760` | Per-file log rotation threshold (10 MB) |
| `LOG_BACKUP_COUNT` | `7` | Rollover files retained |

#### Streamlit dashboard

| Variable | Default | Description |
|---|---|---|
| `LIVE_MONITOR_REFRESH_SECS` | `3` | Live Monitor fragment auto-refresh tick (was 30 pre-watchdog) |
| `BROKER_STATE_TTL_SECS` | `30` | TTL on cached Alpaca account/positions/clock fetches |
| `WATCHLIST_REFRESH_SECS` | `60` | TTL on per-ticker multi-timeframe classification cache (Watchlist tab) |
| `WATCHDOG_DISABLE` | `0` | Set to `1` to disable the journal `Observer` (cache keys go to mtime+size only) |
| `WATCHDOG_FORCE_POLLING` | `0` | Set to `1` for `PollingObserver` on NFS / network mounts where inotify is unavailable |

#### Core Intelligence (analyst)

| Variable | Default | Description |
|---|---|---|
| `LLM_ENABLED` | `false` | Master switch for the LLM intelligence layer |
| `LLM_PROVIDER` | `ollama` | `ollama`, `lmstudio`, `openai`, `anthropic` |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM API endpoint |
| `LLM_MODEL` | `mistral` | Primary reasoning model |
| `LLM_EMBEDDING_MODEL` | `nomic-embed-text` | Embeddings model for RAG |
| `LLM_TEMPERATURE` | `0.3` | Analyst sampling temperature |
| `LLM_MAX_TOKENS` | `2048` | Analyst response cap |
| `LLM_TIMEOUT` | `60` | Analyst HTTP timeout (s) |
| `TRADE_JOURNAL_DIR` | `trade_journal` | Trade lifecycle logs |
| `KNOWLEDGE_BASE_DIR` | `knowledge_base` | RAG vector store |

All three LLM callers (analyst, FinGPT, verifier) share the same `make_llm_client(role, cfg)` factory so their parameters live in one place.

#### Sentiment pipeline

| Variable | Default | Description |
|---|---|---|
| `FINGPT_ENABLED` | `false` | Enable sentiment pipeline |
| `FINGPT_MODEL` | `qwen2.5-trading` | Ollama model for FinGPT analysis |
| `FINGPT_NEWS_LIMIT` | `10` | Max headlines from yfinance fallback |
| `FINGPT_CACHE_TTL` | `300` | FinGPT in-process cache TTL (s) |
| `FINGPT_TEMPERATURE` | `0.1` | Keep deterministic |
| `FINGPT_MAX_TOKENS` | `512` | Short JSON cap |
| `FINGPT_TIMEOUT` | `45` | HTTP timeout (s) |
| `NEWS_SOURCES` | `yahoo,sec_edgar,fed_rss` | Comma-separated source keys |
| `NEWS_LOOKBACK_HOURS` | `24` | How far back to fetch news |
| `NEWS_MAX_ITEMS_PER_SOURCE` | `20` | Items per source per cycle |
| `NEWS_CACHE_TTL` | `240` | Per-`(ticker, source)` cache TTL (s) |
| `NEWS_SOURCE_WEIGHTS_JSON` | _(empty)_ | JSON object overriding `DEFAULT_SOURCE_WEIGHTS` |
| `REDDIT_CLIENT_ID` / `_SECRET` | _(empty)_ | PRAW credentials — enables all Reddit sources |
| `REDDIT_USER_AGENT` | `TradingAgent/1.0` | PRAW user agent |
| `TWITTER_BEARER_TOKEN` | _(empty)_ | Twitter API v2 Bearer token |
| `VERIFIER_ENABLED` | `false` | Enable reasoning-model verification |
| `VERIFIER_PROVIDER` | `ollama` | `ollama` (local) or `anthropic` (cloud) |
| `VERIFIER_MODEL` | `qwq:32b` | Verifier model |
| `VERIFIER_API_KEY` | _(empty)_ | Anthropic key when `VERIFIER_PROVIDER=anthropic` |
| `VERIFIER_TEMPERATURE` | `0.15` | Low but non-zero — reasoning models benefit |
| `VERIFIER_MAX_TOKENS` | `2048` | Response cap |
| `VERIFIER_TIMEOUT` | `90` | HTTP timeout (s) — reasoning is slower |
| `EARNINGS_CALENDAR_ENABLED` | `true` | Tier-0 short-circuit |
| `EARNINGS_CALENDAR_LOOKAHEAD_DAYS` | `7` | Tier-0 firing window |
| `EARNINGS_CALENDAR_REFRESH_HOURS` | `12` | Per-ticker earnings cache freshness |
| `SENTIMENT_HASH_CACHE_SIZE` | `32` | Tier-1 LRU cap; TTL auto-scales to `max(NEWS_CACHE_TTL, FINGPT_CACHE_TTL)` |

### 3. Run

```bash
python -m trading_agent.agent              # paper trading
python -m trading_agent.agent --dry-run    # log plans, no orders
python -m trading_agent.agent --env /path/to/.env

# tests
python run_tests.py                        # full repo suite
pytest tests/ -v                           # equivalent
```

After-hours: exits cleanly before 9:25 AM ET, after 4:05 PM ET, and on weekends. Override with `FORCE_MARKET_OPEN=true`.

---

## Data Sources

| Source | Purpose | Auth |
|---|---|---|
| Yahoo Finance | Regime detection (SMA / RSI / BB), backtest history, `^VIX` | None |
| Alpaca Market Data | Real-time snapshots, option chains, Greeks | API key |
| Alpaca Paper API | Order execution, account equity, market clock | API key |
| SEC EDGAR | 8-K / 10-Q (sentiment pipeline) | None |
| Federal Reserve RSS | FOMC statements (sentiment pipeline) | None |
| Reddit | r/wsb, r/stocks, r/options (sentiment pipeline) | PRAW |
| Twitter / X | Cashtag stream (sentiment pipeline) | Bearer token |

---

## Project Structure

```
trading-agent/
├── .env                              # API keys + config (not committed)
├── requirements.txt
├── README.md
├── architecture_diagram.html
├── setup_intelligence.sh             # Ollama setup helper
├── run_tests.py
│
├── trading_agent/
│   ├── agent.py                      # Orchestrator: two-stage cycle, timeout guard, sentiment pipeline
│   ├── config.py                     # AppConfig + IntelligenceConfig
│   ├── ports.py                      # Hexagonal protocols: MarketDataPort, BrokerPort, SentimentReadout
│   ├── market_profile.py             # MarketProfile (TZ, session bounds, trading-day oracle)
│   ├── logger_setup.py
│   │
│   │   # ── Core Phases ──
│   ├── market_data.py                # Phase I — yfinance + Alpaca (TTL cache, parallel, split timeouts, fetch_intraday_bars)
│   ├── regime.py                     # Phase II — SMA / RSI / Bollinger / VIX-z
│   ├── multi_tf_regime.py            # Multi-timeframe regime wrapper (reuses _determine_regime, no shadow scorer)
│   ├── strategy.py                   # Phase III — strike selection, nearest-Friday DTE
│   ├── chain_scanner.py              # Phase III — adaptive (Δ × DTE × width) sweep
│   ├── decision_engine.py            # Pure scoring entrypoint shared by live + backtest
│   ├── calendar_utils.py             # NYSE trading-day oracle (lazy lru_cache)
│   ├── strategy_presets.py           # Conservative / Balanced / Aggressive presets
│   ├── risk_manager.py               # Phase IV — 8-guardrail validator
│   ├── executor.py                   # Phase VI — mleg execution + HTML report
│   ├── trade_plan_report.py
│   ├── watchlist_store.py            # Persistent JSON watchlist (atomic writes, RLock for nested CRUD)
│   │
│   │   # ── Position Management ──
│   ├── position_monitor.py
│   ├── order_tracker.py
│   │
│   │   # ── Core Intelligence ──
│   ├── journal_kb.py                 # Always-on signal logger (live | backtest split)
│   ├── trade_journal.py              # Full-lifecycle trade logging
│   ├── knowledge_base.py             # File-based RAG vector store
│   ├── llm_client.py                 # OpenAI-compatible client + make_llm_client(role) factory
│   ├── llm_analyst.py                # Pre/post-trade LLM analysis
│   ├── fine_tuning.py                # Training data export (JSONL / Alpaca / DPO)
│   │
│   │   # ── Sentiment Pipeline ──
│   ├── sentiment_pipeline.py         # Tier-0/1/2 facade, cycle-scoped pool
│   ├── earnings_calendar.py          # Tier-0
│   ├── sentiment_cache.py            # Tier-1
│   ├── news_aggregator.py            # Tier-2 — NewsItem + NewsAggregator
│   ├── fingpt_analyser.py            # Tier-2 — FinGPT specialist
│   ├── sentiment_verifier.py         # Tier-2 — Reasoning verifier
│   │
│   └── streamlit/
│       ├── app.py                    # 4-tab dashboard entrypoint
│       ├── live_monitor.py           # Live tab — broker-gated, watchdog-driven refresh
│       ├── backtest_ui.py            # Backtest tab — Backtester + unified-engine toggle
│       ├── llm_extension.py          # LLM tab — RAG over signals_live.jsonl
│       ├── watchlist_ui.py           # Watchlist tab — multi-tf regime table + macro strip
│       ├── watchlist_chart.py        # Watchlist tab — 4-row Plotly chart, pure-pandas indicators
│       ├── file_watcher.py           # Per-process Observer + version counters
│       └── components.py
│
├── trade_journal/                    # Auto-created
│   ├── trades/
│   ├── index.json
│   ├── stats.json
│   ├── signals_live.jsonl            # Always-on live-mode signal log
│   ├── signals_backtest.jsonl        # Backtest-mode signal log (deliberately separate)
│   └── signals_*.md                  # Human-readable mirrors
│
├── trade_plans/                      # Per-ticker persistent trade-plan files
├── knowledge_base/                   # RAG vector store (LLM layer) + watchlist.json (Watchlist tab)
└── logs/
```

---

## Signal Journal Format

Live cycles append one JSON object per line to `trade_journal/signals_live.jsonl`; backtests append to `trade_journal/signals_backtest.jsonl`. The two files use identical schema. When the sentiment pipeline is active each record also carries `fingpt_sentiment`, `fingpt_event_risk`, `fingpt_recommendation`, `fingpt_agreement`, `fingpt_hallucination_flags`, and `fingpt_verified_by`.

**Action values:** `dry_run`, `submitted`, `rejected`, `skipped_by_llm`, `skipped_existing`, `skipped_liquidation_mode`, `skipped_bias`, `skipped`, `error`, `cycle_timeout`, `daily_drawdown_circuit_breaker`.

### `scan_results` block (adaptive scanner only)

When the active preset is in adaptive mode, every `plan()` invocation that runs the chain scanner also writes a `raw_signal.scan_results` block — the single source of truth for *why* the scanner picked / rejected each ticker (populated even when zero candidates pass).

```jsonc
"scan_results": {
  "scan_mode":        "adaptive",
  "side":             "bull_put",
  "edge_buffer":      0.10,
  "min_pop":          0.55,
  "candidates_total": 0,
  "selected_index":   -1,
  "top_k":            [],
  "diagnostics": {
    "grid_points_total":    16,
    "grid_points_priced":   12,
    "expirations_resolved": 4,
    "rejects_by_reason":    {"cw_below_floor": 11, "no_long_contract": 1},
    "best_near_miss": {
      "expiration":   "2026-05-15",
      "dte":          14,
      "short_strike": 590.0,
      "long_strike":  585.0,
      "short_delta":  -0.20,
      "credit":       0.95,
      "width":        5.0,
      "cw_ratio":     0.19,
      "cw_floor":     0.22,
      "pop":          0.80,
      "ev":          -0.04
    }
  }
}
```

**How to read it.** When `candidates_total == 0` the answer to *"why didn't the scanner trade?"* is in two fields: `rejects_by_reason` (which filter dominated) and `best_near_miss` (how close the chain came). If `cw_ratio` is close to `cw_floor`, one click of `EDGE_BUFFER` toward zero unblocks the trade; if they're far apart, the chain isn't paying enough — wait or skip.

**Reject-reason taxonomy** (stable string keys):

| Key | Meaning |
|---|---|
| `no_chain` | `fetch_option_chain()` returned empty for that expiration |
| `no_short_contract` | No contract matches the target Δ |
| `no_long_contract` | No protective strike at requested width (grid too sparse) |
| `non_positive_width` | Snapped width came out as 0 or negative |
| `dte_non_positive` | Resolved expiration is today or earlier |
| `pop_below_min` | `1 − |Δshort| < min_pop` (Δ-target grid too aggressive) |
| `credit_non_positive` | `bid_short − ask_long ≤ 0` |
| `credit_ge_width` | Credit ≥ width — would be a debit, not a credit |
| `cw_below_floor` | `C/W < |Δ| × (1 + edge_buffer)` — most common in thin-premium regimes |
