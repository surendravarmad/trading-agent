# Trading Agent — Project Context Briefing

> Paste or attach this file at the start of a fresh LLM session. It captures
> the project's architecture, invariants, recent changes, and conventions so
> the assistant can produce changes that don't drift from the codebase's
> design intent.

---

## 1. What this project is

An autonomous options credit-spread trading agent. Primary goal is **capital
preservation** — every trade has a known, capped maximum loss. The agent
runs a 5-minute cycle that scans option chains, picks the highest-EV
risk-defined credit spread, and submits a multi-leg order via the Alpaca
paper-trading API.

The same scoring logic powers two surfaces:

- **Live agent** — `python -m trading_agent.agent` runs one cycle (or a
  loop). Designed to be cron-driven every 5 minutes during market hours.
- **Backtester** — runs from the Streamlit UI; replays historical chains
  through the same decision engine the live agent uses (when
  `use_unified_engine=True`).

Repo root: `trading-agent/`. Python 3.11+ is the support floor.

---

## 2. Hard invariants (DO NOT BREAK)

These are enforced by AST checks in CI; violating any of them is a
revertable change.

### 2.1 Single C/W floor formula

The credit-to-width floor `|Δshort| × (1 + edge_buffer)` must appear
identically in three files:

- `trading_agent/chain_scanner.py` (candidate filtering)
- `trading_agent/risk_manager.py` (final gate, in adaptive mode)
- `trading_agent/executor.py` (live-quote re-validation)

The static `MIN_CREDIT_RATIO=0.33` is only used when adaptive mode is
disabled. **Never** introduce a fourth copy of the floor; never deviate
the formula on one side without the other two.

### 2.2 Single source of scoring

The scoring helpers `_score_candidate`, `_score_candidate_with_reason`,
and `_quote_credit` may **only be defined** in
`trading_agent/chain_scanner.py` and `trading_agent/decision_engine.py`.
A definition anywhere else is a "shadow scorer" — the AST check
(`scripts/checks/scan_invariant_check.py`) fails CI on detection.

### 2.3 Backtester must wire through `decide()`

`trading_agent/streamlit/backtest_ui.py` must contain at least one call
to `decide(...)` — the imported `decision_engine.decide` entrypoint. If
that call disappears the unified path becomes dead code and the
backtester silently regresses to its old σ-distance heuristic.

### 2.4 Journal-file split

- Live cycles append to `trade_journal/signals_live.jsonl`
- Backtests append to `trade_journal/signals_backtest.jsonl`
- The pre-May-2026 `trade_journal/signals.jsonl` is no longer written
  by either path; it survives only as a back-compat read fallback in
  `visualize_logs.py` and `live_monitor.py`/`llm_extension.py` loaders.
- `JournalKB.__init__(run_mode=...)` rejects unknown modes with
  `ValueError` so a typo can't silently fork a third file.

The LLM analyst corpus and the live-monitor diagnostics deliberately
read **only** the live file so synthetic backtest counterfactuals can't
bias guardrail recommendations.

---

## 3. Architecture overview

### 3.1 Two-stage cycle (`agent.py`)

```
STAGE 1 — Monitor open positions
  PositionMonitor → ExitSignalCheck → OrderTracker
  (50 % stop-loss, 75 % profit, regime shift, DTE safety)

STAGE 2 — Open new positions (per ticker)
  I·Perceive → II·Classify → III·Plan → IV·Risk → V·LLM → VI·Execute
  yfinance     SMA / RSI     ChainScanner   8 guardrails
  Alpaca       Bollinger     decision_      buying-power
  snapshots    VIX-z gate    engine.decide  daily-DD breaker
```

The Sentiment Pipeline runs concurrently in a single-worker
`ThreadPoolExecutor` owned by the cycle's `with pipeline:` block — every
Future belongs to exactly one cycle. The Future is cancelled if Phase
III/IV reject the trade (no wasted local-LLM calls).

### 3.2 Decision-engine unification (May 2026)

```
chain_scanner.py            ← pure helpers (_quote_credit, _score_candidate_with_reason)
        │
        ▼
decision_engine.decide()    ← pure scoring entrypoint (no I/O)
        │
        ├──── ChainScanner.scan()                          (live)
        └──── Backtester._build_alpaca_plan_via_decide()   (backtest)
```

`decision_engine.decide(DecisionInput) → DecisionOutput` is pure: no
I/O, no calendar lookups, no broker calls. It takes `ChainSlice`s in
(one expiration's `{strike, delta, bid, ask, symbol}` dicts plus the
DTE), runs the full `(Δ × width)` sweep, and returns ranked
`SpreadCandidate`s plus a `ScanDiagnostics` block.

### 3.3 Strategy selection priority

| Priority | Regime | Trigger | Strategy |
|---|---|---|---|
| 1 | Mean Reversion | 3-σ Bollinger touch | MR Spread |
| 2 | VIX inhibit | `vix_z > +2 σ` AND regime ∈ {Bullish, Sideways} | Bear Call (demoted) |
| 3 | Bullish + Lead-z | Bullish AND `leadership_z > +1.5 σ` | Bull Put (leadership bias) |
| 4 | Sideways + Lead-z | Sideways AND `leadership_z > +1.5 σ` | Bull Put (leadership bias) |
| 5 | Bullish | Price > SMA-200 AND SMA-50 slope > 0 | Bull Put |
| 6 | Bearish | Price < SMA-200 AND SMA-50 slope < 0 | Bear Call |
| 7 | Sideways | Between SMAs / narrow Bollinger | Iron Condor |

Mean-reversion bypasses the VIX gate (the band touch already encodes
the volatility condition).

### 3.4 Risk guardrails (8 checks)

1. Plan validity
2. C/W ≥ `|Δshort| × (1 + edge_buffer)` (adaptive) or `MIN_CREDIT_RATIO=0.33` (static)
3. Sold delta ≤ `MAX_DELTA` (default 0.20)
4. Max loss ≤ `MAX_RISK_PCT × equity` (default 2 %)
5. Account type = paper
6. Market hours
7. Underlying liquidity: spread < `max(LIQUIDITY_MAX_SPREAD, LIQUIDITY_BPS_OF_MID × mid)`
8. Buying power ≥ `(1 − MAX_BUYING_POWER_PCT) × equity` (default 80 %)

Plus: daily-drawdown circuit breaker (5 %); liquidation mode (skip Stage
2 above 80 % BP); macro guard (skip Bull Put when price < SMA-200);
high-IV block (skip all entries when realized-vol IV-rank > 95th
percentile); 3-cycle exit debounce (bypassed by HARD_STOP /
STRIKE_PROXIMITY / DTE_SAFETY).

### 3.5 Adaptive spread width

```
width = max(SPREAD_WIDTH_FLOOR, 3 × strike_grid_step, 0.025 × spot)
        snapped UP to the strike grid
```

So a $80 ticker gets $5 (floor wins), SPY/QQQ at $700 gets $15-20
(spot-percentage wins). Legacy `SPREAD_WIDTH = $5` is now a **hard
floor**, never a target. Applied to all four spread types.

### 3.6 Score formula

```
POP         ≈ 1 − |Δshort|
C/W         = credit / width
EV/$risked  = (POP × C/W − (1 − POP) × (1 − C/W)) / (1 − C/W)
annualized  = EV/$risked × (365 / DTE)
```

Hard pre-scoring filters: `POP ≥ min_pop` (0.55), `C/W ≥ |Δshort| × (1 + edge_buffer)` (default `edge_buffer = 0.10`), positive net credit, both legs quoting non-zero bid/ask. Annualized score breaks ties.

### 3.7 Credit pricing — `_quote_credit`

`short_mid − long_mid − fill_haircut`, where each leg's mid is
`(bid+ask)/2` when both are positive and the conservative side
(short→bid, long→ask) when a quote is missing. Default
`fill_haircut = $0.02` matches the executor's per-leg slippage budget so
scored credit and targeted-fill limit price stay in sync. Worst-case
`short_bid − long_ask` is **no longer used for scoring**.

---

## 4. Streamlit dashboard

Four tabs, all under `trading_agent/streamlit/`:

| Tab | File | Purpose |
|---|---|---|
| 📡 Live Monitoring | `live_monitor.py` | Start/Stop agent, equity, P&L, regime, positions, journal |
| 📊 Backtesting | `backtest_ui.py` | Date range, multi-ticker, **Unified Decision Engine** toggle, P&L charts |
| 🤖 LLM Extension | `llm_extension.py` | Chat with local Ollama model (RAG over `signals_live.jsonl`) |
| 📊 Watchlist | `watchlist_ui.py` + `watchlist_chart.py` | Persistent ticker watchlist, multi-tf regime table (1d/4h/1h/15m/5m), 4-row Plotly chart. **Read-only — must not import `decision_engine`, `chain_scanner`, `executor`, `risk_manager`** (architectural invariant; the watchlist is a display surface only) |

### 4.1 Refresh model (watchdog)

A single per-process `Observer` (wrapped in `@st.cache_resource`)
watches `trade_journal/` and `trade_plans/`. Loaders cache by
`(version, mtime, size)`:

```python
@st.cache_data
def _parse_journal_df(path: str, version: int, mtime: float, size: int) -> pd.DataFrame:
    ...
```

Unrelated reruns hit the cache (zero I/O); only real journal writes
invalidate. Implementation lives in `trading_agent/streamlit/file_watcher.py`.

**Kill switches** (env vars):
- `WATCHDOG_DISABLE=1` — fall back to mtime+size polling only
- `WATCHDOG_FORCE_POLLING=1` — use `PollingObserver` for NFS / network mounts (~1 s resolution)
- `LIVE_MONITOR_REFRESH_SECS` (default 3) — fragment auto-refresh tick
- `BROKER_STATE_TTL_SECS` (default 30) — TTL on Alpaca account/positions/clock fetches

### 4.2 Broker-state gating

Alpaca account / positions / clock fetches **only run when the agent
loop is active** (sentinel: `AGENT_RUNNING` file at repo root).
Opening Streamlit alone makes zero broker calls. A manual `↻ Refresh
broker state` button is shown when the loop is stopped, for ad-hoc
inspection. This was added because pre-fix the position monitor logged
"Fetched N positions / Grouped into N spread(s)" every ~3 seconds even
when the agent wasn't running.

### 4.3 Watchlist tab (May 2026)

`watchlist_ui.py` registers as the 4th tab and lazy-imports `watchlist_chart.py` only when a user opens it (keeps cold-start light).

**Multi-timeframe regime parity invariant.** `multi_tf_regime.classify_multi_tf` reuses `RegimeClassifier._determine_regime` — the same pure rule the live agent uses on daily bars — fed intraday bars. There is **no fork** of regime logic. The "1d" timeframe is delegated directly to the existing `RegimeClassifier` (single source of truth). A regression test (`tests/test_multi_tf_regime.py::TestNoShadowScorer`) AST-walks the module and rejects any new `_determine_regime`-shaped function.

**Hybrid intraday data path.** `MarketDataProvider.fetch_intraday_bars(ticker, interval, lookback_days=None, include_live_overlay=True)` pulls history from yfinance (5m/15m/30m capped at 60 days; 4h synthesised via `df.resample("4h")` from 60m) and overlays the right-most live tick from the Alpaca snapshot. Cached for `INTRADAY_BARS_TTL=60s` by `(ticker, interval)`.

**Indicator stack — pure pandas.** `watchlist_chart.py` implements MACD, Stoch RSI, ATR bands, Ichimoku, plus the existing SMA/RSI/Bollinger/ADX in ~80 lines of pandas/numpy. **Do not re-introduce `pandas-ta`** — its `numba` transitive dep has no Python 3.13+ wheels yet (source build needs LLVM, fails on most user machines). `TA-Lib` and `VectorBT` were also rejected: TA-Lib needs a system C library install (worse portability), VectorBT has the same numba problem and is aimed at backtesting not chart rendering. The pure-pandas approach is performance-adequate (chart render dominated by Plotly serialisation, not indicator math) and dependency-clean. The ADX line on the chart and the strength badge in the table both use the same `_adx_series` math so they cannot drift.

**Persistence.** `watchlist_store.py` writes `knowledge_base/watchlist.json` atomically (`tmp + os.replace`). Uses `threading.RLock` (not `Lock`) because `add_ticker → save_watchlist` would otherwise self-deadlock on the nested acquire.

**Refresh model.** `@st.cache_data(ttl=WATCHLIST_REFRESH_SECS)` (default 60s) keyed on `(ticker, intervals_tuple, refresh_token)`. Manual `↻ Refresh` button bumps the token for immediate invalidation.

**Architectural firewall.** `watchlist_ui.py` and `watchlist_chart.py` import only `market_data`, `multi_tf_regime`, `regime`, `watchlist_store`, plus `streamlit`/`plotly`. They explicitly do **not** import `decision_engine`, `chain_scanner`, `executor`, or `risk_manager`. If a future PR needs to break this, it requires explicit sign-off — the read-only invariant is what makes the tab safe to hand to a less-trusted analyst.

### 4.4 Strategy presets

The Live tab has a **Strategy Profile** expander writing to
`STRATEGY_PRESET.json` (atomic temp+rename). Three built-in profiles
plus Custom — Conservative / Balanced (default) / Aggressive — each
controlling Δ-short, DTE per strategy, width policy, C/W floor. The
agent re-reads the file at the start of every cycle (no restart).

`agent.__init__` calls `load_active_preset()` and forwards knobs into
`StrategyPlanner` + `RiskManager`. `strategy._pick_expiration(kind=...)`
honours the per-strategy DTE override only when passed (preserves
legacy unit-test behaviour). `strategy._pick_spread_width` honours
`width_mode = "pct_of_spot" | "fixed_dollar"`.

A `directional_bias` filter (`auto` / `bullish_only` / `bearish_only`
/ `neutral_only`) fires immediately after Phase II classify so
disallowed regimes short-circuit before sentiment / chain fetch.
Mean-reversion is always allowed regardless of bias.

---

## 5. File / directory layout

```
trading-agent/
├── .env                              # API keys + config (NEVER committed)
├── .gitignore                        # sectioned: secrets / build / IDE / runtime / agent-state
├── .github/workflows/ci.yml          # AST invariants → pytest → smoke checks
├── README.md                         # 591 lines, TOC at top
├── PROJECT_CONTEXT.md                # ← THIS FILE
├── architecture_diagram.html         # interactive React diagram
├── requirements.txt                  # includes watchdog>=3,<5
├── run_tests.py                      # full repo unittest entrypoint
├── run_chain_scanner_tests.py        # 40-case adaptive scanner suite
├── run_risk_manager_quick.py         # static vs adaptive C/W smoke
├── visualize_logs.py                 # one-shot HTML report (signals_live.jsonl by default; legacy fallback)
│
├── trading_agent/
│   ├── agent.py                      # Orchestrator (two-stage cycle, timeout guard, sentiment pipeline)
│   ├── config.py                     # AppConfig + IntelligenceConfig
│   ├── ports.py                      # Hexagonal protocols (MarketDataPort, BrokerPort, SentimentReadout)
│   ├── market_profile.py             # MarketProfile (TZ, session bounds)
│   ├── logger_setup.py               # idempotent setup (sentinel-tagged handlers)
│   │
│   │   # — Core phases —
│   ├── market_data.py                # Phase I — yfinance + Alpaca with TTL cache, parallel, split timeouts, fetch_intraday_bars
│   ├── regime.py                     # Phase II — SMA / RSI / Bollinger / VIX-z classifier
│   ├── multi_tf_regime.py            # Multi-tf wrapper — reuses _determine_regime, no shadow scorer
│   ├── strategy.py                   # Phase III — strike selection, nearest-Friday DTE
│   ├── chain_scanner.py              # Phase III — adaptive (Δ × DTE × width) sweep [SCORING SOURCE]
│   ├── decision_engine.py            # Pure scoring entrypoint (live + backtest delegate here)
│   ├── calendar_utils.py             # NYSE oracle (lazy lru_cache singleton)
│   ├── strategy_presets.py           # PresetConfig + Conservative/Balanced/Aggressive
│   ├── risk_manager.py               # Phase IV — 8-guardrail validator
│   ├── executor.py                   # Phase VI — mleg execution + HTML report + live-quote refresh
│   ├── trade_plan_report.py          # Per-ticker HTML
│   ├── watchlist_store.py            # Persistent JSON watchlist (atomic writes, RLock for nested CRUD)
│   │
│   │   # — Position management —
│   ├── position_monitor.py           # Stage 1 — fetch + group spreads (DEBUG-level on hot path)
│   ├── order_tracker.py
│   │
│   │   # — Core intelligence —
│   ├── journal_kb.py                 # Always-on logger; run_mode={"live","backtest"}
│   ├── trade_journal.py              # Full-lifecycle TradeEntry
│   ├── knowledge_base.py             # File-based RAG vector store
│   ├── llm_client.py                 # OpenAI-compatible client + make_llm_client(role, cfg) factory
│   ├── llm_analyst.py                # Pre/post-trade analysis (advisory only)
│   ├── fine_tuning.py                # Chat JSONL / Alpaca / DPO export
│   │
│   │   # — Sentiment pipeline (Tier 0/1/2) —
│   ├── sentiment_pipeline.py         # Facade, cycle-scoped pool, gating
│   ├── earnings_calendar.py          # Tier-0 — yfinance authoritative
│   ├── sentiment_cache.py            # Tier-1 — SHA-1 content-hash gate
│   ├── news_aggregator.py            # Tier-2 — NewsItem + multi-source aggregator
│   ├── fingpt_analyser.py            # Tier-2 — FinGPT specialist (SentimentReport)
│   ├── sentiment_verifier.py         # Tier-2 — reasoning verifier (VerifiedSentimentReport)
│   │
│   └── streamlit/
│       ├── app.py                    # 4-tab entrypoint
│       ├── live_monitor.py           # Live tab (broker-gated, watchdog-driven)
│       ├── backtest_ui.py            # Backtest tab (Backtester + unified-engine toggle)
│       ├── llm_extension.py          # LLM tab (RAG over signals_live.jsonl)
│       ├── watchlist_ui.py           # Watchlist tab — multi-tf regime table + macro strip
│       ├── watchlist_chart.py        # Watchlist tab — 4-row Plotly chart, pure-pandas indicators
│       ├── file_watcher.py           # Per-process Observer + version counters
│       └── components.py
│
├── scripts/checks/                   # Drift-prevention smoke tests (CI)
│   ├── README.md                     # When-X-fails troubleshooting
│   ├── scan_invariant_check.py       # AST: floor formula + scoring source + decide() wired
│   ├── run_scan_diagnostics_check.py
│   ├── run_unified_backtest_check.py
│   ├── run_journal_split_check.py
│   └── run_live_vs_backtest_parity_check.py
│
├── tests/                            # pytest suite
│
├── trade_journal/                    # AUTO-CREATED, .gitignored
│   ├── signals_live.jsonl            # Live cycles
│   ├── signals_backtest.jsonl        # Backtest runs
│   └── trades/index.json/stats.json
│
├── trade_plans/                      # AUTO-CREATED, .gitignored
├── knowledge_base/                   # RAG store, .gitignored
├── logs/                             # Rotating logs (10 MB × 7 default), .gitignored
│
├── AGENT_LOG                         # Streamlit-captured stdout, .gitignored
├── AGENT_RUNNING                     # Sentinel file ("agent loop is active"), .gitignored
└── STRATEGY_PRESET.json              # Atomic write from Streamlit, re-read every cycle, .gitignored
```

---

## 6. Conventions

### 6.1 Logging

- **INFO** is for once-per-cycle headlines + actionable events. Keep INFO
  clean enough that an operator can spot anomalies at a glance.
- **DEBUG** is for per-ticker / per-fetch chatter that fires multiple
  times per cycle. Recent demotions: `position_monitor.py`'s "Fetched N
  positions / Grouped into N spread(s)" pair; `market_data.py`'s
  per-ticker price-history fetch, batch snapshot summary, per-ticker
  Alpaca real-time quote, per-(ticker, expiration) chain fetch.
- **WARNING** for soft-passes / fallback paths that succeeded. Keep at
  WARNING — operator wants to see "we recovered from X".
- **ERROR** for failures that affect this cycle's outcome.

When demoting INFO → DEBUG, leave a comment explaining *why* (typically
"hot-path; fires N×/cycle; cycle-summary INFO already conveys aggregate
progress"). This prevents the next contributor from re-promoting it.

### 6.2 Caching

- `@st.cache_resource` — **per-process singletons** (the watchdog
  `Observer`, e.g.). Never returns stale objects because there's only
  ever one.
- `@st.cache_data` — **value caches keyed by arguments**. Used for
  journal parsing (`(path, version, mtime, size)`) and broker-state
  fetches (`ttl=BROKER_STATE_TTL_SECS`). The watchdog `version` counter
  is the *primary* invalidation signal; `mtime` + `size` are
  belt-and-suspenders for when watchdog is disabled.
- `make_llm_client(role, cfg)` is the **only** place LLM client params
  live. All three callers (analyst, FinGPT, verifier) share it — there
  are no per-module hard-coded `LLMConfig` blocks.

### 6.3 Streamlit widget+session_state collisions

Don't pass both `value=` (or `default=`) **and** `key=` to a widget when
the session-state slot for that key has been pre-seeded elsewhere.
Streamlit raises a "widget with key X was created with a default value
but also had its value set via the Session State API" warning. The fix
is to **drop the default** and rely on the session-state seeding code.

### 6.4 Test paths

The smoke checks now live in `scripts/checks/` and use:

```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
```

…so they resolve repo root from two directory levels deep. If you add a
new check, follow the same pattern.

### 6.5 Journal-path resolution (visualize_logs.py)

`visualize_logs.py` defaults to `signals_live.jsonl` and falls back to
the legacy `signals.jsonl` *only* when the caller asked for the default.
**Never** silently rewrite an explicit user path — that would render
wrong data on a `--journal trade_journal/signals_backtest.jsonl` call.
The behaviour is pinned by `TestResolveJournalPath` in
`tests/test_visualize_logs.py`.

---

## 7. Pitfalls / things that have bitten us

### 7.1 Backtester drift

Three drift sources are documented as "still open" — don't claim to
have fixed them without writing a regression test:

1. **Adaptive spread width** — backtester uses fixed `spread_width`; live
   scales with spot/grid. Material on $500+ underlyings.
2. **Chain-sourced δ picker** — δ is *labelled* via the analytic
   `_delta_from_sigma_distance` mapping, not read from the chain's
   listed `delta` column. Near-no-op when realized σ ≈ implied σ.
3. **Mean-Reversion priority** — Priority 1 of live `plan()` is absent
   from the backtest run-loop.
4. **Bid/ask spread modelling** — Alpaca-historical uses `close − close`;
   live fills against bid/ask. Slightly over-estimates real credit.

### 7.2 Recently closed gaps (Apr 2026) — verify in next backtest

- Friday-weekly preference in `_pick_alpaca_expiration` (penalty for
  non-Friday expirations). Regression: `TestPickAlpacaExpirationFridayPreference`.
- Credit-ratio gate now fires in alpaca-historical mode (was silently
  bypassed). Regression: `TestCreditRatioGateAlpacaHistorical`.
- Structured rejection reasons in `_alpaca_historical_plan` (returns
  `(plan, reason)` instead of `Optional[Dict]`).
- Trade-journal `expiry_date` reflects actual OCC contract expiration
  (was always `entry+1d` before).
- Expiration-fallback loop on data-availability failures (up to 3
  retries; new token `no_bars_after_fallbacks`). Regression:
  `TestAlpacaHistoricalPlanFallback`.

### 7.3 σ-horizon fix

The backtester used to project σ over the full DTE, producing intraday
strikes ~13 % OTM (|Δ| ≈ 0.02). It now projects σ over the **hold
horizon** (`hold_bars / bars_per_year`), landing strikes ~1 % OTM
(|Δ| ≈ 0.20) — apples-to-apples with live's `MIN_DELTA(0.15) ≤ |Δ| ≤ 0.20`.
Regression: `TestAlpacaHistoricalSigmaHorizon`.

### 7.4 Alpaca timeouts

Every Alpaca HTTP call uses `(connect=2s, read=10s)` (and `(2s, 15s)` for
order submission). Single `timeout=10` was stalling cycles for 10s/call
on unreachable hosts. Centralised as `ALPACA_TIMEOUT` /
`ALPACA_TIMEOUT_LONG` in `market_data.py` and reused by
`position_monitor`, `order_tracker`, `executor`.

### 7.5 SMA-50 slope units

`MarketDataProvider.sma_slope()` returns the 5-day average **dollar
change per day**, not a percentage. Logs and the LLM prompt annotate
with `$/day` so a reader can't mistake the magnitude. Downstream
consumers only read the sign.

### 7.6 Trading-library choice (locked)

The Watchlist tab's indicator stack is **pure pandas/numpy**. Three
candidates were evaluated and rejected:

- **`pandas-ta`** — `numba` transitive dep has no Python 3.13+ wheels;
  source build needs LLVM. User hit this on Python 3.14 install. Fatal
  for portability.
- **`TA-Lib`** — fastest option (10–50× over pandas), but needs a
  system-level C library install (`brew install ta-lib`, etc). Worse
  portability than what we just escaped from. The watchlist isn't
  performance-bound anyway — Plotly serialisation dominates render time
  by an order of magnitude over indicator math.
- **`VectorBT`** — same `numba` problem; also wrong tool for the job
  (built for backtesting sweeps, not chart panels).

If a future PR ever has a real performance need: **finta** (pure
Python, no compiled deps) is the safe swap; **TA-Lib** if the deploy
environment is controlled and the C-lib install is tolerable. Don't
re-introduce `pandas-ta` until/unless `numba` ships Python 3.13+
wheels and our minimum Python floor is raised.

### 7.7 `pd.NA` vs `np.nan` on float Series

`Series.replace(0, pd.NA)` on a `float64` Series **coerces dtype to
`object`** (because `pd.NA` is the masked-array NA scalar built for
nullable dtypes). After that, `.ewm().mean()` raises
`pandas.errors.DataError: No numeric types to aggregate`. **Always
use `np.nan`** when replacing into a float Series. Two fixed sites:
`multi_tf_regime.adx_strength` and `streamlit/components.py::drawdown_chart`.
Inline comments left at both call-sites so this trap doesn't get
re-introduced.

---

## 8. Commands

```bash
# Run a single cycle
python -m trading_agent.agent --dry-run            # log plans, no orders
python -m trading_agent.agent                      # paper trading
python -m trading_agent.agent --env /path/.env

# Cron schedule (5-min during US market hours)
*/5 9-16 * * 1-5 cd /path/to/trading-agent && python -m trading_agent.agent >> logs/cron.log 2>&1

# Streamlit dashboard
streamlit run trading_agent/streamlit/app.py

# Tests
python run_tests.py                                # full repo
pytest tests/ -v                                   # equivalent
python run_chain_scanner_tests.py                  # 40-case scanner suite
python run_risk_manager_quick.py                   # static vs adaptive smoke

# Drift-prevention smoke checks (CI)
python3 scripts/checks/scan_invariant_check.py
python3 scripts/checks/run_scan_diagnostics_check.py
python3 scripts/checks/run_unified_backtest_check.py
python3 scripts/checks/run_journal_split_check.py
python3 scripts/checks/run_live_vs_backtest_parity_check.py

# All five at once
for f in scripts/checks/*.py; do python3 "$f" || break; done

# Static HTML report
python visualize_logs.py                                     # today's live data
python visualize_logs.py --date 2026-04-15 --tickers SPY QQQ
python visualize_logs.py --all-dates --output reports/full_history.html
python visualize_logs.py --journal trade_journal/signals_backtest.jsonl
```

---

## 9. Environment variables (most important)

### Core trading

| Var | Default | Notes |
|---|---|---|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | — | Required |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets/v2` | |
| `TICKERS` | `SPY,QQQ` | Comma-separated |
| `DRY_RUN` | `true` | Log plans, don't submit |
| `MAX_RISK_PCT` | `0.02` | Per-trade max-loss as fraction of equity |
| `MAX_DELTA` | `0.20` | Sold-strike Δ cap |
| `EDGE_BUFFER` | `0.10` | Adaptive C/W margin: floor = `\|Δ\| × (1+edge_buffer)` |
| `SCAN_MODE` | `adaptive` | or `static` |
| `DAILY_DRAWDOWN_LIMIT` | `0.05` | Kill process at >5 % equity drop |
| `MAX_BUYING_POWER_PCT` | `0.80` | Liquidation Mode threshold |
| `ALPACA_STOCKS_FEED` | `iex` | Free tier; `sip` for paid SIP |
| `ALPACA_OPTIONS_FEED` | `indicative` | Free 15-min delayed; `opra` for paid |
| `FORCE_MARKET_OPEN` | `false` | Bypass for paper testing |

### Streamlit

| Var | Default | Notes |
|---|---|---|
| `LIVE_MONITOR_REFRESH_SECS` | `3` | Was 30 pre-watchdog |
| `BROKER_STATE_TTL_SECS` | `30` | Cache TTL on Alpaca account/positions |
| `WATCHLIST_REFRESH_SECS` | `60` | TTL on Watchlist tab's per-ticker multi-tf classification cache |
| `WATCHDOG_DISABLE` | `0` | `1` to disable Observer |
| `WATCHDOG_FORCE_POLLING` | `0` | `1` for NFS / network mounts |

### Intelligence layer (all optional)

`LLM_ENABLED`, `LLM_PROVIDER` (`ollama` / `lmstudio` / `openai` /
`anthropic`), `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TEMPERATURE`,
`LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `TRADE_JOURNAL_DIR`, `KNOWLEDGE_BASE_DIR`.

### Sentiment pipeline (all optional)

`FINGPT_ENABLED`, `FINGPT_MODEL`, `FINGPT_TEMPERATURE`,
`FINGPT_TIMEOUT`; `NEWS_SOURCES` (default `yahoo,sec_edgar,fed_rss`),
`NEWS_LOOKBACK_HOURS`, `NEWS_CACHE_TTL`, `NEWS_SOURCE_WEIGHTS_JSON`;
`REDDIT_CLIENT_ID` / `_SECRET` / `_USER_AGENT`; `TWITTER_BEARER_TOKEN`;
`VERIFIER_ENABLED`, `VERIFIER_PROVIDER` (`ollama` / `anthropic`),
`VERIFIER_MODEL` (`qwq:32b`, `deepseek-r1:32b`, `claude-sonnet-4-6`),
`VERIFIER_API_KEY`, `VERIFIER_TIMEOUT`; `EARNINGS_CALENDAR_ENABLED`,
`EARNINGS_CALENDAR_LOOKAHEAD_DAYS`; `SENTIMENT_HASH_CACHE_SIZE`.

---

## 10. Recent change log (work done in the May-2026 session)

In rough chronological order. If asked to undo or extend any of these,
verify the invariants in §2 still hold afterwards.

1. **Journal split fixed.** Tests `test_journal_kb_logs_signal_on_dry_run`
   and `test_signal_contains_thesis` updated to read `agent.journal_kb.jsonl_path`
   (canonical attribute) instead of hardcoded `signals.jsonl`. Streamlit
   tests patch both `JOURNAL_PATH` and `LEGACY_JOURNAL_PATH`.

2. **Unified Decision Engine toggle in Streamlit.** Backtest tab now
   has a "Unified Decision Engine" expander with on/off toggle and a
   preset selectbox. `_run_cached` accepts `preset_name` +
   `use_unified_engine`; resolves `PresetConfig` via
   `PRESETS.get(preset_name)`. Run caption appends
   `· unified-engine=ON (preset=<name>)` when active.

3. **Watchdog integration (Phases A+B+D).**
   - Phase A: `trading_agent/streamlit/file_watcher.py` (per-process
     Observer wrapped in `@st.cache_resource`; `_BumpHandler` increments
     version counters; kill switches `WATCHDOG_DISABLE`,
     `WATCHDOG_FORCE_POLLING`).
   - Phase B: `_load_journal_df` + `_load_recent_signals` decomposed
     into thin wrapper + `@st.cache_data`-decorated `_parse_journal_df`
     keyed by `(version, mtime, size)`.
   - Phase D: TTL cache on broker-state fetches
     (`_fetch_account_cached`, `_fetch_spreads_cached`,
     `_is_market_open_cached` with `ttl=BROKER_STATE_TTL_SECS`).
   - `REFRESH_INTERVAL` reduced from 30 → 3 s.
   - `requirements.txt`: added `watchdog>=3.0,<5`.

4. **Streamlit widget warning fix.** Removed `value=st.session_state.get("bt_start_date", default_start)` from the date_input; only `key="bt_start_date"` remains. Session-state is seeded by the timeframe-change branch on first render.

5. **Position-monitor gating.** `render_live_monitor` only fetches
   broker state when `loop_running` (sentinel: `AGENT_RUNNING` exists)
   OR a manual `↻ Refresh broker state` click sets `_bm_force_refresh`.
   Eliminated the "Fetched 4 positions / Grouped into 0 spread(s)"
   every-3-seconds log spam when the agent isn't running.

6. **README rewrite.** 1016 → 591 lines. Added TOC at top; compressed
   Architecture Overview; merged Setup + Configuration Reference;
   trimmed Sentiment Pipeline section; promoted Live ↔ Backtest Unified
   Decision Engine to its own H2; replaced all `signals.jsonl`
   references with `signals_live.jsonl` / `signals_backtest.jsonl`;
   documented watchdog + broker-gating + new env vars.

7. **`.gitignore` cleanup.** Sectioned into Secrets / Build / IDE /
   Runtime / Agent state / Generated reports. Added `AGENT_RUNNING`,
   `STRATEGY_PRESET.json`, `knowledge_base/`, `.idea/`,
   `daily_report.html`, `reports/`.

8. **Smoke checks moved to `scripts/checks/`.** Five files:
   `scan_invariant_check.py`, `run_journal_split_check.py`,
   `run_live_vs_backtest_parity_check.py`,
   `run_unified_backtest_check.py`,
   `run_scan_diagnostics_check.py`. `sys.path` inserts updated to walk
   two levels up. New `scripts/checks/README.md` with troubleshooting.

9. **`.github/workflows/ci.yml` added.** Two jobs: `invariants` (AST
   checks, runs first as fast gate) → `tests` (Python 3.11/3.12 matrix
   running pytest then the four runtime smoke checks).

10. **Logging volume pass.** Demoted hot-path INFO → DEBUG in
    `position_monitor.py` (Fetched N positions / Grouped into N
    spread(s)) and `market_data.py` (per-ticker price history fetch,
    batch snapshot summary, per-ticker Alpaca real-time quote,
    per-(ticker, expiration) chain fetch). Each demotion has an
    inline comment explaining why. Recovery / fallback paths and
    per-cycle macro logs (VIX z-score, catalog lookup) intentionally
    kept at INFO.

11. **`daily_report.html` removed and gitignored.** Stale auto-generated
    artifact (38 KB, generated by `visualize_logs.py`). Added
    `daily_report.html` and `reports/` to `.gitignore`.

12. **`visualize_logs.py` journal-path fix.**
    - New constants `DEFAULT_JOURNAL = "trade_journal/signals_live.jsonl"`,
      `LEGACY_JOURNAL = "trade_journal/signals.jsonl"`.
    - New `_resolve_journal_path()` helper: tries the requested path,
      falls back to legacy *only* when the caller asked for the default
      (`signals_live.jsonl`). Explicit user paths (e.g.
      `signals_backtest.jsonl`) are never silently rewritten.
    - CLI `--journal` default updated; help text mentions the fallback.
    - Module docstring rewritten with new examples.
    - New `TestResolveJournalPath` class in `tests/test_visualize_logs.py`
      pins all four resolver branches.

13. **Watchlist tab (4 PRs).** New 4th Streamlit tab for multi-timeframe
    regime monitoring, shipped as a sequence:
    - **PR #1** `MarketDataProvider.fetch_intraday_bars(ticker, interval,
      lookback_days=None, include_live_overlay=True)` — yfinance for
      depth, Alpaca snapshot for the right-most live tick. 4h synthesised
      via `df.resample("4h")` from 60m bars. `INTRADAY_BARS_TTL=60s`,
      cached by `(ticker, interval)`. Tests: `TestFetchIntradayBars`.
    - **PR #2** `trading_agent/multi_tf_regime.py` — `classify_multi_tf`
      reuses `RegimeClassifier._determine_regime` (no shadow scorer,
      enforced by `TestNoShadowScorer`). Per-tf SMA windows scale
      `(50, 200)` for daily vs `(20, 50)` for intraday. `adx_strength`
      and `adx_strength_label` helpers (Wilder smoothing, pure pandas).
      `MultiTFRegime.agreement_score` collapses regimes to a 3-way
      trend bucket for cross-tf alignment.
    - **PR #3** `watchlist_store.py` (atomic JSON, `threading.RLock` for
      nested `add_ticker → save_watchlist` calls — `Lock` deadlocks) +
      `streamlit/watchlist_ui.py` (controls / macro strip / regime
      table). 17-case persistence test suite.
    - **PR #4** `streamlit/watchlist_chart.py` — 4-row Plotly stack
      (Price · Volume · Oscillators · Trend) with collapsible rows; six
      timeframe options (5m → 1d); indicator toggles for SMA/BB/ATR/
      Ichimoku/RSI/StochRSI/MACD/ADX. ADX line on chart and badge in
      table both delegate to the same `_adx_series` math.

14. **`pandas-ta` rejected — pure-pandas indicator stack.** Discovered
    `numba` (transitive dep of `pandas-ta`) has no Python 3.13+ wheels;
    source build needs LLVM, fails on most user machines. Rolled MACD,
    Stoch RSI, ATR, Ichimoku in pure pandas/numpy directly in
    `watchlist_chart.py` (~80 lines, matches existing in-house pattern
    for SMA/RSI/Bollinger/ADX). `TA-Lib` (system C dep) and `VectorBT`
    (same numba problem, wrong tool for live charting) also rejected;
    rationale captured in §7.6 below.

15. **ADX dtype bug fix.** `multi_tf_regime.adx_strength` raised
    `pandas.errors.DataError: No numeric types to aggregate` on flat or
    near-flat OHLC tapes. Root cause: `.replace(0, pd.NA)` on a float
    Series coerces dtype to `object`, after which `.ewm().mean()` can't
    aggregate. Fix: `pd.NA` → `np.nan` (keeps `float64`); added
    `pd.to_numeric(dx, errors="coerce")` belt-and-suspenders + early
    return when `dx.dropna().empty`. Same bug fixed pre-emptively in
    `streamlit/components.py::drawdown_chart`. Regression in
    `verify_adx_dtype_fix.py` (4 cases incl. flat tape, trending,
    short-input, NaN-laden).

---

## 11. Open / not-yet-done

Nothing actively in flight as of the writing of this brief.

Things on the radar but not started:

- Phase E of the watchdog plan was **intentionally skipped** (LLM
  fingerprint cache for the in-app LLM extension). User runs local
  Ollama models so token cost isn't a concern there. Don't pick this
  up without an explicit ask.
- Adaptive spread width / chain-sourced δ picker / Mean-Reversion in
  backtester — see §7.1. These are documented residual drift; closing
  any of them needs a new regression test alongside.

---

## 12. How to use this file with another LLM

Suggested opening prompt:

> I'm working on the trading agent project described in the attached
> `PROJECT_CONTEXT.md`. Read it carefully — pay particular attention to
> §2 (hard invariants) and §7 (pitfalls) before proposing any change.
> Then [insert your task here]. If your change might violate one of
> the invariants in §2, flag it explicitly rather than working around
> it.

For change requests that touch scoring, pricing, or the C/W floor:
remind the LLM to run `scripts/checks/scan_invariant_check.py` and
`scripts/checks/run_live_vs_backtest_parity_check.py` after the change.
