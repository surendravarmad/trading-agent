"""
Trading Rules Config Loader
============================
Reads trading_rules.yaml and exposes typed dataclasses.

Priority order for the YAML file:
  1. yaml_path argument passed to load_trading_rules()
  2. TRADING_RULES_YAML_PATH environment variable
  3. trading_agent/config/trading_rules.yaml  (package default)

The YAML file is **required**.  Missing file, missing pyyaml, or a
parse error all raise immediately so misconfigured environments fail
loudly rather than silently running on stale defaults.

Partial YAML is fine — any key absent from the file falls back to the
dataclass field default for that specific field.

Secrets and environment-specific values (API keys, risk limits already
in .env, feature flags) live in .env — not here.  This file is for
trader-tunable algorithm parameters only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]

_DEFAULT_YAML = Path(__file__).parent / "trading_rules.yaml"


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StrategyRules:
    """Strike selection and spread construction parameters."""
    min_delta: float = 0.15
    target_dte: int = 35
    dte_range_min: int = 28
    dte_range_max: int = 45
    spread_width_floor: float = 5.0
    rs_zscore_threshold: float = 1.5

    @property
    def dte_range(self) -> Tuple[int, int]:
        return (self.dte_range_min, self.dte_range_max)


@dataclass
class RegimeRules:
    """Market regime classification thresholds."""
    vix_inhibit_zscore: float = 2.0
    bollinger_narrow_threshold: float = 0.04
    leadership_anchors: Dict[str, str] = field(default_factory=lambda: {
        "SPY": "QQQ",
        "QQQ": "SPY",
        "IWM": "SPY",
        "DIA": "SPY",
        "XLK": "SPY",
        "XLF": "SPY",
        "XLE": "SPY",
        "XLV": "SPY",
        "XLY": "SPY",
        "XLI": "SPY",
        "XLP": "SPY",
        "XLU": "SPY",
        "XLB": "SPY",
        "XLC": "SPY",
        "XLRE": "SPY",
    })


@dataclass
class PositionMonitorRules:
    """Position exit signal thresholds."""
    profit_target_pct: float = 0.50
    hard_stop_multiplier: float = 3.0
    strike_proximity_pct: float = 0.01
    dte_safety_hour: int = 15
    dte_safety_minute: int = 30


@dataclass
class AgentRules:
    """Cycle orchestration parameters."""
    cycle_timeout_seconds: int = 270
    exit_debounce_required: int = 3


@dataclass
class ExecutionRules:
    """Order execution parameters."""
    max_history: int = 200
    price_drift_warn_pct: float = 0.10


@dataclass
class CacheRules:
    """Data cache TTLs (seconds) and concurrency settings."""
    price_history_ttl: int = 14400
    snapshot_ttl: int = 60
    option_chain_ttl: int = 180
    intraday_return_ttl: int = 60
    max_prefetch_workers: int = 5


@dataclass
class SentimentRules:
    """News source authority weights for sentiment scoring."""
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "sec_edgar": 1.00,
        "fed_rss": 0.95,
        "yahoo": 0.70,
        "twitter": 0.50,
        "reddit_options": 0.45,
        "reddit_stocks": 0.45,
        "reddit_investing": 0.40,
        "reddit_wsb": 0.35,
    })


@dataclass
class BacktestRules:
    """Backtesting simulation parameters."""
    starting_equity: float = 100_000.0
    commission_round_trip: float = 2.60
    daily_otm_pct: float = 0.03
    intraday_otm_pct: float = 0.005


@dataclass
class TradingRulesConfig:
    """Root container for all trader-tunable parameters."""
    strategy: StrategyRules = field(default_factory=StrategyRules)
    regime: RegimeRules = field(default_factory=RegimeRules)
    position_monitor: PositionMonitorRules = field(default_factory=PositionMonitorRules)
    agent: AgentRules = field(default_factory=AgentRules)
    execution: ExecutionRules = field(default_factory=ExecutionRules)
    cache: CacheRules = field(default_factory=CacheRules)
    sentiment: SentimentRules = field(default_factory=SentimentRules)
    backtest: BacktestRules = field(default_factory=BacktestRules)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_trading_rules(yaml_path: str | None = None) -> TradingRulesConfig:
    """
    Load trading rules from YAML.

    Raises
    ------
    ImportError       if pyyaml is not installed
    FileNotFoundError if the YAML file does not exist
    ValueError        if the YAML file cannot be parsed
    """
    if _yaml is None:
        raise ImportError(
            "pyyaml is required to load trading rules. "
            "Run: pip install pyyaml"
        )

    path = Path(yaml_path or os.getenv("TRADING_RULES_YAML_PATH", str(_DEFAULT_YAML)))
    if not path.exists():
        raise FileNotFoundError(
            f"trading_rules.yaml not found at {path}. "
            "Ensure the file exists or set TRADING_RULES_YAML_PATH to its location."
        )

    try:
        with path.open() as f:
            data = _yaml.safe_load(f) or {}
    except Exception as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc

    s = data.get("strategy", {})
    r = data.get("regime", {})
    pm = data.get("position_monitor", {})
    ag = data.get("agent", {})
    ex = data.get("execution", {})
    ca = data.get("cache", {})
    se = data.get("sentiment", {})
    bk = data.get("backtest", {})

    return TradingRulesConfig(
        strategy=StrategyRules(
            **{k: v for k, v in {
                "min_delta": s.get("min_delta"),
                "target_dte": s.get("target_dte"),
                "dte_range_min": s.get("dte_range_min"),
                "dte_range_max": s.get("dte_range_max"),
                "spread_width_floor": s.get("spread_width_floor"),
                "rs_zscore_threshold": s.get("rs_zscore_threshold"),
            }.items() if v is not None}
        ),
        regime=RegimeRules(
            **{k: v for k, v in {
                "vix_inhibit_zscore": r.get("vix_inhibit_zscore"),
                "bollinger_narrow_threshold": r.get("bollinger_narrow_threshold"),
                "leadership_anchors": r.get("leadership_anchors"),
            }.items() if v is not None}
        ),
        position_monitor=PositionMonitorRules(
            **{k: v for k, v in {
                "profit_target_pct": pm.get("profit_target_pct"),
                "hard_stop_multiplier": pm.get("hard_stop_multiplier"),
                "strike_proximity_pct": pm.get("strike_proximity_pct"),
                "dte_safety_hour": pm.get("dte_safety_hour"),
                "dte_safety_minute": pm.get("dte_safety_minute"),
            }.items() if v is not None}
        ),
        agent=AgentRules(
            **{k: v for k, v in {
                "cycle_timeout_seconds": ag.get("cycle_timeout_seconds"),
                "exit_debounce_required": ag.get("exit_debounce_required"),
            }.items() if v is not None}
        ),
        execution=ExecutionRules(
            **{k: v for k, v in {
                "max_history": ex.get("max_history"),
                "price_drift_warn_pct": ex.get("price_drift_warn_pct"),
            }.items() if v is not None}
        ),
        cache=CacheRules(
            **{k: v for k, v in {
                "price_history_ttl": ca.get("price_history_ttl"),
                "snapshot_ttl": ca.get("snapshot_ttl"),
                "option_chain_ttl": ca.get("option_chain_ttl"),
                "intraday_return_ttl": ca.get("intraday_return_ttl"),
                "max_prefetch_workers": ca.get("max_prefetch_workers"),
            }.items() if v is not None}
        ),
        sentiment=SentimentRules(
            **{k: v for k, v in {
                "source_weights": se.get("source_weights"),
            }.items() if v is not None}
        ),
        backtest=BacktestRules(
            **{k: v for k, v in {
                "starting_equity": bk.get("starting_equity"),
                "commission_round_trip": bk.get("commission_round_trip"),
                "daily_otm_pct": bk.get("daily_otm_pct"),
                "intraday_otm_pct": bk.get("intraday_otm_pct"),
            }.items() if v is not None}
        ),
    )
