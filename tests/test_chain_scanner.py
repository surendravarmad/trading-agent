"""
Unit tests for trading_agent.chain_scanner.

Two layers of coverage:
  1. Pure scoring helpers (_pop_from_delta, _cw_floor, _ev_per_dollar_risked,
     _score_candidate) — exercised against algebraic invariants without any
     market data or expiration calendar.
  2. ChainScanner.scan() — driven through a stubbed data_provider so we
     can shape the chain to force specific selection / ranking outcomes,
     and through a monkey-patched next_weekly_expiration() so the test
     calendar is deterministic in the sandbox.

No real Alpaca API calls. No pandas_market_calendars dependency.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from typing import Dict, List

import pytest

from trading_agent import chain_scanner as cs
from trading_agent.chain_scanner import (
    DEFAULT_FILL_HAIRCUT,
    ChainScanner,
    SpreadCandidate,
    _cw_floor,
    _ev_per_dollar_risked,
    _pop_from_delta,
    _quote_credit,
    _score_candidate,
)
from trading_agent.strategy_presets import BALANCED


# =============================================================================
# Pure scoring helpers
# =============================================================================

class TestPopFromDelta:
    """POP ≈ 1 − |Δshort|. Sign-agnostic; clamps at 0."""

    def test_typical_put_delta(self):
        assert _pop_from_delta(-0.30) == pytest.approx(0.70)

    def test_typical_call_delta(self):
        assert _pop_from_delta(0.25) == pytest.approx(0.75)

    def test_atm(self):
        assert _pop_from_delta(0.50) == pytest.approx(0.50)

    def test_zero_delta_is_full_pop(self):
        assert _pop_from_delta(0.0) == pytest.approx(1.0)

    def test_clamped_at_zero(self):
        # |Δ| > 1 is unrealistic but should not return a negative POP.
        assert _pop_from_delta(-1.5) == 0.0


class TestCWFloor:
    """C/W floor = |Δ| × (1 + edge_buffer). 10% edge buffer is the default."""

    def test_default_edge_buffer(self):
        assert _cw_floor(-0.30, 0.10) == pytest.approx(0.33)

    def test_zero_edge_buffer_is_breakeven(self):
        # No edge buffer → floor IS the breakeven (POP × CW = (1-POP)(1-CW)
        # collapses to CW = |Δ|).
        assert _cw_floor(-0.25, 0.0) == pytest.approx(0.25)

    def test_high_edge_buffer(self):
        assert _cw_floor(-0.20, 0.50) == pytest.approx(0.30)


class TestEVPerDollarRisked:
    """
    EV/$ = (POP*CW − (1−POP)*(1−CW)) / (1 − CW).
    Returns None for structurally invalid (zero/neg credit, zero/neg width,
    or credit ≥ width which is a debit not a credit).
    """

    def test_breakeven_returns_zero_or_near_zero(self):
        """Exactly at breakeven: POP*CW = (1-POP)(1-CW). EV/$ ≈ 0."""
        # |Δ| = 0.30 → breakeven CW = 0.30. credit/width = 0.30.
        ev = _ev_per_dollar_risked(credit=1.50, width=5.0, short_delta=-0.30)
        assert ev is not None
        # Within float-precision band
        assert abs(ev) < 1e-9

    def test_above_breakeven_positive(self):
        # 10% over breakeven → EV must be positive
        ev = _ev_per_dollar_risked(credit=1.65, width=5.0, short_delta=-0.30)
        assert ev > 0

    def test_below_breakeven_negative(self):
        # 10% under breakeven → EV negative (can still be sized but EV<0)
        ev = _ev_per_dollar_risked(credit=1.35, width=5.0, short_delta=-0.30)
        assert ev < 0

    def test_zero_credit_returns_none(self):
        assert _ev_per_dollar_risked(0.0, 5.0, -0.30) is None

    def test_negative_credit_returns_none(self):
        assert _ev_per_dollar_risked(-0.50, 5.0, -0.30) is None

    def test_zero_width_returns_none(self):
        assert _ev_per_dollar_risked(1.0, 0.0, -0.30) is None

    def test_credit_above_width_returns_none(self):
        # That's a debit, not a credit
        assert _ev_per_dollar_risked(6.0, 5.0, -0.30) is None

    def test_arbitrage_credit_equals_width_returns_none(self):
        # CW = 1 → max-loss = 0 → division by zero would explode
        assert _ev_per_dollar_risked(5.0, 5.0, -0.30) is None


class TestQuoteCredit:
    """
    NBBO-mid based credit estimator. Validates the four behaviours that
    matter for live operation:

      • Both legs quoted → mid-mid minus haircut (the common case).
      • Missing/zero short bid → fall back to bid (conservative for sold leg).
      • Missing/zero long ask → fall back to ask (conservative for bought leg).
      • Tight market with no haircut tolerance → never produces negative credit.
    """

    def test_both_quotes_present_uses_mid_minus_haircut(self):
        # short bid 1.00 / ask 1.10  → mid 1.05
        # long  bid 0.40 / ask 0.50  → mid 0.45
        # raw mid credit = 1.05 - 0.45 = 0.60; haircut $0.02 → 0.58
        c = _quote_credit(short_bid=1.00, short_ask=1.10,
                          long_bid =0.40, long_ask =0.50)
        assert c == pytest.approx(0.58)

    def test_zero_haircut_returns_pure_mid(self):
        c = _quote_credit(short_bid=1.00, short_ask=1.10,
                          long_bid =0.40, long_ask =0.50,
                          fill_haircut=0.0)
        assert c == pytest.approx(0.60)

    def test_strictly_better_than_worst_case(self):
        # mid pricing should never be worse than the legacy
        # ``short_bid − long_ask`` estimate by more than the haircut
        sb, sa, lb, la = 1.20, 1.30, 0.55, 0.65
        worst_case = round(sb - la, 2)            # legacy
        c = _quote_credit(sb, sa, lb, la, fill_haircut=0.0)
        assert c >= worst_case                    # never worse at zero haircut
        assert c == pytest.approx(((sb + sa) / 2) - ((lb + la) / 2))

    def test_missing_short_bid_falls_back_to_bid(self):
        # short_bid = 0 → can't form mid; fall back to bid (=0).
        # Effective short side is 0, so credit can only be zero or negative
        # → clipped to 0.
        c = _quote_credit(short_bid=0.0,  short_ask=1.10,
                          long_bid =0.40, long_ask =0.50)
        assert c == 0.0

    def test_missing_long_ask_falls_back_to_ask(self):
        # long_ask = 0 → fall back to ask (=0). Cheap long leg lifts credit
        # past the haircut.
        # short mid = 1.05, long ask = 0 → credit = 1.05 - 0.02 = 1.03
        c = _quote_credit(short_bid=1.00, short_ask=1.10,
                          long_bid =0.40, long_ask =0.0)
        assert c == pytest.approx(1.03)

    def test_negative_credit_clipped_to_zero(self):
        # short bid/ask both 0, long bid/ask both 1.00 → credit hugely
        # negative; helper must not propagate that as a "credit" — caller
        # rejects credit_non_positive on its own, but the contract is a
        # non-negative dollar amount.
        c = _quote_credit(short_bid=0.0, short_ask=0.0,
                          long_bid =1.0, long_ask =1.0)
        assert c == 0.0

    def test_default_haircut_constant_used(self):
        # Sanity: the helper's default haircut equals the public constant,
        # so callers (and tests) don't drift.
        c_default  = _quote_credit(short_bid=1.00, short_ask=1.10,
                                   long_bid =0.40, long_ask =0.50)
        c_explicit = _quote_credit(short_bid=1.00, short_ask=1.10,
                                   long_bid =0.40, long_ask =0.50,
                                   fill_haircut=DEFAULT_FILL_HAIRCUT)
        assert c_default == c_explicit


class TestScoreCandidate:
    """
    The full hard-filter pipeline:
      • dte > 0
      • POP >= min_pop
      • width > 0, credit > 0
      • CW >= floor (|Δ|×(1+edge))
      • EV per $-risked > 0
    Any failure → None.
    """

    def test_typical_pass(self):
        # Δ -0.30 → floor 0.33 (10% edge). Use credit 1.70 → CW 0.34, just
        # above the float-precision boundary at 1.65.
        out = _score_candidate(
            credit=1.70, width=5.0, short_delta=-0.30,
            dte=14, edge_buffer=0.10, min_pop=0.55,
        )
        assert out is not None
        cw, pop, floor, ev, annualized = out
        assert cw == pytest.approx(0.34)
        assert pop == pytest.approx(0.70)
        assert floor == pytest.approx(0.33)
        assert ev > 0
        # Annualized = ev * (365/14)
        assert annualized == pytest.approx(ev * (365 / 14))

    def test_below_floor_rejected(self):
        # Δ -0.30, edge 0.10 → floor 0.33. Credit 1.50 → CW = 0.30.
        assert _score_candidate(
            credit=1.50, width=5.0, short_delta=-0.30,
            dte=14, edge_buffer=0.10, min_pop=0.55,
        ) is None

    def test_pop_below_min_rejected(self):
        # |Δ| 0.50 → POP 0.50, below default min_pop 0.55
        assert _score_candidate(
            credit=2.80, width=5.0, short_delta=-0.50,
            dte=14, edge_buffer=0.10, min_pop=0.55,
        ) is None

    def test_zero_dte_rejected(self):
        assert _score_candidate(
            credit=1.65, width=5.0, short_delta=-0.30,
            dte=0, edge_buffer=0.10, min_pop=0.55,
        ) is None

    def test_zero_credit_rejected(self):
        assert _score_candidate(
            credit=0.0, width=5.0, short_delta=-0.30,
            dte=14, edge_buffer=0.10, min_pop=0.55,
        ) is None

    def test_higher_dte_lower_annualized_at_same_ev(self):
        """7 DTE candidate beats 30 DTE candidate at the same EV/$."""
        s7 = _score_candidate(
            credit=1.70, width=5.0, short_delta=-0.30,
            dte=7, edge_buffer=0.10, min_pop=0.55,
        )
        s30 = _score_candidate(
            credit=1.70, width=5.0, short_delta=-0.30,
            dte=30, edge_buffer=0.10, min_pop=0.55,
        )
        assert s7 is not None and s30 is not None
        assert s7[4] > s30[4], "shorter DTE → higher annualised score"

    def test_edge_buffer_zero_relaxes_floor(self):
        # With edge_buffer=0 (no margin), CW=0.30 vs Δ-0.30 sits exactly
        # at breakeven → EV ≈ 0 → still rejected (ev <= 0 guard).
        out_zero = _score_candidate(
            credit=1.50, width=5.0, short_delta=-0.30,
            dte=14, edge_buffer=0.0, min_pop=0.55,
        )
        assert out_zero is None

        # Tiny credit bump above breakeven → passes with zero edge buffer
        out_above = _score_candidate(
            credit=1.55, width=5.0, short_delta=-0.30,
            dte=14, edge_buffer=0.0, min_pop=0.55,
        )
        assert out_above is not None
        assert out_above[3] > 0


# =============================================================================
# ChainScanner.scan() — integration through a stubbed data provider
# =============================================================================

# A deterministic put chain centred near $150 with ~$1 strike spacing.
# Bids/asks are crafted so:
#   Δ-0.20 short with $5 wide gives credit 1.10 → CW=0.22 (just at floor 0.22)
#   Δ-0.30 short with $5 wide gives credit 1.65 → CW=0.33 (at floor 0.33)
#   Δ-0.35 short with $5 wide gives credit 2.00 → CW=0.40 (above floor 0.385)
# That last point should be the scanner's pick.
def _build_put_chain(spot: float = 150.0, expiration: str = "2026-05-08") -> List[Dict]:
    chain: List[Dict] = []
    for i, (strike, delta, bid, ask) in enumerate([
        (155, -0.55, 6.10, 6.30),
        (153, -0.45, 4.50, 4.70),
        (151, -0.35, 3.10, 3.30),
        (150, -0.30, 2.20, 2.40),
        (149, -0.25, 1.60, 1.80),
        (148, -0.20, 1.10, 1.30),
        (147, -0.15, 0.80, 0.95),
        (146, -0.12, 0.55, 0.70),
        (145, -0.10, 0.40, 0.55),
        (144, -0.08, 0.25, 0.40),
        (143, -0.06, 0.15, 0.30),
        (142, -0.04, 0.10, 0.20),
        (140, -0.02, 0.05, 0.10),
    ]):
        chain.append({
            "symbol": f"X{expiration.replace('-','')}P{int(strike*1000):08d}",
            "strike": float(strike), "delta": float(delta),
            "bid": float(bid), "ask": float(ask),
            "expiration": expiration, "type": "put",
        })
    return chain


def _build_call_chain(spot: float = 150.0, expiration: str = "2026-05-08") -> List[Dict]:
    chain: List[Dict] = []
    for strike, delta, bid, ask in [
        (140, 0.85, 10.0, 10.2),
        (145, 0.65,  6.0,  6.2),
        (148, 0.45,  3.5,  3.7),
        (149, 0.40,  3.0,  3.2),
        (150, 0.30,  2.4,  2.6),
        (151, 0.25,  1.7,  1.9),
        (152, 0.20,  1.2,  1.4),
        (153, 0.15,  0.85, 0.95),
        (155, 0.10,  0.60, 0.75),
    ]:
        chain.append({
            "symbol": f"X{expiration.replace('-','')}C{int(strike*1000):08d}",
            "strike": float(strike), "delta": float(delta),
            "bid": float(bid), "ask": float(ask),
            "expiration": expiration, "type": "call",
        })
    return chain


class _StubDataProvider:
    """
    Records every fetch_option_chain() call and returns the put / call chain
    fixture. Lets tests assert call patterns without spinning up a real
    MarketDataProvider.
    """

    def __init__(self, put_chain: List[Dict], call_chain: List[Dict]):
        self._put = put_chain
        self._call = call_chain
        self.calls: List[tuple] = []

    def fetch_option_chain(self, ticker: str, expiration: str, opt_type: str):
        self.calls.append((ticker, expiration, opt_type))
        return list(self._put if opt_type == "put" else self._call)


@pytest.fixture
def fake_today():
    # Pin to the Monday before May 8, 2026 (the chain fixture's expiration)
    return date(2026, 4, 27)


@pytest.fixture
def adaptive_preset():
    """BALANCED but flipped to adaptive scan mode + tight grids."""
    return replace(
        BALANCED,
        name="custom",
        scan_mode="adaptive",
        edge_buffer=0.10,
        min_pop=0.55,
        # Use a 4-DTE grid that all collapse onto the same Friday so we hit
        # the dedup branch (different target_dte → same expiration).
        dte_grid=(7, 10, 14, 17),
        delta_grid=(0.20, 0.25, 0.30, 0.35),
        width_grid_pct=(0.020, 0.025, 0.033),  # 3.0/5.0/$ at 150 spot
    )


@pytest.fixture(autouse=True)
def patch_calendar(monkeypatch, fake_today):
    """
    Monkey-patch ChainScanner's expiration picker to a deterministic Friday
    May 8, 2026. Every grid DTE collapses to the same expiration in this
    test setup so we exercise the seen_exps dedup path.
    """
    target_exp = date(2026, 5, 8)

    def _fake_pick(today, target_dte, dte_min, dte_max):
        return target_exp

    monkeypatch.setattr(cs, "next_weekly_expiration", _fake_pick)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

class TestScannerSelection:
    def test_returns_candidates_sorted_desc(self, adaptive_preset, fake_today):
        provider = _StubDataProvider(_build_put_chain(), _build_call_chain())
        scanner = ChainScanner(provider, adaptive_preset, max_candidates=20)

        out = scanner.scan("X", "bull_put", today=fake_today)

        assert out, "scanner should find at least one positive-EV candidate"
        # Strictly monotonic non-increasing on annualized score
        scores = [c.annualized_score for c in out]
        assert scores == sorted(scores, reverse=True)

    def test_top_pick_above_floor(self, adaptive_preset, fake_today):
        provider = _StubDataProvider(_build_put_chain(), _build_call_chain())
        scanner = ChainScanner(provider, adaptive_preset)

        out = scanner.scan("X", "bull_put", today=fake_today)

        assert out
        top = out[0]
        # Selected candidate must clear its own |Δ|×(1+edge) floor
        assert top.cw_ratio >= top.cw_floor - 1e-9
        # And the floor must equal |Δ|*1.10 (10% edge buffer)
        assert top.cw_floor == pytest.approx(abs(top.short_delta) * 1.10)

    def test_dedup_collapses_grid_to_one_chain_fetch_per_type(
        self, adaptive_preset, fake_today,
    ):
        """All four grid DTEs collapse onto the same fake Friday → only one
        fetch_option_chain call per (ticker, expiration, type)."""
        provider = _StubDataProvider(_build_put_chain(), _build_call_chain())
        scanner = ChainScanner(provider, adaptive_preset)

        scanner.scan("X", "bull_put", today=fake_today)
        # All 4 DTE grid points → same expiration → 1 fetch
        assert len(provider.calls) == 1
        assert provider.calls[0] == ("X", "2026-05-08", "put")

    def test_max_candidates_caps_output(self, adaptive_preset, fake_today):
        provider = _StubDataProvider(_build_put_chain(), _build_call_chain())
        scanner = ChainScanner(provider, adaptive_preset, max_candidates=3)

        out = scanner.scan("X", "bull_put", today=fake_today)
        assert 0 < len(out) <= 3

    def test_no_edge_returns_empty(self, fake_today):
        """When the grid sits below the floor for every (Δ, w), return []."""
        # Deflate every credit so CW < floor across the board
        cheap = [{**c, "bid": c["bid"] * 0.5, "ask": c["ask"] * 0.5}
                 for c in _build_put_chain()]
        provider = _StubDataProvider(cheap, _build_call_chain())
        preset = replace(
            BALANCED,
            scan_mode="adaptive",
            edge_buffer=0.50,   # demand 50% over breakeven
            min_pop=0.55,
            dte_grid=(14,),
            delta_grid=(0.30,),
            width_grid_pct=(0.020, 0.033),
        )
        scanner = ChainScanner(provider, preset)
        out = scanner.scan("X", "bull_put", today=fake_today)
        assert out == []

    def test_empty_chain_returns_empty(self, adaptive_preset, fake_today):
        provider = _StubDataProvider([], [])
        scanner = ChainScanner(provider, adaptive_preset)
        assert scanner.scan("X", "bull_put", today=fake_today) == []

    def test_bull_put_uses_put_chain_bear_call_uses_call_chain(
        self, adaptive_preset, fake_today,
    ):
        provider = _StubDataProvider(_build_put_chain(), _build_call_chain())
        scanner = ChainScanner(provider, adaptive_preset)

        scanner.scan("X", "bull_put", today=fake_today)
        scanner.scan("X", "bear_call", today=fake_today)

        types_called = {c[2] for c in provider.calls}
        assert types_called == {"put", "call"}

    def test_invalid_side_raises(self, adaptive_preset, fake_today):
        provider = _StubDataProvider(_build_put_chain(), _build_call_chain())
        scanner = ChainScanner(provider, adaptive_preset)
        with pytest.raises(ValueError):
            scanner.scan("X", "iron_condor", today=fake_today)

    def test_candidate_floor_label_matches_risk_manager_formula(
        self, adaptive_preset, fake_today,
    ):
        """Each candidate's stored cw_floor must equal the formula used
        by RiskManager Check 2 — same source of truth for both gates."""
        provider = _StubDataProvider(_build_put_chain(), _build_call_chain())
        scanner = ChainScanner(provider, adaptive_preset)

        out = scanner.scan("X", "bull_put", today=fake_today)
        for c in out:
            expected = abs(c.short_delta) * (1.0 + adaptive_preset.edge_buffer)
            assert c.cw_floor == pytest.approx(expected, rel=1e-9), (
                f"candidate {c.short_strike}/{c.long_strike} "
                f"cw_floor {c.cw_floor} != |Δ|×(1+edge) {expected}"
            )

    def test_candidates_are_journal_safe(self, adaptive_preset, fake_today):
        import json
        provider = _StubDataProvider(_build_put_chain(), _build_call_chain())
        scanner = ChainScanner(provider, adaptive_preset)
        out = scanner.scan("X", "bull_put", today=fake_today)
        assert out
        for c in out:
            d = c.to_journal_dict()
            json.dumps(d)   # must serialise; no Decimal / np.float64 leaks


class TestChainHelpers:
    """Quick sanity coverage on the inference helpers."""

    def test_infer_grid_step_modal(self):
        chain = [{"strike": s} for s in (140, 141, 142, 144, 145, 147)]
        # Most gaps are 1; the 2-gap at 142→144 and 145→147 don't dominate.
        assert ChainScanner._infer_grid_step(chain) == 1.0

    def test_infer_spot_proxy_picks_atm(self):
        chain = [
            {"strike": 145, "delta": 0.80},
            {"strike": 150, "delta": 0.50},
            {"strike": 155, "delta": 0.20},
        ]
        # Δ closest to 0.50 is the 150 strike
        assert ChainScanner._infer_spot_proxy(chain) == 150.0

    def test_infer_spot_proxy_falls_back_to_median(self):
        chain = [{"strike": s} for s in (140, 145, 150)]
        # No deltas → median strike
        assert ChainScanner._infer_spot_proxy(chain) == 145.0

    def test_snap_width_to_grid_rounds_up(self):
        # raw 4.0 with 5-wide grid → 5.0 (rounded UP to nearest step ≥ 1)
        assert ChainScanner._snap_width_to_grid(4.0, 5.0) == 5.0
        # raw 7.0 with 5-wide → 10.0
        assert ChainScanner._snap_width_to_grid(7.0, 5.0) == 10.0

    def test_find_short_picks_closest_delta(self):
        chain = _build_put_chain()
        out = ChainScanner._find_short(chain, 0.30)
        assert out is not None
        assert abs(abs(float(out["delta"])) - 0.30) < 0.06   # the 150 / Δ-0.30 row

    def test_find_short_skips_zero_bid(self):
        # A contract with bid=0 should be filtered out even if Δ matches
        chain = [
            {"strike": 145, "delta": -0.30, "bid": 0.0, "ask": 0.10},
            {"strike": 144, "delta": -0.32, "bid": 1.20, "ask": 1.40},
        ]
        out = ChainScanner._find_short(chain, 0.30)
        assert out is not None
        assert out["strike"] == 144   # the only priceable one

    def test_find_strike_closest(self):
        chain = [{"strike": s} for s in (140, 145, 150, 155)]
        assert ChainScanner._find_strike(chain, 147)["strike"] == 145
        assert ChainScanner._find_strike(chain, 148)["strike"] == 150
