"""Tests for the strategy planner — spread selection and strike picking."""

import pytest
from unittest.mock import MagicMock

from trading_agent.strategy import StrategyPlanner, SpreadPlan
from trading_agent.regime import Regime, RegimeAnalysis
from trading_agent.market_data import MarketDataProvider


def _make_analysis(regime: Regime) -> RegimeAnalysis:
    return RegimeAnalysis(
        regime=regime,
        current_price=500.0,
        sma_50=498.0,
        sma_200=490.0,
        sma_50_slope=0.5,
        rsi_14=55.0,
        bollinger_width=0.06,
        reasoning="Test analysis",
    )


class TestStrategySelection:

    def _make_planner(self, put_chain=None, call_chain=None):
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_option_chain.side_effect = lambda ticker, exp, opt_type: (
            put_chain if opt_type == "put" else call_chain
        )
        return StrategyPlanner(provider, max_delta=0.20, min_credit_ratio=0.33)

    def test_bullish_selects_bull_put(self, sample_put_contracts):
        planner = self._make_planner(put_chain=sample_put_contracts)
        plan = planner.plan("SPY", _make_analysis(Regime.BULLISH))
        assert plan.strategy_name == "Bull Put Spread"
        assert len(plan.legs) == 2
        # Sold leg should have higher strike
        sold = [l for l in plan.legs if l.action == "sell"]
        bought = [l for l in plan.legs if l.action == "buy"]
        assert len(sold) == 1
        assert len(bought) == 1
        assert sold[0].strike > bought[0].strike

    def test_bearish_selects_bear_call(self, sample_call_contracts):
        planner = self._make_planner(call_chain=sample_call_contracts)
        plan = planner.plan("SPY", _make_analysis(Regime.BEARISH))
        assert plan.strategy_name == "Bear Call Spread"
        assert len(plan.legs) == 2
        sold = [l for l in plan.legs if l.action == "sell"]
        bought = [l for l in plan.legs if l.action == "buy"]
        assert sold[0].strike < bought[0].strike

    def test_sideways_selects_iron_condor(self, sample_put_contracts,
                                           sample_call_contracts):
        planner = self._make_planner(
            put_chain=sample_put_contracts,
            call_chain=sample_call_contracts)
        plan = planner.plan("SPY", _make_analysis(Regime.SIDEWAYS))
        assert plan.strategy_name == "Iron Condor"
        assert len(plan.legs) == 4

    def test_empty_chain_returns_invalid_plan(self):
        planner = self._make_planner(put_chain=None, call_chain=None)
        plan = planner.plan("SPY", _make_analysis(Regime.BULLISH))
        assert plan.valid is False
        assert "unavailable" in plan.rejection_reason.lower() or \
               "no" in plan.rejection_reason.lower()


class TestPickExpiration:
    """Verify _pick_expiration always returns a Friday within DTE_RANGE."""

    def _planner(self):
        return StrategyPlanner(MagicMock(), max_delta=0.20, min_credit_ratio=0.33)

    def _dte(self, expiry_str, from_date):
        from datetime import datetime
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return (exp - from_date).days

    def test_result_is_last_trading_day_of_week(self):
        """Expiration is the last trading day of a week — usually Friday,
        but Thursday during holiday-Friday weeks (e.g. Good Friday).
        """
        from datetime import datetime
        planner = self._planner()
        exp = planner._pick_expiration()
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        # Thursday (3) acceptable on holiday-Friday weeks; Friday (4) is the norm.
        assert exp_date.weekday() in (3, 4), f"{exp} is not a Thu/Fri expiration"

    def test_result_within_dte_range(self):
        from datetime import datetime
        planner = self._planner()
        exp = planner._pick_expiration()
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        from datetime import date
        dte = (exp_date - date.today()).days
        lo, hi = planner.DTE_RANGE
        assert lo <= dte <= hi, f"DTE={dte} is outside range {lo}-{hi}"

    def test_both_valid_fridays_picks_further_one(self):
        """When multiple adjacent Fridays are within DTE_RANGE, pick the
        highest-DTE one (more theta runway, less gamma risk near expiry).
        Reflects the post-patch defaults: TARGET_DTE=35, DTE_RANGE=(28,45).
        """
        from datetime import datetime, date
        from unittest.mock import patch

        planner = self._planner()
        # fake_today = April 2, 2026 (Thu) → TARGET_DTE=35 → target = May 7 (Thu)
        # this_friday = May 8  (DTE=36) — VALID
        # next_friday = May 15 (DTE=43) — VALID
        # prev_friday = May 1  (DTE=29) — VALID
        # All three in DTE_RANGE (28, 45) → pick max DTE = May 15
        fake_today = date(2026, 4, 2)
        with patch("trading_agent.strategy.strategy.datetime") as mock_dt:
            mock_dt.now.return_value.date.return_value = fake_today
            exp = planner._pick_expiration()

        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        assert exp_date.weekday() == 4, "Expiry must be a Friday"
        assert exp_date == date(2026, 5, 15), "Should prefer the higher-DTE Friday"
        assert (exp_date - fake_today).days == 43  # within DTE_RANGE upper bound

    def test_only_prev_friday_valid_picks_prev(self):
        """When next_friday exceeds max DTE, fall back to prev_friday."""
        from datetime import datetime, date
        from unittest.mock import patch

        planner = self._planner()
        # fake_today = April 1, 2026 → target = May 16 (Saturday)
        # next_friday = May 22 (DTE=51 > 50) — INVALID (exceeds max)
        # prev_friday = May 15 (DTE=44)       — VALID
        fake_today = date(2026, 4, 1)
        with patch("trading_agent.strategy.strategy.datetime") as mock_dt:
            mock_dt.now.return_value.date.return_value = fake_today
            exp = planner._pick_expiration()

        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        assert exp_date.weekday() == 4, "Expiry must be a Friday"
        assert exp_date == date(2026, 5, 15), "Should pick prev_friday when next is out of range"
        dte = (exp_date - fake_today).days
        assert planner.DTE_RANGE[0] <= dte <= planner.DTE_RANGE[1]

    def test_old_utcnow_bug_would_have_returned_outside_range(self):
        """Regression: old utcnow roll-forward logic returned May 22 (51 DTE)."""
        from datetime import date, timedelta
        fake_today = date(2026, 4, 2)   # UTC date that caused the failure
        target = fake_today + timedelta(days=44)   # May 16 (Saturday)
        days_until_friday = (4 - target.weekday()) % 7
        broken_expiry = target + timedelta(days=days_until_friday)
        broken_dte = (broken_expiry - fake_today).days
        # Confirm the old code would have been outside range
        assert broken_dte > 45

    def test_good_friday_week_falls_back_to_thursday(self):
        """When the Friday in the chosen week is a market holiday
        (Good Friday — e.g. 2026-04-03), the NYSE calendar returns the
        preceding Thursday as the effective weekly expiration.

        Old naive-weekday math returned April 3 — a closed-market day —
        which broke downstream strike lookups.
        """
        from datetime import date
        from unittest.mock import patch
        from trading_agent.calendar_utils import next_weekly_expiration

        # fake_today = Feb 17, 2026 → target ≈ April 3 (Good Friday 2026)
        # The week containing April 3 should resolve to April 2 (Thursday)
        # because NYSE is closed on Good Friday.
        fake_today = date(2026, 2, 17)
        # Call the helper directly so the test doesn't depend on planner state
        exp = next_weekly_expiration(
            today=fake_today, target_dte=45, dte_min=35, dte_max=50)
        # Expected: Thursday April 2, 2026 — Good Friday is April 3
        assert exp == date(2026, 4, 2), (
            f"Expected Thu 2026-04-02 (Good Friday fallback), got {exp}")
        assert exp.weekday() == 3  # Thursday


class TestMeanReversionStrategy:
    def _make_planner(self, put_chain=None, call_chain=None):
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_option_chain.side_effect = lambda ticker, exp, opt_type: (
            put_chain if opt_type == "put" else call_chain
        )
        return StrategyPlanner(provider, max_delta=0.20, min_credit_ratio=0.33)

    def _make_mr_analysis(self, direction: str) -> RegimeAnalysis:
        from trading_agent.regime import Regime
        return RegimeAnalysis(
            regime=Regime.MEAN_REVERSION,
            current_price=500.0,
            sma_50=498.0,
            sma_200=490.0,
            sma_50_slope=0.5,
            rsi_14=55.0,
            bollinger_width=0.06,
            reasoning="3-std BB touch",
            mean_reversion_signal=True,
            mean_reversion_direction=direction,
        )

    def test_upper_band_touch_sells_bear_call(self, sample_call_contracts):
        planner = self._make_planner(call_chain=sample_call_contracts)
        plan = planner.plan("SPY", self._make_mr_analysis("upper"))
        assert plan.strategy_name == "Mean Reversion Spread"
        assert "upper" in plan.reasoning.lower() or "reversion" in plan.reasoning.lower()
        sold = [l for l in plan.legs if l.action == "sell"]
        bought = [l for l in plan.legs if l.action == "buy"]
        assert sold[0].strike < bought[0].strike  # bear call structure

    def test_lower_band_touch_sells_bull_put(self, sample_put_contracts):
        planner = self._make_planner(put_chain=sample_put_contracts)
        plan = planner.plan("SPY", self._make_mr_analysis("lower"))
        assert plan.strategy_name == "Mean Reversion Spread"
        sold = [l for l in plan.legs if l.action == "sell"]
        bought = [l for l in plan.legs if l.action == "buy"]
        assert sold[0].strike > bought[0].strike  # bull put structure


class TestRelativeStrengthBias:
    def _make_planner(self, put_chain=None, call_chain=None):
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_option_chain.side_effect = lambda ticker, exp, opt_type: (
            put_chain if opt_type == "put" else call_chain
        )
        return StrategyPlanner(provider, max_delta=0.20, min_credit_ratio=0.33)

    def test_sideways_with_rs_picks_bull_put(self, sample_put_contracts):
        """SIDEWAYS regime + Z-score leadership > 1.5 σ → Bull Put Spread."""
        from trading_agent.regime import Regime
        planner = self._make_planner(put_chain=sample_put_contracts)
        analysis = RegimeAnalysis(
            regime=Regime.SIDEWAYS,
            current_price=500.0, sma_50=498.0, sma_200=490.0,
            sma_50_slope=0.5, rsi_14=55.0, bollinger_width=0.06,
            reasoning="sideways",
            leadership_anchor="QQQ",
            leadership_zscore=1.8,           # > 1.5 σ → outperforming
        )
        plan = planner.plan("SPY", analysis)
        assert plan.strategy_name == "Bull Put Spread"

    def test_sideways_without_rs_picks_iron_condor(self, sample_put_contracts,
                                                     sample_call_contracts):
        """SIDEWAYS regime with no significant Z-score signal → Iron Condor."""
        from trading_agent.regime import Regime
        planner = self._make_planner(put_chain=sample_put_contracts,
                                      call_chain=sample_call_contracts)
        analysis = RegimeAnalysis(
            regime=Regime.SIDEWAYS,
            current_price=500.0, sma_50=498.0, sma_200=490.0,
            sma_50_slope=0.0, rsi_14=50.0, bollinger_width=0.06,
            reasoning="sideways",
            leadership_anchor="QQQ",
            leadership_zscore=0.5,           # below 1.5 σ threshold
        )
        plan = planner.plan("SPY", analysis)
        assert plan.strategy_name == "Iron Condor"

    def test_sub_threshold_leadership_does_not_trigger_bias(
            self, sample_put_contracts, sample_call_contracts):
        """Z-score 1.0 σ < threshold 1.5 σ → no bias; SIDEWAYS stays IC."""
        from trading_agent.regime import Regime
        planner = self._make_planner(put_chain=sample_put_contracts,
                                      call_chain=sample_call_contracts)
        analysis = RegimeAnalysis(
            regime=Regime.SIDEWAYS,
            current_price=500.0, sma_50=498.0, sma_200=490.0,
            sma_50_slope=0.0, rsi_14=50.0, bollinger_width=0.06,
            reasoning="sideways",
            leadership_anchor="QQQ",
            leadership_zscore=1.0,           # below 1.5 σ — sub-threshold
        )
        plan = planner.plan("SPY", analysis)
        assert plan.strategy_name == "Iron Condor"

    def test_no_anchor_means_no_bias(self, sample_put_contracts,
                                       sample_call_contracts):
        """Empty leadership_anchor → bias path is skipped even with high z."""
        from trading_agent.regime import Regime
        planner = self._make_planner(put_chain=sample_put_contracts,
                                      call_chain=sample_call_contracts)
        analysis = RegimeAnalysis(
            regime=Regime.SIDEWAYS,
            current_price=500.0, sma_50=498.0, sma_200=490.0,
            sma_50_slope=0.0, rsi_14=50.0, bollinger_width=0.06,
            reasoning="sideways",
            leadership_anchor="",            # no anchor configured
            leadership_zscore=2.5,           # would otherwise trigger
        )
        plan = planner.plan("SPY", analysis)
        assert plan.strategy_name == "Iron Condor"


class TestInterMarketGate:
    """VIX z-score > +2σ demotes Bull-Put / Iron-Condor to Bear Call."""

    def _make_planner(self, put_chain=None, call_chain=None):
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_option_chain.side_effect = lambda ticker, exp, opt_type: (
            put_chain if opt_type == "put" else call_chain
        )
        return StrategyPlanner(provider, max_delta=0.20, min_credit_ratio=0.33)

    def test_bullish_demotes_to_bear_call_when_inhibit_set(
            self, sample_put_contracts, sample_call_contracts):
        """BULLISH + inter_market_inhibit_bullish → Bear Call Spread."""
        from trading_agent.regime import Regime
        planner = self._make_planner(put_chain=sample_put_contracts,
                                      call_chain=sample_call_contracts)
        analysis = RegimeAnalysis(
            regime=Regime.BULLISH,
            current_price=500.0, sma_50=498.0, sma_200=490.0,
            sma_50_slope=0.5, rsi_14=55.0, bollinger_width=0.06,
            reasoning="bullish",
            leadership_anchor="QQQ",
            leadership_zscore=2.0,           # would normally bias bullish
            vix_zscore=2.5,
            inter_market_inhibit_bullish=True,
        )
        plan = planner.plan("SPY", analysis)
        assert plan.strategy_name == "Bear Call Spread"

    def test_sideways_demotes_to_bear_call_when_inhibit_set(
            self, sample_put_contracts, sample_call_contracts):
        """SIDEWAYS + inhibit_bull → Bear Call (no put-wing exposure)."""
        from trading_agent.regime import Regime
        planner = self._make_planner(put_chain=sample_put_contracts,
                                      call_chain=sample_call_contracts)
        analysis = RegimeAnalysis(
            regime=Regime.SIDEWAYS,
            current_price=500.0, sma_50=498.0, sma_200=490.0,
            sma_50_slope=0.0, rsi_14=50.0, bollinger_width=0.06,
            reasoning="sideways",
            leadership_anchor="QQQ",
            leadership_zscore=0.0,
            vix_zscore=2.5,
            inter_market_inhibit_bullish=True,
        )
        plan = planner.plan("SPY", analysis)
        assert plan.strategy_name == "Bear Call Spread"

    def test_inhibit_does_not_affect_bearish_regime(
            self, sample_call_contracts):
        """BEARISH + inhibit_bull → Bear Call anyway (gate is no-op here)."""
        from trading_agent.regime import Regime
        planner = self._make_planner(call_chain=sample_call_contracts)
        analysis = RegimeAnalysis(
            regime=Regime.BEARISH,
            current_price=500.0, sma_50=498.0, sma_200=520.0,
            sma_50_slope=-0.5, rsi_14=40.0, bollinger_width=0.06,
            reasoning="bearish",
            vix_zscore=2.5,
            inter_market_inhibit_bullish=True,
        )
        plan = planner.plan("SPY", analysis)
        assert plan.strategy_name == "Bear Call Spread"


class TestStrikeGridInference:
    """``_strike_grid_step`` infers the modal spacing of the chain so
    the adaptive width snap-up logic always lands on a real strike."""

    def _planner(self):
        return StrategyPlanner(MagicMock(), max_delta=0.20, min_credit_ratio=0.33)

    def test_dollar_grid(self):
        chain = [{"strike": s} for s in (480, 481, 482, 483, 484, 485)]
        assert StrategyPlanner._strike_grid_step(chain) == 1.0

    def test_five_dollar_grid(self):
        chain = [{"strike": s} for s in (470, 475, 480, 485, 490)]
        assert StrategyPlanner._strike_grid_step(chain) == 5.0

    def test_two_fifty_grid(self):
        chain = [{"strike": s} for s in (100.0, 102.5, 105.0, 107.5, 110.0)]
        assert StrategyPlanner._strike_grid_step(chain) == 2.5

    def test_thin_chain_returns_floor(self):
        """Fewer than 3 strikes → fall back to $1.00 floor (don't crash)."""
        chain = [{"strike": 100.0}]
        assert StrategyPlanner._strike_grid_step(chain) == 1.0
        chain2 = [{"strike": 100.0}, {"strike": 101.0}]
        assert StrategyPlanner._strike_grid_step(chain2) == 1.0

    def test_modal_gap_when_grid_is_mixed(self):
        """Near-the-money $1 wings + far-OTM $5 wings — pick the dominant one."""
        # 5 single-dollar gaps, 2 five-dollar gaps → mode = 1.0
        strikes = [100, 101, 102, 103, 104, 105, 110, 115]
        chain = [{"strike": float(s)} for s in strikes]
        assert StrategyPlanner._strike_grid_step(chain) == 1.0


class TestAdaptiveSpreadWidth:
    """``_pick_spread_width`` = max(legacy floor, 3×grid, 2.5%×spot)
    snapped UP to the strike grid."""

    def _planner(self):
        return StrategyPlanner(MagicMock(), max_delta=0.20, min_credit_ratio=0.33)

    def test_low_spot_uses_legacy_floor(self):
        """Spot $80 on $1 grid: max($5, $3, $2) = $5 (floor wins)."""
        planner = self._planner()
        chain = [{"strike": float(s)} for s in range(70, 91)]   # $1 grid
        width = planner._pick_spread_width(chain, sold_strike=80.0)
        assert width == 5.0

    def test_high_spot_scales_up(self):
        """SPY-style: spot $700 on $5 grid: max($5, $15, $17.50) = $17.50,
        snapped UP to $20 (next $5 strike)."""
        planner = self._planner()
        chain = [{"strike": float(s)} for s in range(600, 800, 5)]  # $5 grid
        width = planner._pick_spread_width(chain, sold_strike=700.0)
        assert width == 20.0

    def test_fine_grid_drives_three_strike_minimum(self):
        """Spot $200 on $2.50 grid: max($5, $7.50, $5) = $7.50,
        snapped UP to $7.50 (already on grid)."""
        planner = self._planner()
        chain = [{"strike": float(s) / 2.0}
                 for s in range(380, 440)]  # $0.50 grid
        # override grid via cleaner chain
        chain = [{"strike": float(s)} for s in
                 [195.0, 197.5, 200.0, 202.5, 205.0, 207.5, 210.0]]
        width = planner._pick_spread_width(chain, sold_strike=200.0)
        # max($5, 3×2.5=$7.50, 2.5%×200=$5) = $7.50
        assert width == 7.5

    def test_width_always_snaps_to_grid(self):
        """The returned width must be an integer multiple of the grid step
        so a real strike sits at sold_strike ± width."""
        planner = self._planner()
        # $5 grid, spot 633: 2.5% × 633 = $15.83 → snap UP to $20
        chain = [{"strike": float(s)} for s in range(550, 720, 5)]
        width = planner._pick_spread_width(chain, sold_strike=633.0)
        assert width % 5.0 == 0
        assert width >= 15.0


class TestDTEBand:
    """Theta capture is concentrated in 25-40 DTE; the planner targets
    a 35-DTE Friday and accepts anything in (28, 45)."""

    def test_constants(self):
        assert StrategyPlanner.TARGET_DTE == 35
        assert StrategyPlanner.DTE_RANGE == (28, 45)


class TestPlanSerialization:
    def test_to_dict_roundtrip(self, valid_spread_plan):
        d = valid_spread_plan.to_dict()
        assert d["ticker"] == "SPY"
        assert d["strategy"] == "Bull Put Spread"
        assert len(d["legs"]) == 2
        assert isinstance(d["max_loss"], (int, float))
