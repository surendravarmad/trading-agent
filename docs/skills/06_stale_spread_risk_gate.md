# Stale-spread risk gate

> **One-line summary:** When the underlying's quoted bid/ask spread exceeds 1 % of mid, treat the quote as stale (e.g. pre-market, lunch lull, single-print) and **soft-pass** the candidate with a logged warning instead of failing on the absolute liquidity rule.
> **Source of truth:** [`trading_agent/risk_manager.py:50, 79, 166-190`](../../trading_agent/risk_manager.py)
> **Phase:** 1  •  **Group:** risk
> **Depends on:** nothing — independent risk gate.
> **Consumed by:** `risk_manager.RiskGate.evaluate` (only path).

---

## 1. Theory & Objective

Liquidity gates compare the underlying's quoted spread against an absolute floor (`liquidity_max_spread`) plus a basis-points-of-mid floor (`liquidity_bps_of_mid × mid`). For most regular-hours quotes this works — a tight spread relative to mid means real two-sided liquidity.

But there's a failure mode: a **stale** quote. Pre-market, post-market, lunch lulls, and single-print bars can produce quotes where the spread is *huge* both absolutely and relatively. The absolute gate would reject these as illiquid. But often the right thing is *not* to reject the candidate — the spread is wide because the quote is stale, not because the underlying is genuinely illiquid. The next live-tick will probably be tight again.

So we add a **third** rail: if the relative spread (`spread / mid`) exceeds `stale_spread_pct` (default 1 %), we classify it as STALE rather than ILLIQUID, log a warning, and **soft-pass** — the candidate proceeds. The downstream order-pricing logic limits what we'll actually pay, so a soft-pass on a genuinely illiquid name still won't fill at a bad price.

The order of checks is critical: **stale first, illiquid second**. A wide-spread name that's stale should not also be flagged as illiquid; that would double-count. The `if rel_spread > self.stale_spread_pct` branch returns before the absolute liquidity check runs.

## 2. Mathematical Formula

```text
spread       = ask − bid
mid          = (bid + ask) / 2
rel_spread   = spread / mid                                          (∞ if mid == 0)
scaled_floor = max(liquidity_max_spread, liquidity_bps_of_mid × mid)

decision:
  if rel_spread > stale_spread_pct:    STALE  (soft-pass + warn)
  elif spread < scaled_floor:          LIQUID (pass)
  else:                                ILLIQUID (fail)

where
  stale_spread_pct       ∈ (0, 1]   default 0.01
  liquidity_max_spread   ∈ ℝ⁺       absolute dollar floor
  liquidity_bps_of_mid   ∈ ℝ⁺       e.g. 0.0005 = 5 bps
```

## 3. Reference Python Implementation

```python
# trading_agent/risk_manager.py:50  (constructor default)
stale_spread_pct: float = 0.01,
```

```python
# trading_agent/risk_manager.py:79  (stored on instance)
self.stale_spread_pct = stale_spread_pct
```

```python
# trading_agent/risk_manager.py:166-190
if underlying_bid_ask is not None:
    bid, ask = underlying_bid_ask
    spread = ask - bid
    mid = (bid + ask) / 2.0 if (bid + ask) > 0 else 0.0
    scaled_floor = max(
        self.liquidity_max_spread,
        self.liquidity_bps_of_mid * mid,
    )
    rel_spread = (spread / mid) if mid > 0 else float("inf")
    if mid > 0 and rel_spread > self.stale_spread_pct:
        passed.append(
            f"Underlying spread ${spread:.4f} / mid ${mid:.2f} "
            f"= {rel_spread*100:.2f}% > stale threshold "
            f"{self.stale_spread_pct*100:.1f}% — treating as "
            f"STALE quote (soft-pass)"
        )
        logger.warning(
            "[%s] Stale-quote soft-pass: spread=$%.4f rel=%.2f%% "
            "(threshold %.1f%%) bid=$%.4f ask=$%.4f",
            plan.ticker, spread, rel_spread * 100,
            self.stale_spread_pct * 100, bid, ask,
        )
    elif spread < scaled_floor:
        passed.append(
            f"Underlying bid/ask spread ${spread:.4f} "
            f"< ${scaled_floor:.4f} (mid=${mid:.2f}, liquid)")
    else:
        failed.append(
            f"Underlying bid/ask spread ${spread:.4f} "
            f">= ${scaled_floor:.4f} "
            f"(floor=${self.liquidity_max_spread:.2f}, "
            f"{self.liquidity_bps_of_mid*1e4:.1f}bps×mid="
            f"${self.liquidity_bps_of_mid*mid:.4f}, illiquid)")
```

## 4. Edge Cases / Guardrails

- **`mid = 0`** — `rel_spread` becomes `float("inf")`. We do **not** trip the stale branch (`mid > 0` guard); we fall through to the absolute liquidity check, which fails as illiquid. This avoids a runaway "everything is stale" mis-classification when feed quotes both sides at zero.
- **`underlying_bid_ask is None`** — entire block is skipped. Liquidity is then untested. Upstream feed code is supposed to never return `None` here for live tickers; if it does, an option-chain check elsewhere will catch the missing data.
- **`stale_spread_pct = 0`** — every non-degenerate spread is "stale," so liquidity gates are effectively disabled. Log noise will be loud; useful for backtesting.
- **`stale_spread_pct = 1.0`** — almost nothing is ever classified stale; effectively reverts to original two-rail behaviour.
- **Soft-pass != silent** — every stale-pass logs a `WARNING` with full bid/ask/mid, so a post-mortem can find every fill that proceeded against a wide quote.
- **Different from "stale data"** — the watchlist UI's "⏰ stale data" badge is about `last_bar_ts` being old; this gate is about wide *current* quotes. Different problem, different rule. Don't conflate.

## 5. Cross-References

- [03 Credit-to-Width floor](03_credit_to_width_floor.md) — also runs in `RiskGate`; both can fail a candidate independently.
- (Phase 2) `16_stale_data_age_detection.md` — the unrelated `last_bar_ts` chip; intentionally distinct.

---

*Last verified against repo HEAD on 2026-05-03.*
