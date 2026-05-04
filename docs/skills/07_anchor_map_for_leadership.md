# Anchor map for leadership

> **One-line summary:** Every tradeable ticker maps to a "leadership anchor" — a sibling instrument we measure relative-strength against. Sector ETFs anchor to SPY; large-cap single names anchor to their sector ETF; everything unmapped falls back to SPY.
> **Source of truth:** [`trading_agent/regime.py:36-110`](../../trading_agent/regime.py)
> **Phase:** 1  •  **Group:** bias
> **Depends on:** nothing — pure data structure + helper.
> **Consumed by:** `regime.classify`, `multi_tf_regime._classify_intraday`, the watchlist UI for the "vs X" badge.

---

## 1. Theory & Objective

Relative strength is only meaningful against a comparable benchmark. JPM running ahead of SPY tells you "financials are leading the market." JPM running ahead of XLF tells you "JPM is leading financials." The second statement is more actionable for trade routing because it's idiosyncratic — it survives even when the broader market is flat.

So we maintain a hand-curated map:

- **Sector ETFs (XLK, XLF, …) → SPY.** These are sector-vs-broad-market comparisons.
- **Single names (AAPL, JPM, TSLA, …) → matching sector ETF.** Single-name-vs-peer-sector.
- **Everything else → SPY.** Conservative fallback added 2026-05-02.

The fallback is important. Before that, `LEADERSHIP_ANCHORS.get(ticker, "")` returned the empty string for unmapped tickers, the classifier silently set `leadership_zscore = 0.0`, and the watchlist UI rendered `+0.000` indistinguishable from a real near-zero reading. The `leadership_anchor_for(ticker)` helper exists so a new ticker on a watchlist gets *some* signal automatically rather than silently failing.

The map is **append-only**. Adding a new ticker is safe. Changing an existing mapping (e.g. switching MSFT from XLK to XLC) would silently invalidate any historical analysis that compared old z-scores. If you need to change a mapping, version the change and document it in the git log.

## 2. Mathematical Formula

N/A — pure data structure. The math lives in skill 08 (Leadership Z-score), which consumes this map.

## 3. Reference Python Implementation

```python
# trading_agent/regime.py:36-88
LEADERSHIP_ANCHORS: dict = {
    # ---- ETFs ----
    "SPY": "QQQ",     # broad market vs growth proxy
    "QQQ": "SPY",     # growth vs broad market
    "IWM": "SPY",     # small-cap vs broad market
    "DIA": "SPY",     # blue-chip vs broad market
    "XLK": "SPY",     # tech sector vs broad market
    "XLF": "SPY",     # financial sector vs broad market
    "XLE": "SPY",     # energy sector vs broad market
    "XLV": "SPY",     # healthcare sector vs broad market
    "XLY": "SPY",     # cons. discretionary vs broad market
    "XLI": "SPY",     # industrial sector vs broad market
    "XLP": "SPY",     # cons. staples vs broad market
    "XLU": "SPY",     # utilities sector vs broad market
    "XLB": "SPY",     # materials sector vs broad market
    "XLC": "SPY",     # comms sector vs broad market
    "XLRE": "SPY",    # real-estate sector vs broad market
    # ---- Large-cap single names → matching sector ETF ----
    # Anchoring a stock to its sector ETF gives a "vs peers" leadership
    # signal: e.g. JPM running ahead of XLF means JPM is leading financials,
    # not just riding the sector tide.  Without these entries, the watchlist
    # silently fell through to a 0.0 z-score for any non-ETF ticker.
    "JPM":   "XLF",   # financials
    "BAC":   "XLF",
    "WFC":   "XLF",
    "C":     "XLF",
    "GS":    "XLF",
    "MS":    "XLF",
    "AAPL":  "XLK",   # tech
    "MSFT":  "XLK",
    "NVDA":  "XLK",
    "AMD":   "XLK",
    "INTC":  "XLK",
    "GOOGL": "XLC",   # comms
    "META":  "XLC",
    "NFLX":  "XLC",
    "TSLA":  "XLY",   # consumer discretionary
    "AMZN":  "XLY",
    "HD":    "XLY",
    "MCD":   "XLY",
    "NKE":   "XLY",
    "XOM":   "XLE",   # energy
    "CVX":   "XLE",
    "COP":   "XLE",
    "JNJ":   "XLV",   # healthcare
    "UNH":   "XLV",
    "PFE":   "XLV",
    "LLY":   "XLV",
    "PG":    "XLP",   # consumer staples
    "KO":    "XLP",
    "PEP":   "XLP",
    "WMT":   "XLP",
}
```

```python
# trading_agent/regime.py:91-110
def leadership_anchor_for(ticker: str) -> str:
    """Resolve a ticker's leadership anchor with a SPY fallback.

    Why a helper rather than just dict lookup
    -----------------------------------------
    Old code: ``LEADERSHIP_ANCHORS.get(ticker, "")`` — non-ETF, non-listed
    tickers returned the empty string and the classifier silently dropped
    the leadership signal to 0.0.  The watchlist UI then rendered ``+0.00``
    indistinguishable from a real near-zero reading.

    New behaviour: any ticker not in the explicit map falls back to ``SPY``
    so "vs broad market" is computable for everything.  ``ticker == "SPY"``
    is excluded (would self-anchor); SPY's anchor stays QQQ via the dict.
    Returns ``""`` only for the special "SPY → SPY would self-anchor" case,
    which can't happen given the dict has SPY → QQQ — defensive only.
    """
    explicit = LEADERSHIP_ANCHORS.get(ticker)
    if explicit:
        return explicit
    return "SPY" if ticker != "SPY" else ""
```

## 4. Edge Cases / Guardrails

- **Ticker == its own anchor (self-anchor)** — e.g. asking for the anchor of `SPY` when SPY is the fallback. Helper returns `""`, the leadership Z-score function (skill 08) returns `None`, the classifier sees `leadership_signal_available = False`, and the UI renders `— (no data vs SPY)`. No infinite-recursion or div-by-zero.
- **New ticker added to a watchlist** — automatic SPY fallback. The watchlist row shows leadership vs SPY immediately, without any code change. Add an explicit mapping later when the sector is known.
- **Sector reclassification (e.g. GOOGL: XLK → XLC)** — historically GOOGL anchored XLK; we moved it to XLC after the GICS reorg. Any historical z-score computed before the change is implicitly invalidated. The map carries no version history.
- **Lowercase / whitespace** — the map keys are uppercase and trimmed. Callers must pass `ticker.upper().strip()` before lookup. The helper does **not** normalize.
- **Bidirectional pair (SPY ↔ QQQ)** — the only intentional bidirectional anchor. Both are tradeable; neither is a sector ETF. The cross-reference makes their relative-strength symmetric and meaningful in either direction.

## 5. Cross-References

- [08 Leadership Z-score](08_leadership_zscore.md) — the consumer; turns the anchor pairing into a numerical bias.
- [09 VIX Z-score inhibitor](09_vix_zscore_inhibitor.md) — sibling overlay; both apply on top of regime.
- (Phase 2) `17_signal_availability_sentinel.md` — explains the `*_signal_available: bool` pattern that prevents the silent-zero failure mode this map's fallback was designed to fix.

---

*Last verified against repo HEAD on 2026-05-03.*
