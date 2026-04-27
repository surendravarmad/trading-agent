"""
FinGPT Sentiment Analyser
==========================
Second-eye advisory layer: fetches recent news headlines for a ticker
and runs a locally-hosted finance LLM to produce a SentimentReport.

The report injects macro/event context into Phase V (LLM Analyst) that
pure technical signals cannot see — especially important for:
  • Earnings within 7 days  → binary event, avoid selling premium
  • Fed meeting / macro day → IV spike risk
  • Post-earnings IV crush  → favorable entry window for premium sellers
  • Sector-wide headwinds   → correlated move risk

News is sourced via yfinance (already a project dependency).
Inference runs through Ollama — set FINGPT_MODEL to any locally pulled
model; defaults to the same model the primary analyst already uses so
no extra download is required on first run.

Cache: per-ticker TTL (default 5 min) prevents redundant inference
calls when the same ticker appears in consecutive trading cycles.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from trading_agent.config import IntelligenceConfig
    from trading_agent.llm_client import LLMClient
    from trading_agent.news_aggregator import NewsItem

logger = logging.getLogger(__name__)

SENTIMENT_CACHE_TTL = 300   # seconds — matches intraday 5-min bar TTL


@dataclass
class SentimentReport:
    """Structured output from the FinGPT sentiment analyser."""
    ticker: str
    sentiment_score: float      # -1.0 (very bearish) to +1.0 (very bullish)
    event_risk: float           # 0.0 (none) to 1.0 (high: earnings/Fed/FDA/legal)
    confidence: float           # 0.0 to 1.0
    headlines: List[str]        # raw headlines fed to the model
    key_themes: List[str]       # e.g. ["earnings_beat", "fed_hawkish"]
    recommendation: str         # "favorable" | "neutral" | "caution" | "avoid"
    reasoning: str              # model's chain-of-thought
    cached: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_prompt_section(self) -> str:
        """Render as a markdown block for injection into the LLM analyst prompt."""
        sign = "+" if self.sentiment_score >= 0 else ""
        themes = ", ".join(self.key_themes) if self.key_themes else "none identified"
        headline_lines = "\n".join(f"  • {h}" for h in self.headlines[:5])
        return (
            f"## FinGPT Sentiment Analysis\n"
            f"**Sentiment:** {sign}{self.sentiment_score:.2f} "
            f"(−1=bearish, +1=bullish)  |  "
            f"**Event risk:** {self.event_risk:.2f}  |  "
            f"**Confidence:** {self.confidence:.2f}\n"
            f"**Recommendation:** {self.recommendation.upper()}\n"
            f"**Key themes:** {themes}\n"
            f"**Recent headlines:**\n{headline_lines}\n"
            f"**FinGPT reasoning:** {self.reasoning}"
        )


_SYSTEM_PROMPT = """You are FinGPT, a financial sentiment analysis expert specializing in options premium selling (credit spread) strategies.

Analyze news headlines for a stock and output a structured risk assessment. Credit spread sellers profit from time decay and want:
- Low event risk: no binary catalysts (earnings, FDA, Fed decision) within 7 days
- Elevated but stable IV: not event-driven spikes that collapse post-announcement
- Clear directional regime: not chaotic reversals driven by surprise news

Event risk scoring guide:
- 0.8–1.0: Earnings announcement within 7 days, Fed rate decision, FDA ruling, major M&A, legal verdict
- 0.5–0.7: Analyst day, product launch, sector regulatory review, geopolitical escalation
- 0.2–0.4: Routine analyst upgrades/downgrades, sector rotation, macro data prints
- 0.0–0.2: No near-term catalysts, stable background news

Recommendation guide:
- "avoid":     event_risk > 0.7 (binary risk, premium sellers must stay out)
- "caution":   event_risk 0.4–0.7 OR sentiment_score < -0.6 (directional risk elevated)
- "neutral":   mixed signals, no clear edge either way
- "favorable": event_risk < 0.4 AND sentiment stable-to-positive (good premium selling environment)

Respond ONLY in valid JSON — no markdown fences, no extra text:
{
  "sentiment_score": <float -1.0 to 1.0>,
  "event_risk": <float 0.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "key_themes": ["theme1", "theme2"],
  "recommendation": "favorable" | "neutral" | "caution" | "avoid",
  "reasoning": "<2-3 sentence explanation of the scores>"
}"""

_ITEMS_USER_TEMPLATE = """Analyze multi-source news for {ticker} (strategy: {strategy}).

Current market context:
- Regime: {regime}
- Price: ${price:.2f}
- RSI-14: {rsi:.1f}
- IV rank: {iv_rank:.1f}

News by source authority ({n_total} items, {n_sources} sources):
{grouped_evidence}

Source authority: SEC EDGAR=1.00 > Fed Reserve=0.95 > Yahoo=0.70 > Twitter=0.50 > Reddit options=0.45 > Reddit stocks=0.45 > Reddit WSB=0.35

Instructions:
1. Weight SEC filings and Fed items most heavily — they contain authoritative facts
2. Yahoo Finance adds context; Reddit/Twitter reveal retail positioning and momentum
3. Check ALL sources for near-term earnings, Fed meetings, or binary catalysts
4. Elevated Reddit/Twitter buzz on a quiet news day = retail positioning, not event risk
5. Assess whether this environment favors SELLING options premium on {ticker}"""

_USER_TEMPLATE = """Analyze recent news for {ticker} (strategy: {strategy}).

Current market context:
- Regime: {regime}
- Price: ${price:.2f}
- RSI-14: {rsi:.1f}
- IV rank: {iv_rank:.1f}

Recent news headlines ({n} articles):
{headlines}

Assess sentiment and event risk. Remember: we are deciding whether to SELL options premium (credit spreads) on {ticker} in the next few days."""


class FinGPTAnalyser:
    """
    Sentiment analysis layer using a locally-hosted finance LLM via Ollama.

    Maintains a per-ticker TTL cache so repeated calls within the same
    trading cycle return instantly without redundant model inference.

    Thread-safe: safe to call from a ThreadPoolExecutor worker alongside
    the primary LLM analyst without race conditions.
    """

    def __init__(
        self,
        cfg: Optional["IntelligenceConfig"] = None,
        client: Optional["LLMClient"] = None,
        enabled: Optional[bool] = None,
        # Legacy kwargs — retained for ad-hoc instantiation in tests.
        ollama_base_url: Optional[str] = None,
        fingpt_model: Optional[str] = None,
        news_limit: Optional[int] = None,
        cache_ttl: Optional[int] = None,
    ):
        # Pull tuning from IntelligenceConfig when provided; the legacy
        # kwargs (used only in tests now) win over the config fallback
        # so existing fixtures keep working.
        self.news_limit = (
            news_limit if news_limit is not None
            else (cfg.fingpt_news_limit if cfg else 10)
        )
        self.cache_ttl = (
            cache_ttl if cache_ttl is not None
            else (cfg.fingpt_cache_ttl if cfg else SENTIMENT_CACHE_TTL)
        )
        self.enabled = (
            enabled if enabled is not None
            else (bool(cfg and cfg.fingpt_enabled))
        )

        # Dedicated LLMClient so FinGPT inference never contends with
        # the primary analyst's in-flight requests.  Preferred path:
        # pass a pre-built client (from ``make_llm_client("fingpt", cfg)``).
        # Fallback path: legacy ad-hoc construction from explicit kwargs.
        from trading_agent.llm_client import LLMClient, LLMConfig, make_llm_client

        if client is not None:
            self._client = client
            self._model_name = client.config.model
        elif cfg is not None:
            self._client = make_llm_client("fingpt", cfg)
            self._model_name = cfg.fingpt_model
        else:
            self._model_name = fingpt_model or "qwen2.5-trading"
            self._client = LLMClient(LLMConfig(
                provider="ollama",
                base_url=ollama_base_url or "http://localhost:11434",
                model=self._model_name,
                temperature=0.1,
                max_tokens=512,
                timeout=45,
            ))

        # ticker → (SentimentReport, monotonic timestamp)
        self._cache: Dict[str, Tuple[SentimentReport, float]] = {}
        self._lock = threading.Lock()

        if self.enabled:
            logger.info(
                "FinGPT Analyser ENABLED — model=%s, news_limit=%d, ttl=%ds",
                self._model_name, self.news_limit, self.cache_ttl,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        ticker: str,
        regime: str = "UNKNOWN",
        current_price: float = 0.0,
        rsi: float = 50.0,
        iv_rank: float = 0.0,
        strategy_name: str = "credit spread",
    ) -> Optional[SentimentReport]:
        """
        Return a SentimentReport for ticker, or None if unavailable/disabled.
        Safe to call from a background thread.
        """
        if not self.enabled:
            return None

        hit = self._get_cached(ticker)
        if hit:
            return hit

        headlines = self._fetch_headlines(ticker)
        if not headlines:
            logger.info("[%s] FinGPT: no headlines available — skipping", ticker)
            return None

        report = self._infer(
            ticker, headlines, regime, current_price, rsi, iv_rank, strategy_name
        )
        if report:
            self._put_cache(ticker, report)
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_headlines(self, ticker: str) -> List[str]:
        """Fetch recent news via yfinance. Returns empty list on failure."""
        try:
            import yfinance as yf
            raw_news = yf.Ticker(ticker).news or []
            titles: List[str] = []
            for item in raw_news[: self.news_limit]:
                # yfinance news items vary by version: dict or object
                if isinstance(item, dict):
                    title = (
                        item.get("title")
                        or item.get("headline")
                        or item.get("content", {}).get("title", "")
                    )
                elif hasattr(item, "title"):
                    title = item.title
                else:
                    title = ""
                if title:
                    titles.append(str(title).strip())
            logger.debug("[%s] FinGPT: fetched %d headlines", ticker, len(titles))
            return titles
        except Exception as exc:
            logger.warning("[%s] FinGPT headline fetch failed: %s", ticker, exc)
            return []

    def _infer(
        self,
        ticker: str,
        headlines: List[str],
        regime: str,
        current_price: float,
        rsi: float,
        iv_rank: float,
        strategy_name: str,
    ) -> Optional[SentimentReport]:
        """Run FinGPT model inference and parse the JSON response."""
        headline_text = "\n".join(
            f"{i + 1}. {h}" for i, h in enumerate(headlines)
        )
        user_prompt = _USER_TEMPLATE.format(
            ticker=ticker,
            strategy=strategy_name,
            regime=regime,
            price=current_price,
            rsi=rsi,
            iv_rank=iv_rank,
            n=len(headlines),
            headlines=headline_text,
        )

        data = self._client.chat_json([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        if not data:
            logger.warning("[%s] FinGPT: empty or unparseable response", ticker)
            return None

        try:
            report = SentimentReport(
                ticker=ticker,
                sentiment_score=max(-1.0, min(1.0, float(data.get("sentiment_score", 0.0)))),
                event_risk=max(0.0, min(1.0, float(data.get("event_risk", 0.0)))),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                headlines=headlines,
                key_themes=list(data.get("key_themes", [])),
                recommendation=str(data.get("recommendation", "neutral")),
                reasoning=str(data.get("reasoning", "")),
            )
            logger.info(
                "[%s] FinGPT → sentiment=%.2f  event_risk=%.2f  rec=%s",
                ticker, report.sentiment_score, report.event_risk, report.recommendation,
            )
            return report
        except Exception as exc:
            logger.warning("[%s] FinGPT report parse error: %s", ticker, exc)
            return None

    def analyse_items(
        self,
        ticker: str,
        news_items: List["NewsItem"],
        regime: str = "UNKNOWN",
        current_price: float = 0.0,
        rsi: float = 50.0,
        iv_rank: float = 0.0,
        strategy_name: str = "credit spread",
    ) -> Optional[SentimentReport]:
        """
        Preferred entry point when multi-source data from NewsAggregator is
        available. Builds a source-grouped, authority-weighted prompt so FinGPT
        can distinguish an SEC 8-K from a Reddit post before scoring.

        Falls back to returning None if disabled or no items provided.
        Result is cached under the ticker key (same TTL as analyse()).
        """
        if not self.enabled or not news_items:
            return None

        hit = self._get_cached(ticker)
        if hit:
            return hit

        grouped = self._group_by_source(news_items)
        grouped_lines: List[str] = []
        # Sort source groups by descending authority weight
        for source, items in sorted(
            grouped.items(),
            key=lambda kv: kv[1][0].source_weight,
            reverse=True,
        ):
            weight = items[0].source_weight
            grouped_lines.append(f"\n[{source} | authority={weight:.2f}]")
            for item in items[:10]:
                upvote_tag = f" ({item.upvotes:,}↑)" if item.upvotes else ""
                form_tag = f" [{item.form_type}]" if getattr(item, "form_type", "") else ""
                grouped_lines.append(f"  • {item.title}{form_tag}{upvote_tag}")

        user_prompt = _ITEMS_USER_TEMPLATE.format(
            ticker=ticker,
            strategy=strategy_name,
            regime=regime,
            price=current_price,
            rsi=rsi,
            iv_rank=iv_rank,
            n_total=len(news_items),
            n_sources=len(grouped),
            grouped_evidence="\n".join(grouped_lines),
        )

        data = self._client.chat_json([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        if not data:
            logger.warning("[%s] FinGPT (multi-source): empty response", ticker)
            return None

        try:
            report = SentimentReport(
                ticker=ticker,
                sentiment_score=max(-1.0, min(1.0, float(data.get("sentiment_score", 0.0)))),
                event_risk=max(0.0, min(1.0, float(data.get("event_risk", 0.0)))),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                headlines=[i.title for i in news_items[:20]],
                key_themes=list(data.get("key_themes", [])),
                recommendation=str(data.get("recommendation", "neutral")),
                reasoning=str(data.get("reasoning", "")),
            )
            logger.info(
                "[%s] FinGPT multi-source (%d items, %d sources) → "
                "sentiment=%.2f  event_risk=%.2f  rec=%s",
                ticker, len(news_items), len(grouped),
                report.sentiment_score, report.event_risk, report.recommendation,
            )
            self._put_cache(ticker, report)
            return report
        except Exception as exc:
            logger.warning("[%s] FinGPT multi-source parse error: %s", ticker, exc)
            return None

    @staticmethod
    def _group_by_source(
        items: List["NewsItem"],
    ) -> Dict[str, List["NewsItem"]]:
        grouped: Dict[str, List["NewsItem"]] = {}
        for item in items:
            grouped.setdefault(item.source, []).append(item)
        return grouped

    def _get_cached(self, ticker: str) -> Optional[SentimentReport]:
        with self._lock:
            entry = self._cache.get(ticker)
            if not entry:
                return None
            report, ts = entry
            if time.monotonic() - ts < self.cache_ttl:
                report.cached = True
                logger.debug("[%s] FinGPT: cache hit", ticker)
                return report
            del self._cache[ticker]
        return None

    def _put_cache(self, ticker: str, report: SentimentReport) -> None:
        with self._lock:
            self._cache[ticker] = (report, time.monotonic())
