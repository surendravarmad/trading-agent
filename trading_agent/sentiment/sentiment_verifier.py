"""
Sentiment Verifier — Reasoning Model Anti-Hallucination Layer
==============================================================
Takes FinGPT's SentimentReport plus the raw NewsItem evidence and sends
both to a stronger reasoning model to cross-check every claim.

Pattern: Specialist → Verifier
  FinGPT (fast, finance-tuned, noisy)
      ↓ SentimentReport + raw evidence
  Reasoning Model (slow, rigorous, sceptical)
      ↓ VerifiedSentimentReport

The verifier does NOT re-analyse the news from scratch. It reads FinGPT's
reasoning chain, maps each claim to a specific headline or filing, and
flags anything that is not supported by the evidence — the classic
hallucination pattern where a model confidently states a fact that
isn't actually present in the input.

Supported verifier backends:
  • "anthropic"  — Claude API (claude-sonnet-4-6, claude-opus-4-7)
                   Best reasoning quality; requires VERIFIER_API_KEY
  • "ollama"     — Local reasoning model (QwQ-32B, DeepSeek-R1, etc.)
                   Privacy-first; your M5 Pro Max runs these comfortably

Fallback: if the verifier is unavailable, the original SentimentReport
is wrapped in a VerifiedSentimentReport with agreement_score=1.0 and
no hallucination flags — the pipeline never blocks.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from trading_agent.sentiment.fingpt_analyser import SentimentReport
from trading_agent.sentiment.news_aggregator import NewsItem

if TYPE_CHECKING:
    from trading_agent.config import IntelligenceConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Output types
# --------------------------------------------------------------------------

@dataclass
class EvidenceClaim:
    """One claim from FinGPT's reasoning mapped to its supporting evidence."""
    claim: str
    support_level: str      # "supported" | "partially_supported" | "unsupported"
    evidence_ref: str       # "item N" or "none"


@dataclass
class VerifiedSentimentReport:
    """
    FinGPT's SentimentReport after independent verification by a reasoning model.
    Replaces SentimentReport as the final sentiment signal passed to LLM Analyst.
    """
    original: SentimentReport
    verified_sentiment_score: float
    verified_event_risk: float
    verified_confidence: float
    verified_recommendation: str    # "favorable" | "neutral" | "caution" | "avoid"
    verified_reasoning: str
    evidence_mapping: List[EvidenceClaim] = field(default_factory=list)
    hallucination_flags: List[str] = field(default_factory=list)
    agreement_score: float = 1.0    # 0=full disagreement, 1=full agreement
    confidence_delta: float = 0.0   # positive = verifier raised confidence
    verifier_warnings: List[str] = field(default_factory=list)
    verifier_model: str = ""
    passthrough: bool = False       # True when verifier was skipped

    # ------------------------------------------------------------------
    # SentimentReadout protocol surface
    # ------------------------------------------------------------------
    # LLMAnalyst.analyze_trade is typed against ports.SentimentReadout.
    # These aliases expose the verified_* fields under the canonical
    # names so the analyst prompt builder can consume either a raw
    # SentimentReport or this VerifiedSentimentReport without caring
    # which pipeline stage produced it.  The verified numbers are
    # authoritative — always prefer them over the originals.

    @property
    def ticker(self) -> str:
        return self.original.ticker

    @property
    def sentiment_score(self) -> float:
        return self.verified_sentiment_score

    @property
    def event_risk(self) -> float:
        return self.verified_event_risk

    @property
    def confidence(self) -> float:
        return self.verified_confidence

    @property
    def recommendation(self) -> str:
        return self.verified_recommendation

    @property
    def reasoning(self) -> str:
        return self.verified_reasoning

    @property
    def key_themes(self) -> List[str]:
        return self.original.key_themes

    @property
    def headlines(self) -> List[str]:
        return self.original.headlines

    def to_prompt_section(self) -> str:
        """Inject into LLM Analyst Phase V prompt."""
        sign = "+" if self.verified_sentiment_score >= 0 else ""
        themes = ", ".join(self.original.key_themes) if self.original.key_themes else "none"
        top_headlines = "\n".join(
            f"  • {h}" for h in self.original.headlines[:5]
        )
        flags = ""
        if self.hallucination_flags:
            flag_lines = "\n".join(f"  ⚠ {f}" for f in self.hallucination_flags)
            flags = f"\n**Hallucination flags (claims not found in evidence):**\n{flag_lines}"
        avoid_warning = (
            "\n\n> ⛔ HIGH EVENT RISK — binary catalyst detected. "
            "Strongly consider skipping premium selling."
            if self.verified_recommendation == "avoid" else ""
        )
        passthrough_note = (
            "\n_(Verification skipped — FinGPT output passed through)_"
            if self.passthrough else
            f"\n_(Verified by {self.verifier_model} — agreement={self.agreement_score:.2f})_"
        )
        return (
            f"## FinGPT Sentiment (Verified)\n"
            f"**Sentiment:** {sign}{self.verified_sentiment_score:.2f}  |  "
            f"**Event risk:** {self.verified_event_risk:.2f}  |  "
            f"**Confidence:** {self.verified_confidence:.2f}\n"
            f"**Recommendation:** {self.verified_recommendation.upper()}\n"
            f"**Key themes:** {themes}\n"
            f"**Top headlines:**\n{top_headlines}\n"
            f"**Verified reasoning:** {self.verified_reasoning}"
            f"{flags}{avoid_warning}{passthrough_note}"
        )


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a critical reasoning verifier for a financial sentiment analysis pipeline.

A specialist model called FinGPT has already analysed a set of news headlines and produced a sentiment assessment. Your role is to VERIFY that assessment — not to produce a new one from scratch.

YOUR SPECIFIC TASKS:
1. Read each claim in FinGPT's reasoning
2. Find the specific headline/filing in the evidence list that supports each claim
3. Flag any claim that is NOT supported by any item in the evidence (hallucination)
4. Based on evidence quality, confirm or adjust the sentiment score, event risk, and recommendation

WHAT COUNTS AS A HALLUCINATION:
  • "Earnings announcement expected in X days" — only flag if no evidence mentions earnings
  • "Fed meeting on [date]" — only flag if Fed items don't mention a meeting
  • "Company announced [specific fact]" — only flag if no filing or headline says this
  • General financial knowledge (e.g. "rising rates hurt growth stocks") is NOT a hallucination

CALIBRATION:
  agreement_score 0.9–1.0  → FinGPT fully supported, maintain recommendation
  agreement_score 0.6–0.9  → Minor gaps, slight confidence reduction
  agreement_score 0.3–0.6  → Material unsupported claims, reduce confidence, possibly revise recommendation
  agreement_score < 0.3    → Major contradictions, revise recommendation

Be conservative: only override FinGPT when evidence clearly contradicts it. Absence of evidence for a claim is a partial flag, not automatic reversal.

Respond ONLY in valid JSON (no markdown fences, no preamble):
{
  "evidence_mapping": [
    {"claim": "...", "support_level": "supported|partially_supported|unsupported", "evidence_ref": "item N or none"}
  ],
  "hallucination_flags": ["only genuinely unsupported factual claims"],
  "verified_sentiment_score": <float -1.0 to 1.0>,
  "verified_event_risk": <float 0.0 to 1.0>,
  "verified_confidence": <float 0.0 to 1.0>,
  "verified_recommendation": "favorable|neutral|caution|avoid",
  "verified_reasoning": "<2-3 sentences: what the evidence actually supports>",
  "agreement_score": <float 0.0 to 1.0>,
  "verifier_warnings": ["optional list of concerns"]
}"""


_USER_TEMPLATE = """## FinGPT Analysis to Verify

**Ticker:** {ticker}
**FinGPT Specialist Output:**
- Sentiment score: {sentiment_score:+.2f}
- Event risk: {event_risk:.2f}
- Confidence: {confidence:.2f}
- Recommendation: {recommendation}
- Key themes: {themes}
- FinGPT reasoning: {reasoning}

## Raw Evidence ({n_items} items, sorted by source authority)

{evidence_lines}

## Your Task

Verify the FinGPT analysis above against the raw evidence.
Map each claim in FinGPT's reasoning to a specific evidence item.
Flag any factual claims not found in the evidence.
Output verified scores and a corrected recommendation if warranted."""


# --------------------------------------------------------------------------
# SentimentVerifier
# --------------------------------------------------------------------------

class SentimentVerifier:
    """
    Reasoning model that cross-checks FinGPT output against raw news evidence.

    Initialise with provider="anthropic" for Claude or provider="ollama"
    for a local reasoning model. Falls back to passthrough if unavailable.
    """

    def __init__(
        self,
        cfg: Optional["IntelligenceConfig"] = None,
        enabled: Optional[bool] = None,
        max_evidence_items: int = 40,
        # Legacy kwargs — retained for ad-hoc instantiation in tests.
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if cfg is not None:
            self.provider = provider or cfg.verifier_provider
            self.model = model or cfg.verifier_model
            self.api_key = api_key if api_key is not None else cfg.verifier_api_key
            self._base_url = base_url or cfg.llm_base_url
            self._cfg = cfg
            effective_enabled = (
                enabled if enabled is not None else bool(cfg.verifier_enabled)
            )
        else:
            self.provider = provider or "ollama"
            self.model = model or "qwq:32b"
            self.api_key = api_key or ""
            self._base_url = base_url or "http://localhost:11434"
            self._cfg = None
            effective_enabled = bool(enabled) if enabled is not None else True

        self.enabled = effective_enabled
        self.max_evidence_items = max_evidence_items
        self._client = None

        if self.enabled:
            self._client = self._build_client()
            logger.info(
                "SentimentVerifier ENABLED — provider=%s, model=%s",
                self.provider, self.model,
            )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def verify(
        self,
        sentiment: SentimentReport,
        news_items: List[NewsItem],
    ) -> VerifiedSentimentReport:
        """
        Verify FinGPT's SentimentReport against raw news evidence.
        Returns a passthrough VerifiedSentimentReport if verifier unavailable.
        """
        if not self.enabled or not self._client:
            return self._passthrough(sentiment)

        evidence = news_items[: self.max_evidence_items]
        prompt = self._build_prompt(sentiment, evidence)

        try:
            data = self._infer(prompt)
            if not data:
                logger.warning(
                    "[%s] Verifier returned empty — passing through FinGPT output",
                    sentiment.ticker,
                )
                return self._passthrough(sentiment)
            return self._parse(sentiment, data)
        except Exception as exc:
            logger.warning(
                "[%s] Verifier inference failed: %s — passing through",
                sentiment.ticker, exc,
            )
            return self._passthrough(sentiment)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_client(self):
        """Build the appropriate inference client."""
        if self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning(
                    "anthropic SDK not installed (pip install anthropic). "
                    "Falling back to Ollama for verification."
                )
                # Fall through to the non-anthropic path
                self.provider = "ollama"

        # Ollama / OpenAI-compat — prefer the factory so per-role tuning
        # lives in IntelligenceConfig rather than magic numbers here.
        from trading_agent.intelligence.llm_client import LLMClient, LLMConfig, make_llm_client

        if self._cfg is not None:
            return make_llm_client("verifier", self._cfg)

        # Legacy no-config path (tests) — keep historical defaults
        return LLMClient(LLMConfig(
            provider="ollama" if self.provider not in ("openai", "lmstudio") else self.provider,
            base_url=self._base_url,
            model=self.model,
            api_key=self.api_key,
            temperature=0.15,
            max_tokens=2048,
            timeout=90,
        ))

    def _build_prompt(
        self, sentiment: SentimentReport, evidence: List[NewsItem]
    ) -> str:
        evidence_lines = "\n".join(
            f"{i+1}. {item.as_evidence_line()}"
            for i, item in enumerate(evidence)
        )
        return _USER_TEMPLATE.format(
            ticker=sentiment.ticker,
            sentiment_score=sentiment.sentiment_score,
            event_risk=sentiment.event_risk,
            confidence=sentiment.confidence,
            recommendation=sentiment.recommendation,
            themes=", ".join(sentiment.key_themes) or "none",
            reasoning=sentiment.reasoning,
            n_items=len(evidence),
            evidence_lines=evidence_lines or "(no evidence items)",
        )

    def _infer(self, user_prompt: str) -> Optional[Dict]:
        """Route to the correct backend and return parsed JSON."""
        if self.provider == "anthropic":
            return self._infer_anthropic(user_prompt)
        # Ollama / OpenAI-compat via LLMClient
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._client.chat_json(messages)

    def _infer_anthropic(self, user_prompt: str) -> Optional[Dict]:
        """Call Anthropic Messages API directly."""
        try:
            import anthropic
            max_tokens = self._cfg.verifier_max_tokens if self._cfg else 2048
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = resp.content[0].text if resp.content else ""
            if not raw:
                return None
            # Strip any accidental fences
            raw = raw.strip()
            for fence in ("```json", "```"):
                if raw.startswith(fence):
                    raw = raw[len(fence):]
            if raw.endswith("```"):
                raw = raw[:-3]
            return json.loads(raw.strip())
        except Exception as exc:
            logger.warning("Anthropic verifier call failed: %s", exc)
            return None

    def _parse(
        self, original: SentimentReport, data: Dict
    ) -> VerifiedSentimentReport:
        """Parse verifier JSON into VerifiedSentimentReport."""
        # Evidence mapping
        mapping: List[EvidenceClaim] = []
        for entry in data.get("evidence_mapping", []):
            if isinstance(entry, dict):
                mapping.append(EvidenceClaim(
                    claim=str(entry.get("claim", "")),
                    support_level=str(entry.get("support_level", "unsupported")),
                    evidence_ref=str(entry.get("evidence_ref", "none")),
                ))

        v_score = max(-1.0, min(1.0, float(data.get("verified_sentiment_score", original.sentiment_score))))
        v_risk = max(0.0, min(1.0, float(data.get("verified_event_risk", original.event_risk))))
        v_conf = max(0.0, min(1.0, float(data.get("verified_confidence", original.confidence))))
        agreement = max(0.0, min(1.0, float(data.get("agreement_score", 1.0))))

        report = VerifiedSentimentReport(
            original=original,
            verified_sentiment_score=v_score,
            verified_event_risk=v_risk,
            verified_confidence=v_conf,
            verified_recommendation=str(data.get("verified_recommendation", original.recommendation)),
            verified_reasoning=str(data.get("verified_reasoning", original.reasoning)),
            evidence_mapping=mapping,
            hallucination_flags=list(data.get("hallucination_flags", [])),
            agreement_score=agreement,
            confidence_delta=v_conf - original.confidence,
            verifier_warnings=list(data.get("verifier_warnings", [])),
            verifier_model=self.model,
            passthrough=False,
        )
        logger.info(
            "[%s] Verifier: agreement=%.2f  flags=%d  rec=%s→%s",
            original.ticker, agreement,
            len(report.hallucination_flags),
            original.recommendation,
            report.verified_recommendation,
        )
        return report

    @staticmethod
    def _passthrough(sentiment: SentimentReport) -> VerifiedSentimentReport:
        """Wrap original SentimentReport with no changes — pipeline safe fallback."""
        return VerifiedSentimentReport(
            original=sentiment,
            verified_sentiment_score=sentiment.sentiment_score,
            verified_event_risk=sentiment.event_risk,
            verified_confidence=sentiment.confidence,
            verified_recommendation=sentiment.recommendation,
            verified_reasoning=sentiment.reasoning,
            evidence_mapping=[],
            hallucination_flags=[],
            agreement_score=1.0,
            confidence_delta=0.0,
            verifier_warnings=[],
            verifier_model="passthrough",
            passthrough=True,
        )
