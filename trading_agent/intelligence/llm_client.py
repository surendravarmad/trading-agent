"""
LLM Client
============
OpenAI-compatible client that works with:
  - Ollama (local, default)
  - LM Studio (local)
  - Claude API via proxy
  - Any OpenAI-compatible endpoint

Provides a unified interface for the trading agent's intelligence layer.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import requests

if TYPE_CHECKING:
    from trading_agent.config import IntelligenceConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "ollama"          # "ollama", "lmstudio", "openai", "anthropic"
    base_url: str = "http://localhost:11434"  # Ollama default
    model: str = "mistral"            # Default model
    embedding_model: str = "nomic-embed-text"  # For RAG embeddings
    api_key: str = ""                 # Only needed for cloud providers
    temperature: float = 0.3          # Low temp for consistent trading decisions
    max_tokens: int = 2048
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
            model=os.getenv("LLM_MODEL", "mistral"),
            embedding_model=os.getenv("LLM_EMBEDDING_MODEL", "nomic-embed-text"),
            api_key=os.getenv("LLM_API_KEY", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        )


class LLMClient:
    """
    Unified LLM client with OpenAI-compatible interface.

    Supports:
      - Chat completions (reasoning, analysis, decisions)
      - Embeddings (for RAG vector store)
      - Structured JSON output (for trade decisions)
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._api_url = self._resolve_api_url()
        self._embed_url = self._resolve_embed_url()
        logger.info("LLM Client initialized: provider=%s, model=%s, url=%s",
                     self.config.provider, self.config.model, self._api_url)

    def _resolve_api_url(self) -> str:
        """Build the chat completions endpoint URL."""
        base = self.config.base_url.rstrip("/")
        if self.config.provider == "ollama":
            return f"{base}/api/chat"
        elif self.config.provider in ("lmstudio", "openai"):
            return f"{base}/v1/chat/completions"
        elif self.config.provider == "anthropic":
            return f"{base}/v1/messages"
        return f"{base}/v1/chat/completions"

    def _resolve_embed_url(self) -> str:
        """Build the embeddings endpoint URL."""
        base = self.config.base_url.rstrip("/")
        if self.config.provider == "ollama":
            return f"{base}/api/embed"
        return f"{base}/v1/embeddings"

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    def chat(self, messages: List[Dict], temperature: float = None,
             json_mode: bool = False) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: Override default temperature
            json_mode: Request JSON-formatted response

        Returns:
            The assistant's response text.
        """
        temp = temperature if temperature is not None else self.config.temperature

        if self.config.provider == "ollama":
            return self._chat_ollama(messages, temp, json_mode)
        else:
            return self._chat_openai_compat(messages, temp, json_mode)

    def _chat_ollama(self, messages: List[Dict], temperature: float,
                     json_mode: bool) -> str:
        """Ollama native /api/chat endpoint."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            resp = requests.post(
                self._api_url, json=payload,
                timeout=self.config.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            logger.debug("LLM response (%d chars): %s...",
                         len(content), content[:200])
            return content

        except requests.RequestException as exc:
            logger.error("LLM chat failed: %s", exc)
            return ""

    def _chat_openai_compat(self, messages: List[Dict], temperature: float,
                            json_mode: bool) -> str:
        """OpenAI-compatible /v1/chat/completions endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(
                self._api_url, json=payload, headers=headers,
                timeout=self.config.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content

        except requests.RequestException as exc:
            logger.error("LLM chat failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Structured JSON output
    # ------------------------------------------------------------------

    def chat_json(self, messages: List[Dict],
                  temperature: float = None) -> Optional[Dict]:
        """
        Chat completion that parses the response as JSON.
        Falls back to extracting JSON from markdown code blocks.
        """
        raw = self.chat(messages, temperature=temperature, json_mode=True)
        if not raw:
            return None

        # Try direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` blocks
        if "```json" in raw:
            try:
                start = raw.index("```json") + 7
                end = raw.index("```", start)
                return json.loads(raw[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass

        # Try extracting any JSON object
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Could not parse LLM response as JSON: %s...",
                           raw[:200])
            return None

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Uses Ollama /api/embed or OpenAI-compatible /v1/embeddings.
        """
        if self.config.provider == "ollama":
            return self._embed_ollama(texts)
        else:
            return self._embed_openai_compat(texts)

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        """Ollama native /api/embed endpoint."""
        payload = {
            "model": self.config.embedding_model,
            "input": texts,
        }
        try:
            resp = requests.post(
                self._embed_url, json=payload,
                timeout=self.config.timeout)
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            logger.debug("Generated %d embeddings (dim=%d)",
                         len(embeddings),
                         len(embeddings[0]) if embeddings else 0)
            return embeddings

        except requests.RequestException as exc:
            logger.error("Embedding generation failed: %s", exc)
            return []

    def _embed_openai_compat(self, texts: List[str]) -> List[List[float]]:
        """OpenAI-compatible /v1/embeddings endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.embedding_model,
            "input": texts,
        }
        try:
            resp = requests.post(
                self._embed_url, json=payload, headers=headers,
                timeout=self.config.timeout)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

        except requests.RequestException as exc:
            logger.error("Embedding generation failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the LLM service is reachable."""
        try:
            if self.config.provider == "ollama":
                resp = requests.get(
                    f"{self.config.base_url.rstrip('/')}/api/tags",
                    timeout=5)
            else:
                resp = requests.get(
                    f"{self.config.base_url.rstrip('/')}/v1/models",
                    timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        """List available models on the server."""
        try:
            if self.config.provider == "ollama":
                resp = requests.get(
                    f"{self.config.base_url.rstrip('/')}/api/tags",
                    timeout=5)
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
            else:
                resp = requests.get(
                    f"{self.config.base_url.rstrip('/')}/v1/models",
                    timeout=5)
                resp.raise_for_status()
                return [m["id"] for m in resp.json().get("data", [])]
        except requests.RequestException:
            return []


# ---------------------------------------------------------------------------
# Role-based LLM factory
# ---------------------------------------------------------------------------
# One ``.env`` fans out to three differently-tuned LLM callers
# (analyst / fingpt / verifier).  Each role has historically hard-coded
# its own ``LLMConfig`` inside the consumer module — that produced
# silent config drift where the ``.env`` said one thing and the code
# another.  The factory centralises the per-role profile so the
# intelligence modules only have to ask for ``make_llm_client("fingpt", cfg)``.
#
# Roles:
#   "analyst"  — primary chain-of-thought trade approver.
#                Uses LLM_*/llm_temperature from the general LLM section.
#   "fingpt"   — finance-tuned specialist scoring news.
#                Low temperature, short response, short timeout.
#   "verifier" — reasoning model that cross-checks FinGPT claims.
#                Longer timeout, larger response budget, provider may
#                differ from the other two (e.g. cloud Claude vs local Ollama).

_VALID_ROLES = {"analyst", "fingpt", "verifier"}


def make_llm_client(role: str, cfg: "IntelligenceConfig") -> "LLMClient":
    """
    Build an ``LLMClient`` tuned for the given role from one
    :class:`IntelligenceConfig`.

    Raises ``ValueError`` for unknown roles so a typo surfaces immediately
    instead of silently using analyst defaults.
    """
    if role not in _VALID_ROLES:
        raise ValueError(
            f"make_llm_client: unknown role {role!r} "
            f"(valid: {sorted(_VALID_ROLES)})"
        )

    if role == "analyst":
        return LLMClient(LLMConfig(
            provider=cfg.llm_provider,
            base_url=cfg.llm_base_url,
            model=cfg.llm_model,
            embedding_model=cfg.llm_embedding_model,
            api_key=cfg.llm_api_key,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.analyst_max_tokens,
            timeout=cfg.analyst_timeout,
        ))

    if role == "fingpt":
        # FinGPT currently only supports Ollama in practice — the GGUF
        # weights referenced by ``fingpt_model`` are pulled into an
        # Ollama registry.  We reuse the shared ``llm_base_url`` rather
        # than adding a separate knob; deploy both models on the same
        # local instance.
        return LLMClient(LLMConfig(
            provider="ollama",
            base_url=cfg.llm_base_url,
            model=cfg.fingpt_model,
            embedding_model=cfg.llm_embedding_model,
            api_key="",
            temperature=cfg.fingpt_temperature,
            max_tokens=cfg.fingpt_max_tokens,
            timeout=cfg.fingpt_timeout,
        ))

    # verifier
    # If provider=="anthropic" the caller (SentimentVerifier) will use
    # the anthropic SDK directly — this LLMClient instance only serves
    # the Ollama / OpenAI-compat paths.  We still build it so the
    # fallback is ready when the anthropic SDK isn't installed.
    provider = cfg.verifier_provider if cfg.verifier_provider != "anthropic" else "ollama"
    return LLMClient(LLMConfig(
        provider=provider,
        base_url=cfg.llm_base_url,
        model=cfg.verifier_model,
        embedding_model=cfg.llm_embedding_model,
        api_key=cfg.verifier_api_key,
        temperature=cfg.verifier_temperature,
        max_tokens=cfg.verifier_max_tokens,
        timeout=cfg.verifier_timeout,
    ))


__all__ = ["LLMClient", "LLMConfig", "make_llm_client"]
