"""
Knowledge Base (RAG)
=====================
Vector-based retrieval-augmented generation system that stores trade
history and market lessons as searchable embeddings.

Uses a lightweight file-based vector store (no external dependencies
like ChromaDB required) with cosine similarity search. Can optionally
use ChromaDB when available for better performance at scale.

The knowledge base enables the LLM analyst to:
  1. Find similar past trades (by market conditions)
  2. Recall lessons from wins and losses
  3. Identify pattern-based edge cases
  4. Build institutional memory over time
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class KBDocument:
    """A single document in the knowledge base."""
    doc_id: str
    text: str                          # Human-readable content
    embedding: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    category: str = ""                 # "trade", "lesson", "market_condition", "strategy_note"
    timestamp: str = ""
    source_trade_id: str = ""


class KnowledgeBase:
    """
    File-based vector knowledge base with cosine similarity search.

    Storage:
      kb_dir/
        documents/
          {doc_id}.json       — document + embedding
        index.json            — lightweight document index
        categories/
          trades.json         — trade-specific documents
          lessons.json        — extracted lessons
          strategy_notes.json — strategy pattern notes
    """

    def __init__(self, kb_dir: str = "knowledge_base",
                 embed_fn=None):
        """
        Args:
            kb_dir: Directory to store knowledge base files
            embed_fn: Function that takes List[str] and returns List[List[float]]
                      (provided by LLMClient.embed)
        """
        self.kb_dir = kb_dir
        self.docs_dir = os.path.join(kb_dir, "documents")
        self.index_path = os.path.join(kb_dir, "index.json")
        self.embed_fn = embed_fn
        os.makedirs(self.docs_dir, exist_ok=True)

        # In-memory cache of embeddings for fast search
        self._embedding_cache: Dict[str, List[float]] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Add documents
    # ------------------------------------------------------------------

    def add_trade(self, trade_id: str, text: str,
                  metadata: Dict = None) -> str:
        """Add a completed trade to the knowledge base."""
        doc_id = f"trade_{trade_id}"
        return self._add_document(
            doc_id=doc_id, text=text, category="trade",
            metadata=metadata or {}, source_trade_id=trade_id)

    def add_lesson(self, lesson_text: str, trade_id: str = "",
                   metadata: Dict = None) -> str:
        """Add a lesson learned from trade analysis."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        doc_id = f"lesson_{ts}"
        return self._add_document(
            doc_id=doc_id, text=lesson_text, category="lesson",
            metadata=metadata or {}, source_trade_id=trade_id)

    def add_strategy_note(self, note: str, strategy: str = "",
                          metadata: Dict = None) -> str:
        """Add a strategy insight or pattern observation."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        doc_id = f"strategy_{ts}"
        meta = metadata or {}
        meta["strategy"] = strategy
        return self._add_document(
            doc_id=doc_id, text=note, category="strategy_note",
            metadata=meta)

    def update_trade_outcome(self, trade_id: str, outcome_label: str,
                             realized_pl: float, exit_signal: str,
                             exit_reason: str, updated_text: str = "") -> bool:
        """
        Back-fill the outcome into an existing trade KB document when the
        trade closes.  Without this, all trade documents look identical
        (no win/loss label) and the model can't learn what distinguishes
        a winning setup from a losing one.

        Parameters
        ----------
        trade_id      : matches the trade_id used in add_trade()
        outcome_label : "win" | "loss" | "breakeven" | "expired_worthless"
        realized_pl   : final P&L in dollars
        exit_signal   : "profit_target" | "stop_loss" | "regime_shift" etc.
        exit_reason   : human-readable reason string
        updated_text  : if provided, replace the document text entirely
                        (pass trade.to_embedding_text() for canonical format)

        Returns True if the document was found and updated.
        """
        doc_id = f"trade_{trade_id}"
        path = os.path.join(self.docs_dir, f"{doc_id}.json")
        if not os.path.exists(path):
            logger.warning("KB update_trade_outcome: doc %s not found", doc_id)
            return False

        try:
            with open(path) as fh:
                data = json.load(fh)

            data["metadata"]["outcome_label"] = outcome_label
            data["metadata"]["realized_pl"] = realized_pl
            data["metadata"]["exit_signal"] = exit_signal
            data["metadata"]["exit_reason"] = exit_reason
            data["metadata"]["closed_at"] = datetime.utcnow().isoformat()

            if updated_text:
                data["text"] = updated_text
                # Re-generate embedding for the richer post-outcome text
                if self.embed_fn:
                    try:
                        embs = self.embed_fn([updated_text])
                        if embs:
                            data["embedding"] = embs[0]
                            self._embedding_cache[doc_id] = embs[0]
                    except Exception as exc:
                        logger.warning("Re-embed after outcome update failed: %s", exc)

            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)

            logger.info("KB: outcome back-filled for %s → %s (P&L=$%.2f)",
                        doc_id, outcome_label, realized_pl)
            return True

        except Exception as exc:
            logger.error("KB update_trade_outcome failed for %s: %s", doc_id, exc)
            return False

    def add_market_condition(self, text: str,
                             metadata: Dict = None) -> str:
        """Add a market condition observation."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        doc_id = f"market_{ts}"
        return self._add_document(
            doc_id=doc_id, text=text, category="market_condition",
            metadata=metadata or {})

    def _add_document(self, doc_id: str, text: str, category: str,
                      metadata: Dict, source_trade_id: str = "") -> str:
        """Core document addition with embedding generation."""
        # Generate embedding
        embedding = []
        if self.embed_fn:
            try:
                embeddings = self.embed_fn([text])
                if embeddings:
                    embedding = embeddings[0]
            except Exception as exc:
                logger.warning("Failed to generate embedding for %s: %s",
                               doc_id, exc)

        doc = KBDocument(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
            category=category,
            timestamp=datetime.utcnow().isoformat(),
            source_trade_id=source_trade_id,
        )

        # Save document
        path = os.path.join(self.docs_dir, f"{doc_id}.json")
        with open(path, "w") as f:
            json.dump({
                "doc_id": doc.doc_id,
                "text": doc.text,
                "embedding": doc.embedding,
                "metadata": doc.metadata,
                "category": doc.category,
                "timestamp": doc.timestamp,
                "source_trade_id": doc.source_trade_id,
            }, f, indent=2)

        # Update cache
        if embedding:
            self._embedding_cache[doc_id] = embedding

        # Update index
        self._update_index(doc)

        logger.info("KB: Added document %s (category=%s, has_embedding=%s)",
                     doc_id, category, bool(embedding))
        return doc_id

    # ------------------------------------------------------------------
    # Search / Retrieval
    # ------------------------------------------------------------------

    def search_similar(self, query: str, top_k: int = 5,
                       category: str = None) -> List[Tuple[KBDocument, float]]:
        """
        Find documents most similar to the query using cosine similarity.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            category: Optional filter by document category

        Returns:
            List of (document, similarity_score) tuples, highest first
        """
        if not self.embed_fn:
            logger.warning("No embedding function — falling back to keyword search")
            return self._keyword_search(query, top_k, category)

        # Generate query embedding
        try:
            query_embeddings = self.embed_fn([query])
            if not query_embeddings:
                return self._keyword_search(query, top_k, category)
            query_emb = query_embeddings[0]
        except Exception as exc:
            logger.warning("Query embedding failed: %s", exc)
            return self._keyword_search(query, top_k, category)

        # Score all cached documents
        scores = []
        for doc_id, doc_emb in self._embedding_cache.items():
            if category:
                doc = self._load_document(doc_id)
                if doc and doc.category != category:
                    continue
            similarity = self._cosine_similarity(query_emb, doc_emb)
            scores.append((doc_id, similarity))

        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scores[:top_k]:
            doc = self._load_document(doc_id)
            if doc:
                results.append((doc, score))

        return results

    def get_similar_trades(self, trade_text: str,
                           top_k: int = 5) -> List[Tuple[KBDocument, float]]:
        """Find similar past trades by market conditions."""
        return self.search_similar(trade_text, top_k, category="trade")

    def get_relevant_lessons(self, context: str,
                             top_k: int = 5) -> List[Tuple[KBDocument, float]]:
        """Find lessons relevant to the current market context."""
        return self.search_similar(context, top_k, category="lesson")

    def get_strategy_notes(self, strategy: str,
                           top_k: int = 5) -> List[Tuple[KBDocument, float]]:
        """Find notes relevant to a specific strategy."""
        return self.search_similar(
            f"Strategy: {strategy}", top_k, category="strategy_note")

    def query_by_metadata(self, filters: Dict,
                          top_k: int = 20) -> List[KBDocument]:
        """
        Return documents whose metadata matches ALL key-value pairs in
        *filters*.  Useful for targeted fine-tuning queries such as:

            # All losing Bear Call Spreads
            kb.query_by_metadata({"outcome_label": "loss",
                                  "strategy": "Bear Call Spread"})

            # All wins in a bearish regime
            kb.query_by_metadata({"outcome_label": "win",
                                  "regime": "bearish"})

        Metadata keys set by the agent: outcome_label, realized_pl,
        exit_signal, strategy, regime, ticker.
        """
        index = self._load_index()
        results = []
        for entry in index.get("documents", []):
            doc = self._load_document(entry["doc_id"])
            if doc is None:
                continue
            if all(doc.metadata.get(k) == v for k, v in filters.items()):
                results.append(doc)
            if len(results) >= top_k:
                break
        return results

    def outcome_stats(self) -> Dict:
        """
        Aggregate win/loss counts by strategy and regime.
        Useful for surfacing which setups have the best historical edge.

            {
              "Bull Put Spread": {"bullish": {"win": 12, "loss": 3}},
              "Bear Call Spread": {"bearish": {"win": 8,  "loss": 5}},
              ...
            }
        """
        index = self._load_index()
        stats: Dict = {}
        for entry in index.get("documents", []):
            if entry.get("category") != "trade":
                continue
            doc = self._load_document(entry["doc_id"])
            if not doc:
                continue
            outcome = doc.metadata.get("outcome_label")
            if not outcome:
                continue
            strategy = doc.metadata.get("strategy", "unknown")
            regime = doc.metadata.get("regime", "unknown")
            stats.setdefault(strategy, {}).setdefault(regime, {"win": 0, "loss": 0, "other": 0})
            bucket = outcome if outcome in ("win", "loss") else "other"
            stats[strategy][regime][bucket] += 1
        return stats

    # ------------------------------------------------------------------
    # Keyword fallback search
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str, top_k: int,
                        category: str = None) -> List[Tuple[KBDocument, float]]:
        """Simple keyword-based search when embeddings aren't available."""
        query_words = set(query.lower().split())
        index = self._load_index()

        scores = []
        for entry in index.get("documents", []):
            if category and entry.get("category") != category:
                continue
            doc = self._load_document(entry["doc_id"])
            if not doc:
                continue
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Bulk operations for fine-tuning
    # ------------------------------------------------------------------

    def get_all_trades(self) -> List[KBDocument]:
        """Get all trade documents for fine-tuning data export."""
        index = self._load_index()
        trades = []
        for entry in index.get("documents", []):
            if entry.get("category") == "trade":
                doc = self._load_document(entry["doc_id"])
                if doc:
                    trades.append(doc)
        return trades

    def get_all_lessons(self) -> List[KBDocument]:
        """Get all lesson documents."""
        index = self._load_index()
        lessons = []
        for entry in index.get("documents", []):
            if entry.get("category") == "lesson":
                doc = self._load_document(entry["doc_id"])
                if doc:
                    lessons.append(doc)
        return lessons

    def document_count(self) -> Dict[str, int]:
        """Count documents by category."""
        index = self._load_index()
        counts = {}
        for entry in index.get("documents", []):
            cat = entry.get("category", "unknown")
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Internal storage
    # ------------------------------------------------------------------

    def _load_document(self, doc_id: str) -> Optional[KBDocument]:
        path = os.path.join(self.docs_dir, f"{doc_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return KBDocument(
                doc_id=data["doc_id"],
                text=data["text"],
                embedding=data.get("embedding", []),
                metadata=data.get("metadata", {}),
                category=data.get("category", ""),
                timestamp=data.get("timestamp", ""),
                source_trade_id=data.get("source_trade_id", ""),
            )
        except Exception as exc:
            logger.error("Failed to load document %s: %s", doc_id, exc)
            return None

    def _load_index(self) -> Dict:
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                return json.load(f)
        return {"documents": []}

    def _update_index(self, doc: KBDocument):
        index = self._load_index()
        docs = index.get("documents", [])

        # Update or add
        found = False
        for i, d in enumerate(docs):
            if d["doc_id"] == doc.doc_id:
                docs[i] = {
                    "doc_id": doc.doc_id,
                    "category": doc.category,
                    "timestamp": doc.timestamp,
                    "has_embedding": bool(doc.embedding),
                }
                found = True
                break

        if not found:
            docs.append({
                "doc_id": doc.doc_id,
                "category": doc.category,
                "timestamp": doc.timestamp,
                "has_embedding": bool(doc.embedding),
            })

        index["documents"] = docs
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _load_cache(self):
        """Load all embeddings into memory for fast search."""
        if not os.path.exists(self.docs_dir):
            return
        count = 0
        for fname in os.listdir(self.docs_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.docs_dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                emb = data.get("embedding", [])
                if emb:
                    self._embedding_cache[data["doc_id"]] = emb
                    count += 1
            except Exception:
                pass
        if count:
            logger.info("KB: Loaded %d embeddings into cache", count)
