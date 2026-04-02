#!/bin/bash
# ============================================================
# Intelligence Layer Setup
# ============================================================
# This script installs and configures the local LLM (Ollama)
# for the trading agent's intelligence layer.
#
# Prerequisites: Ollama must be installed (https://ollama.com)
# ============================================================

set -e

echo "============================================"
echo "Trading Agent — Intelligence Layer Setup"
echo "============================================"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed."
    echo ""
    echo "Install Ollama:"
    echo "  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Or download from: https://ollama.com/download"
    echo ""
    exit 1
fi

echo "[1/4] Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✓ Ollama is running"
else
    echo "  Starting Ollama..."
    ollama serve &
    sleep 3
fi

# ============================================================
# Model Selection Guide
# ============================================================
#
# RECOMMENDED for trading decisions:
#   mistral (7B)     — Fast, good reasoning, low RAM (~4GB)
#   llama3 (8B)      — Better reasoning, needs ~5GB RAM
#   phi3 (3.8B)      — Fastest, good for quick decisions
#   deepseek-r1 (7B) — Best reasoning, needs ~5GB RAM
#
# RECOMMENDED for embeddings:
#   nomic-embed-text — 768-dim embeddings, fast, accurate
#   mxbai-embed-large — 1024-dim, slightly better quality
#
# FOR FINE-TUNING (after accumulating 50+ trades):
#   Use unsloth or Ollama's Modelfile for LoRA fine-tuning
#   Export training data with: python -m trading_agent.fine_tuning
# ============================================================

echo ""
echo "[2/4] Pulling reasoning model (mistral 7B)..."
echo "  This is the main brain for trade analysis."
ollama pull mistral

echo ""
echo "[3/4] Pulling embedding model (nomic-embed-text)..."
echo "  This powers the RAG knowledge base."
ollama pull nomic-embed-text

echo ""
echo "[4/4] Verifying models..."
echo ""
echo "Available models:"
ollama list
echo ""

# Test the models
echo "Testing reasoning model..."
RESPONSE=$(curl -s http://localhost:11434/api/chat -d '{
  "model": "mistral",
  "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
  "stream": false
}' | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',{}).get('content','FAIL'))" 2>/dev/null)

if echo "$RESPONSE" | grep -qi "ok"; then
    echo "  ✓ Reasoning model works"
else
    echo "  ⚠ Model test returned: $RESPONSE"
fi

echo "Testing embedding model..."
EMBED_RESULT=$(curl -s http://localhost:11434/api/embed -d '{
  "model": "nomic-embed-text",
  "input": ["test"]
}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('embeddings',[[]])[0]))" 2>/dev/null)

if [ "$EMBED_RESULT" -gt 0 ] 2>/dev/null; then
    echo "  ✓ Embedding model works (${EMBED_RESULT}-dimensional vectors)"
else
    echo "  ⚠ Embedding test returned unexpected result"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To enable the intelligence layer:"
echo "  1. Edit .env and set: LLM_ENABLED=true"
echo "  2. Run the agent: python -m trading_agent.agent"
echo ""
echo "The agent will now:"
echo "  • Analyze trades with LLM reasoning"
echo "  • Learn from past trade outcomes (RAG)"
echo "  • Recommend parameter adjustments over time"
echo "  • Export training data for fine-tuning"
echo ""
echo "Optional: For better reasoning, try:"
echo "  ollama pull llama3         # 8B, better quality"
echo "  ollama pull deepseek-r1    # 7B, best reasoning"
echo "  Then update LLM_MODEL in .env"
echo "============================================"
