# AI Chef — FastAPI image (torch-free: embedding + re-rank run on ONNX Runtime)

# -- builder: resolve locked deps into a venv (uv stays in this stage) --------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv export --locked --no-hashes --no-emit-project --no-default-groups \
      -o /tmp/requirements.txt \
 && uv venv \
 && uv pip install --no-cache -r /tmp/requirements.txt

# -- final --------------------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

# Bake fp32 ONNX graphs + tokenizers into the image (official HF repos —
# numerically equivalent to the sentence-transformers models used at ingestion)
RUN python - <<'EOF'
import pathlib
import urllib.request

FILES = {
    "models/embedder/model.onnx":
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
    "models/embedder/tokenizer.json":
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
    "models/reranker/model.onnx":
        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx",
    "models/reranker/tokenizer.json":
        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json",
}
for dest, url in FILES.items():
    p = pathlib.Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {p}")
    urllib.request.urlretrieve(url, p)
EOF

COPY api/ ./api/

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_BACKEND=onnx \
    MODELS_DIR=/app/models

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
