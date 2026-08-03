# AI Chef — FastAPI image
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

# Deps from the uv lockfile. PyPI's linux torch wheel pulls ~3GB of CUDA libs
# we never use (CPU inference) — install CPU-only torch first and filter the
# nvidia/triton pins out of the exported requirements.
COPY pyproject.toml uv.lock ./
RUN uv export --locked --no-hashes --no-emit-project -o /tmp/requirements.txt \
 && grep -vE '^(nvidia-|triton)' /tmp/requirements.txt > /tmp/req-cpu.txt \
 && uv venv \
 && uv pip install --index-url https://download.pytorch.org/whl/cpu \
      "torch==$(sed -n 's/^torch==\([0-9.]*\).*/\1/p' /tmp/requirements.txt)" \
 && uv pip install -r /tmp/req-cpu.txt

COPY api/ ./api/

ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/.cache/huggingface
# Bake embedding + re-rank models into the image for fast cold starts.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
