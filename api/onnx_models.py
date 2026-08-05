"""ONNX Runtime wrappers for the embedding and re-rank models (no torch).

Used when MODEL_BACKEND=onnx (the Docker images). The fp32 ONNX graphs and
tokenizer.json are downloaded from the official Hugging Face repos at image
build time and are numerically equivalent to the sentence-transformers models
used by pipeline/ingest.py, so query embeddings stay compatible with the
pgvector knowledge base.

Layout expected under MODELS_DIR (default ./models):
    models/embedder/{model.onnx,tokenizer.json}   all-MiniLM-L6-v2
    models/reranker/{model.onnx,tokenizer.json}   ms-marco-MiniLM-L-6-v2
"""

import os
from pathlib import Path

import numpy as np

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
EMBEDDER_MAX_LEN = 256  # all-MiniLM-L6-v2 truncates at 256 word pieces
RERANKER_MAX_LEN = 512


def _session(model_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _tokenizer(model_dir: Path, max_len: int):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding()  # pad to longest in batch — batched ONNX needs uniform length
    return tok


def _inputs(session, encodings) -> dict:
    """Build the ONNX input dict, keeping only what the graph asks for."""
    arrays = {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
    }
    names = {i.name for i in session.get_inputs()}
    return {k: v for k, v in arrays.items() if k in names}


class OnnxEmbedder:
    """Drop-in for SentenceTransformer.encode (mean pooling + normalize)."""

    def __init__(self, model_dir: Path):
        self._tokenizer = _tokenizer(model_dir, EMBEDDER_MAX_LEN)
        self._session = _session(model_dir / "model.onnx")

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        enc = self._tokenizer.encode_batch(texts)
        outputs = {o.name: o for o in self._session.get_outputs()}
        name = "last_hidden_state" if "last_hidden_state" in outputs else next(
            n for n, o in outputs.items() if len(o.shape) == 3
        )
        token_emb = self._session.run([name], _inputs(self._session, enc))[0]
        mask = np.array([e.attention_mask for e in enc], dtype=np.float32)[..., None]
        emb = (token_emb * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        if normalize_embeddings:
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb


class OnnxReranker:
    """Drop-in for CrossEncoder.predict (pair → relevance logit)."""

    def __init__(self, model_dir: Path):
        self._tokenizer = _tokenizer(model_dir, RERANKER_MAX_LEN)
        self._session = _session(model_dir / "model.onnx")

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        enc = self._tokenizer.encode_batch(pairs)
        logits = self._session.run(None, _inputs(self._session, enc))[0]
        return logits.reshape(-1)
