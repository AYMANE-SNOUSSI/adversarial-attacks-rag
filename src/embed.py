"""Embed every chunk; row i of data/embeddings.npy is line i of data/chunks.jsonl."""
import json
import time
from pathlib import Path
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "chunks.jsonl"
EMBEDDINGS_FILE = ROOT / "data" / "embeddings.npy"
INFO_FILE = ROOT / "data" / "embeddings_info.json"

args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
MODEL_NAME = args[0] if args else "intfloat/multilingual-e5-small"
WITH_TITLES = "--titles" in sys.argv  # prepend the paper title to each chunk
PASSAGE_PREFIX = "passage: "  # e5 models are trained with this prefix on documents

with CHUNKS_FILE.open(encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]
metadata = json.loads((ROOT / "data" / "metadata.json").read_text(encoding="utf-8"))


def to_text(chunk):
    """Text actually embedded for a chunk, optionally headed by its paper title."""
    body = chunk["text"]
    if WITH_TITLES:
        body = f"{metadata[chunk['arxiv_id']]['title']}. {body}"
    return PASSAGE_PREFIX + body


texts = [to_text(chunk) for chunk in chunks]

model = SentenceTransformer(MODEL_NAME)
token_counts = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
n_truncated = sum(count > model.max_seq_length for count in token_counts)

start = time.perf_counter()
embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
elapsed = time.perf_counter() - start

np.save(EMBEDDINGS_FILE, embeddings.astype(np.float32))
INFO_FILE.write_text(json.dumps({
    "with_titles": WITH_TITLES,
    "model": MODEL_NAME,
    "n_chunks": len(chunks),
    "dim": int(embeddings.shape[1]),
    "encode_seconds": round(elapsed, 1),
}, indent=2), encoding="utf-8")

print(f"Embeddings: {embeddings.shape} in {elapsed:.0f}s")
print(f"Tokens per chunk: median {int(np.median(token_counts))}, max {max(token_counts)}")
print(f"Truncated chunks (> {model.max_seq_length} tokens): {n_truncated}")