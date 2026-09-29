"""Retrieve the k chunks closest to a question (cosine similarity on normalized embeddings)."""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).resolve().parent.parent / "data"
QUERY_PREFIX = "query: "  # e5 models are trained with this prefix on questions


def load_index():
    """Load chunks and embeddings, and the name of the model that produced them."""
    info = json.loads((DATA / "embeddings_info.json").read_text(encoding="utf-8"))
    embeddings = np.load(DATA / "embeddings.npy")
    with (DATA / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    if len(chunks) != len(embeddings):
        raise RuntimeError("chunks.jsonl and embeddings.npy are out of sync: re-run embed.py")
    return info["model"], embeddings, chunks


def search(question, model, embeddings, chunks, k=5):
    """Return [(score, chunk)] for the k most similar chunks, best first."""
    query = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    scores = embeddings @ query
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in top]


if __name__ == "__main__":
    model_name, embeddings, chunks = load_index()
    model = SentenceTransformer(model_name)
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    for score, chunk in search(sys.argv[1], model, embeddings, chunks):
        title = metadata[chunk["arxiv_id"]]["title"]
        print(f"{score:.3f}  {title} (p. {chunk['page']})")
        print(f"       {chunk['text'][:200]}...\n")