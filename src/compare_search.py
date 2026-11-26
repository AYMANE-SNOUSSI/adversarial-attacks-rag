"""One-off check after moving from NumPy to ChromaDB: same vectors, same top-k for every eval question.

Needs the old index (data/embeddings.npy, data/chunks.jsonl) next to the new one (data/chroma).
"""
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from search import QUERY_PREFIX, load_index, search
from vector_store import DATA

K = 10  # eval_retrieval.py looks at the top 10

model_name, collection = load_index()
model = SentenceTransformer(model_name)
old_embeddings = np.load(DATA / "embeddings.npy")
with (DATA / "chunks.jsonl").open(encoding="utf-8") as f:
    old_ids = [json.loads(line)["chunk_id"] for line in f]

# 1. The vectors stored in Chroma are the ones NumPy used.
stored = collection.get(ids=old_ids, include=["embeddings"])
position = {chunk_id: i for i, chunk_id in enumerate(stored["ids"])}
new_embeddings = np.array([stored["embeddings"][position[chunk_id]] for chunk_id in old_ids])
print(f"Chunks: {len(old_ids)} in NumPy, {collection.count()} in Chroma")
print(f"Largest difference between stored vectors: {np.abs(new_embeddings - old_embeddings).max():.1e}")

# 2. Every question retrieves the same chunks, in the same order.
with (DATA.parent / "eval" / "questions.jsonl").open(encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]
same_top5 = same_topk = 0
for q in questions:
    query = model.encode(QUERY_PREFIX + q["question"], normalize_embeddings=True)
    old_top = [old_ids[i] for i in np.argsort(-(old_embeddings @ query))[:K]]
    new_top = [chunk["chunk_id"] for _, chunk in search(q["question"], model, collection, k=K)]
    same_top5 += old_top[:5] == new_top[:5]
    same_topk += old_top == new_top
    if old_top != new_top:
        print(f"{q['id']}: NumPy {old_top[:5]} / Chroma {new_top[:5]}")
print(f"Same top 5: {same_top5}/{len(questions)}   Same top {K}: {same_topk}/{len(questions)}")
