"""Retrieve the k chunks closest to a question (cosine similarity, computed by ChromaDB)."""
import json
import sys

from vector_store import DATA, get_collection, open_client

QUERY_PREFIX = "query: "  # e5 models are trained with this prefix on questions


def load_index():
    """Open the vector database, and read the name of the model that produced its vectors."""
    info = json.loads((DATA / "embeddings_info.json").read_text(encoding="utf-8"))
    collection = get_collection(open_client())
    if collection.count() != info["n_chunks"]:
        raise RuntimeError("The vector database is incomplete: re-run embed.py")
    return info["model"], collection


def search(question, model, collection, k=5):
    """Return [(score, chunk)] for the k most similar chunks, best first."""
    query = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    result = collection.query(
        query_embeddings=[query], n_results=k, include=["documents", "metadatas", "distances"]
    )
    # One query was sent, so each field holds one list; cosine distance = 1 - cosine similarity.
    return [
        (1 - distance, {"chunk_id": chunk_id, "text": text, **meta})
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model_name, collection = load_index()
    model = SentenceTransformer(model_name)
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    for score, chunk in search(sys.argv[1], model, collection):
        title = metadata[chunk["arxiv_id"]]["title"]
        print(f"{score:.3f}  {title} (p. {chunk['page']})")
        print(f"       {chunk['text'][:200]}...\n")
