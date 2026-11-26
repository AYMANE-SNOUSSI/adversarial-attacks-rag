"""Vector database: each chunk is stored in ChromaDB with its embedding, its text, its paper and its page."""
from pathlib import Path

import chromadb
from chromadb.config import Settings

DATA = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA / "chroma"
COLLECTION = "chunks"

# "cosine": rank by cosine similarity (Chroma's default is L2 distance).
# "ef_search": candidates the HNSW index explores per query. On synthetic vectors of the same size, the
# default (100) gave a different top 5 than exact NumPy search on 14 to 45% of queries; at 1000, the top 10 always matched.
HNSW = {"space": "cosine", "ef_search": 1000}


def open_client(path=CHROMA_DIR):
    """The database is a folder of files (data/chroma); no usage statistics are sent."""
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def create_collection(client):
    """Empty collection: a rebuild never mixes old and new chunks."""
    try:
        client.delete_collection(COLLECTION)
    except chromadb.errors.NotFoundError:
        pass  # first build: nothing to delete
    # embedding_function=None: vectors always come from our e5 model, never from Chroma's default one.
    return client.create_collection(COLLECTION, embedding_function=None, configuration={"hnsw": HNSW})


def get_collection(client):
    """Open the collection built by embed.py."""
    return client.get_collection(COLLECTION, embedding_function=None)


def add_chunks(collection, chunks, embeddings, batch_size=1000):
    """Store chunks in batches (Chroma limits the size of one insert)."""
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[chunk["chunk_id"] for chunk in batch],
            embeddings=embeddings[start:start + batch_size],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[{"arxiv_id": chunk["arxiv_id"], "page": chunk["page"]} for chunk in batch],
        )
