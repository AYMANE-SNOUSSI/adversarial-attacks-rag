"""Vector database tests on a tiny collection: no embedding model, no real index needed."""
import numpy as np
import pytest

from search import search
from vector_store import add_chunks, create_collection, get_collection, open_client

CHUNKS = [
    {"chunk_id": "a_p1_c0", "arxiv_id": "a", "page": 1, "text": "about UAPs"},
    {"chunk_id": "a_p2_c0", "arxiv_id": "a", "page": 2, "text": "about deepfakes"},
    {"chunk_id": "b_p5_c0", "arxiv_id": "b", "page": 5, "text": "about defenses"},
]
VECTORS = np.eye(3, dtype=np.float32)  # chunk i points along axis i


class FakeModel:
    """Stands in for the e5 model: always returns the vector it was given."""

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(self, text, normalize_embeddings=True):
        return self.vector / np.linalg.norm(self.vector)


@pytest.fixture
def collection(tmp_path):
    client = open_client(tmp_path)
    collection = create_collection(client)
    add_chunks(collection, CHUNKS, VECTORS)
    return collection


def test_closest_chunk_first_with_its_text_paper_and_page(collection):
    results = search("q", FakeModel([0.1, 0.0, 1.0]), collection, k=2)
    (score, best), (_, second) = results
    assert best == {"chunk_id": "b_p5_c0", "arxiv_id": "b", "page": 5, "text": "about defenses"}
    assert second["chunk_id"] == "a_p1_c0"
    assert score == pytest.approx(1 / np.sqrt(1.01), abs=1e-5)  # cosine similarity, not a distance


def test_k_limits_the_number_of_results(collection):
    assert len(search("q", FakeModel([1.0, 1.0, 1.0]), collection, k=1)) == 1


def test_rebuild_replaces_the_old_chunks(tmp_path):
    client = open_client(tmp_path)
    add_chunks(create_collection(client), CHUNKS, VECTORS)
    add_chunks(create_collection(client), CHUNKS[:1], VECTORS[:1])
    assert get_collection(client).count() == 1


def test_collection_uses_cosine(collection):
    assert collection.configuration["hnsw"]["space"] == "cosine"
