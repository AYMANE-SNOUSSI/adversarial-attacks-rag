"""API tests with a fake engine: no model, no index, no Gemini key needed."""
import pytest
from fastapi.testclient import TestClient

from api import app, get_engine

FOUND = {
    "found": True,
    "answer": "The perturbation is computed on 10,000 training images [1].",
    "sources": [
        {"n": 1, "arxiv_id": "1610.08401v3", "page": 4, "title": "Universal adversarial perturbations"}
    ],
    "llm": "fake-llm",
}
NOT_FOUND = {"found": False, "answer": "NOT_FOUND", "sources": [], "llm": "fake-llm"}


class FakeEngine:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def ask(self, question):
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client_with():
    """Return a test client whose /ask uses the given fake engine."""

    def make(engine):
        app.dependency_overrides[get_engine] = lambda: engine
        return TestClient(app)  # not used as a context manager: the real engine is never loaded

    yield make
    app.dependency_overrides.clear()


def test_health(client_with):
    response = client_with(FakeEngine()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_answer_with_sources(client_with):
    response = client_with(FakeEngine(result=FOUND)).post("/ask", json={"question": "How many images?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["answer"] == FOUND["answer"]
    assert body["sources"][0]["page"] == 4
    assert body["sources"][0]["url"] == "https://arxiv.org/abs/1610.08401v3"


def test_not_found_has_no_answer(client_with):
    response = client_with(FakeEngine(result=NOT_FOUND)).post("/ask", json={"question": "Unknown topic?"})
    assert response.status_code == 200
    assert response.json() == {"found": False, "answer": None, "sources": [], "llm": "fake-llm"}


@pytest.mark.parametrize("question", ["", "ab", "x" * 501])
def test_invalid_question_rejected(client_with, question):
    response = client_with(FakeEngine(result=FOUND)).post("/ask", json={"question": question})
    assert response.status_code == 422


def test_llm_unavailable_returns_503(client_with):
    engine = FakeEngine(error=RuntimeError("All Gemini models are unavailable"))
    response = client_with(engine).post("/ask", json={"question": "How many images?"})
    assert response.status_code == 503
