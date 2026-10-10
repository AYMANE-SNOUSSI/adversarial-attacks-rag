"""HTTP API: POST /ask answers a question from the corpus and returns its sources."""
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    n: int
    arxiv_id: str
    page: int
    title: str
    url: str


class AskResponse(BaseModel):
    found: bool
    answer: str | None
    sources: list[Source]
    llm: str


class Engine:
    """Index, embedding model and Gemini client, loaded once."""

    def __init__(self):
        # Heavy imports stay here so the API module (and its tests) load without them.
        from google import genai
        from sentence_transformers import SentenceTransformer

        from search import DATA, load_index

        load_dotenv()
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        model_name, self.embeddings, self.chunks = load_index()
        self.embedder = SentenceTransformer(model_name)
        self.metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))

    def ask(self, question):
        from answer import answer

        return answer(question, self.embedder, self.embeddings, self.chunks, self.metadata, self.client)


_engine = None


def get_engine():
    """Create the engine on first use and reuse it afterwards."""
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


@asynccontextmanager
async def lifespan(app):
    get_engine()  # load at startup so the first user does not wait
    yield


app = FastAPI(title="adversarial-attacks-rag", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, engine=Depends(get_engine)):
    try:
        result = engine.ask(request.question)
    except RuntimeError:  # every Gemini model overloaded or out of quota
        raise HTTPException(status_code=503, detail="The language model is unavailable, try again later.")
    return AskResponse(
        found=result["found"],
        answer=result["answer"] if result["found"] else None,
        sources=[
            Source(**source, url=f"https://arxiv.org/abs/{source['arxiv_id']}")
            for source in result["sources"]
        ],
        llm=result["llm"],
    )
