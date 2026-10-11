"""HTTP API: POST /ask answers a question from the corpus and returns its sources."""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine import Engine


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