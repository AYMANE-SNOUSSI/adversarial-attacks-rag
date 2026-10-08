"""Answer a question from the corpus with Gemini, citing the retrieved passages."""
import json
import os
import re
import sys
import time
import logging
import httpx

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from sentence_transformers import SentenceTransformer

from search import DATA, load_index, search

log = logging.getLogger(__name__)

LLM_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash"]  # tried in order when overloaded
TOP_K = 5
NOT_FOUND = "NOT_FOUND"
CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = f"""You answer questions about scientific papers using ONLY the numbered passages provided.
Rules:
- End every factual sentence with the number of each passage that supports it, e.g. [2] or [1][3].
- Never use knowledge that is not in the passages, even if you know the answer.
- If the passages do not contain the answer, reply with exactly: {NOT_FOUND}
- Answer in the language of the question, unless the question asks for another language.
- Be concise: at most four sentences."""


def build_prompt(question, retrieved, metadata):
    """Number the retrieved passages and put the question after them."""
    blocks = []
    for n, (_, chunk) in enumerate(retrieved, 1):
        title = metadata[chunk["arxiv_id"]]["title"]
        blocks.append(f"[{n}] ({title}, page {chunk['page']})\n{chunk['text']}")
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def generate(client, prompt, llm_models=LLM_MODELS, attempts=3):
    """Try each model in turn: retry overloads (5xx) with backoff, skip a model out of quota (429)."""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    for llm in llm_models:
        for attempt in range(attempts):
            try:
                return llm, client.models.generate_content(model=llm, contents=prompt, config=config)
            except errors.APIError as error:
                if error.code not in (429, 500, 503):
                    raise
                reason = error.code
            except httpx.TransportError as error:  # connection dropped or timed out
                reason = type(error).__name__
            log.warning("%s: %s (attempt %d/%d)", llm, reason, attempt + 1, attempts)
            if reason == 429:
                break  # out of quota: retrying this model cannot help
            if attempt < attempts - 1:
                time.sleep(2 * 2**attempt)
    raise RuntimeError("All Gemini models are unavailable, try again later")


def answer(question, embedder, embeddings, chunks, metadata, client, llm_models=LLM_MODELS):
    """Retrieve, generate, then map each [n] back to its paper and page."""
    retrieved = search(question, embedder, embeddings, chunks, k=TOP_K)
    llm, response = generate(client, build_prompt(question, retrieved, metadata), llm_models)
    text = response.text.strip()
    cited = sorted({int(n) for n in CITATION.findall(text)})
    valid = [n for n in cited if 1 <= n <= len(retrieved)]
    return {
        "question": question,
        "llm": llm,
        "answer": text,
        "found": NOT_FOUND not in text,
        "invalid_citations": [n for n in cited if n not in valid],
        "sources": [
            {
                "n": n,
                "arxiv_id": retrieved[n - 1][1]["arxiv_id"],
                "page": retrieved[n - 1][1]["page"],
                "title": metadata[retrieved[n - 1][1]["arxiv_id"]]["title"],
            }
            for n in valid
        ],
        "retrieved": [(chunk["arxiv_id"], chunk["page"]) for _, chunk in retrieved],
        "tokens": response.usage_metadata.total_token_count,
    }


if __name__ == "__main__":
    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model_name, embeddings, chunks = load_index()
    embedder = SentenceTransformer(model_name)
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))

    result = answer(sys.argv[1], embedder, embeddings, chunks, metadata, client)
    if not result["found"]:
        print("No answer in the corpus: the retrieved passages do not contain it.")
    else:
        print(result["answer"], "\n")
        for source in result["sources"]:
            print(f"[{source['n']}] {source['title']}, page {source['page']} "
                  f"- https://arxiv.org/abs/{source['arxiv_id']}")
    if result["invalid_citations"]:
        print("WARNING: citations to passages that were not provided:", result["invalid_citations"])
    print(f"\n({result['llm']}, {result['tokens']} tokens)")