"""End-to-end evaluation: answer every question, then score refusals, citations and correctness.

Resumable: questions already in the details file are skipped, so a run stopped by the
free-tier daily quota continues the next day with the same command.
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from answer import answer
from search import DATA, load_index

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "eval" / "questions.jsonl"
NAME = sys.argv[1] if len(sys.argv) > 1 else "exp04_generation"
SUMMARY_FILE = ROOT / "results" / f"{NAME}.json"
DETAILS_FILE = ROOT / "results" / f"{NAME}_details.jsonl"

ANSWER_MODEL = "gemini-3.5-flash"      # one fixed model, so all answers are comparable
JUDGE_MODEL = "gemini-3.1-flash-lite"  # separate model, with its own daily quota
PAUSE_S = 6                            # stay under the per-minute limit
JUDGE_PROMPT = """Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}
Does the candidate state the facts of the reference answer without contradicting them?
Extra correct details are acceptable."""


class Verdict(BaseModel):
    correct: bool
    reason: str


JUDGE_CONFIG = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=Verdict,
    thinking_config=types.ThinkingConfig(thinking_level="low"),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def judge(client, question, reference, candidate, attempts=3):
    """LLM-as-judge; retries server errors only (a 429 here is the daily quota)."""
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, candidate=candidate)
    for attempt in range(attempts):
        try:
            return client.models.generate_content(
                model=JUDGE_MODEL, contents=prompt, config=JUDGE_CONFIG
            ).parsed
        except errors.ServerError:
            if attempt == attempts - 1:
                raise
            time.sleep(PAUSE_S * 2**attempt)


def evaluate(q, result):
    """Score one answer against the question's references."""
    targets = {(ref["arxiv_id"], ref["page"]) for ref in q["relevant"]}
    record = {
        "id": q["id"],
        "lang": q["lang"],
        "answerable": q["answerable"],
        "found": result["found"],
        "relevant_retrieved": any(tuple(page) in targets for page in result["retrieved"]),
        "cites_relevant": any((s["arxiv_id"], s["page"]) in targets for s in result["sources"]),
        "invalid_citations": result["invalid_citations"],
        "llm": result["llm"],
        "tokens": result["tokens"],
        "question": q["question"],
        "reference": q["answer"],
        "answer": result["answer"],
    }
    if not q["answerable"]:
        record["correct"] = not result["found"]
    elif not result["found"]:
        record["correct"] = False
    else:
        verdict = judge(client, q["question"], q["answer"], result["answer"])
        record["correct"], record["judge_reason"] = verdict.correct, verdict.reason
    return record


def share(records, test):
    """Fraction of records satisfying test, rounded."""
    return round(sum(map(test, records)) / len(records), 3) if records else None


load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
model_name, embeddings, chunks = load_index()
embedder = SentenceTransformer(model_name)
metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
with QUESTIONS_FILE.open(encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

records = []
if DETAILS_FILE.exists():
    with DETAILS_FILE.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
done = {r["id"] for r in records}
print(f"{len(done)} questions already evaluated, {len(questions) - len(done)} to go")

with DETAILS_FILE.open("a", encoding="utf-8") as out:
    for q in questions:
        if q["id"] in done:
            continue
        try:
            result = answer(q["question"], embedder, embeddings, chunks, metadata, client,
                            llm_models=[ANSWER_MODEL])
            time.sleep(PAUSE_S)
            record = evaluate(q, result)
            time.sleep(PAUSE_S)
        except (RuntimeError, errors.APIError) as error:
            print(f"\nStopped at {q['id']}: {str(error)[:150]}")
            print("Finished questions are saved: re-run the same command once the quota resets.")
            break
        records.append(record)
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{q['id']} [{q['lang']}] found={record['found']} "
              f"cites_relevant={record['cites_relevant']} correct={record['correct']}")

answerable = [r for r in records if r["answerable"]]
unanswerable = [r for r in records if not r["answerable"]]
summary = {
    "complete": len(records) == len(questions),
    "n_evaluated": len(records),
    "retrieval": model_name,
    "answer_model": ANSWER_MODEL,
    "judge_model": JUDGE_MODEL,
    "answerable": {
        "n": len(answerable),
        "answered": share(answerable, lambda r: r["found"]),
        "correct": share(answerable, lambda r: r["correct"]),
        "cites_relevant_page": share(answerable, lambda r: r["cites_relevant"]),
        "answered_without_relevant_page": [
            r["id"] for r in answerable if r["found"] and not r["relevant_retrieved"]
        ],
    },
    "unanswerable": {
        "n": len(unanswerable),
        "correct_refusals": share(unanswerable, lambda r: not r["found"]),
    },
    "invalid_citations": sum(len(r["invalid_citations"]) for r in records),
    "mean_tokens": round(sum(r["tokens"] for r in records) / len(records)) if records else None,
    "llm_calls": dict(Counter(r["llm"] for r in records)),
}
SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))