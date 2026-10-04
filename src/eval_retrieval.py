"""Recall@k and MRR of dense retrieval on the answerable questions of eval/questions.jsonl."""
import json
from pathlib import Path
import sys

from sentence_transformers import SentenceTransformer

from search import load_index, search

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "eval" / "questions.jsonl"
RESULTS_FILE = ROOT / "results" / f"{sys.argv[1] if len(sys.argv) > 1 else 'exp01_retrieval_baseline'}.json"
KS = (1, 3, 5, 10)

with QUESTIONS_FILE.open(encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]
answerable = [q for q in questions if q["answerable"]]

model_name, embeddings, chunks = load_index()
model = SentenceTransformer(model_name)

first_ranks = {}
for q in answerable:
    targets = {(ref["arxiv_id"], ref["page"]) for ref in q["relevant"]}
    results = search(q["question"], model, embeddings, chunks, k=max(KS))
    ranks = [rank for rank, (_, chunk) in enumerate(results, 1)
             if (chunk["arxiv_id"], chunk["page"]) in targets]
    first_ranks[q["id"]] = ranks[0] if ranks else None  # None: not in the top max(KS)
    print(f"{q['id']} [{q['lang']}] first relevant rank: {first_ranks[q['id']] or 'not found'}")


def recall_at(k, subset):
    """Share of questions whose first relevant chunk is ranked <= k."""
    return sum(first_ranks[q["id"]] is not None and first_ranks[q["id"]] <= k for q in subset) / len(subset)


mrr = sum(1 / r for r in first_ranks.values() if r) / len(answerable)
report = {
    "model": model_name,
    "n_questions": len(answerable),
    "recall": {f"@{k}": round(recall_at(k, answerable), 3) for k in KS},
    "mrr": round(mrr, 3),
    "recall@5_by_lang": {
        lang: round(recall_at(5, [q for q in answerable if q["lang"] == lang]), 3)
        for lang in sorted({q["lang"] for q in answerable})
    },
}
RESULTS_FILE = ROOT / "results" / f"{sys.argv[1] if len(sys.argv) > 1 else 'exp01_retrieval_baseline'}.json"
print(json.dumps(report, indent=2))