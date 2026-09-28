"""Extract cleaned per-page text from the corpus PDFs into data/pages.jsonl."""
import json
import re
import unicodedata
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
OUTPUT_FILE = ROOT / "data" / "pages.jsonl"

HYPHEN_BREAK = re.compile(r"(\w)-\n([a-z])")  # "pertur-\nbation" -> "perturbation"
WHITESPACE = re.compile(r"\s+")


def clean(text):
    """Expand ligatures, rejoin words split across lines, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)  # "ﬁ" -> "fi"
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    return WHITESPACE.sub(" ", text).strip()


n_papers = n_pages = n_chars = 0
empty_pages = []
with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        n_papers += 1
        with pymupdf.open(pdf_path) as doc:
            for page_index, page in enumerate(doc):
                text = clean(page.get_text())
                if not text:
                    empty_pages.append(f"{pdf_path.stem} p.{page_index + 1}")
                    continue
                record = {"arxiv_id": pdf_path.stem, "page": page_index + 1, "text": text}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_pages += 1
                n_chars += len(text)

print(f"{n_papers} papers, {n_pages} pages, {n_chars:,} characters (~{n_chars // 4:,} tokens)")
print(f"Empty pages ({len(empty_pages)}): {empty_pages}")