"""Split each page of data/pages.jsonl into overlapping word windows (data/chunks.jsonl)."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_FILE = ROOT / "data" / "pages.jsonl"
CHUNKS_FILE = ROOT / "data" / "chunks.jsonl"

CHUNK_WORDS = 250  # ~330 tokens, below the 512-token input limit of common embedding models
OVERLAP_WORDS = 50


def split_words(words, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Overlapping windows; never emits a window fully contained in the previous one."""
    step = size - overlap
    return [words[start:start + size] for start in range(0, max(len(words) - overlap, 1), step)]


chunks = []
with PAGES_FILE.open(encoding="utf-8") as pages:
    for line in pages:
        page = json.loads(line)
        for index, window in enumerate(split_words(page["text"].split())):
            chunks.append({
                "chunk_id": f"{page['arxiv_id']}_p{page['page']}_c{index}",
                "arxiv_id": page["arxiv_id"],
                "page": page["page"],
                "text": " ".join(window),
            })

with CHUNKS_FILE.open("w", encoding="utf-8") as out:
    for chunk in chunks:
        out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

lengths = [len(chunk["text"].split()) for chunk in chunks]
print(f"{len(chunks)} chunks written to {CHUNKS_FILE.name}")
print(f"Words per chunk: min {min(lengths)}, median {statistics.median(lengths)}, max {max(lengths)}")
print(f"\nExample {chunks[1]['chunk_id']}:\n{chunks[1]['text'][:600]}")