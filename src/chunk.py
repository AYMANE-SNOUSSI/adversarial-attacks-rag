"""Split each page into overlapping windows that fit the embedding model's input limit."""
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
PAGES_FILE = ROOT / "data" / "pages.jsonl"
CHUNKS_FILE = ROOT / "data" / "chunks.jsonl"

TOKENIZER_NAME = "intfloat/multilingual-e5-small"  # must match the embedding model
MAX_TOKENS = 400      # 512-token limit minus prefix, special tokens and a safety margin
OVERLAP_TOKENS = 80


def split_by_tokens(word_tokens, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS):
    """Return (start, end) word indices of windows holding at most max_tokens tokens."""
    windows, start = [], 0
    while start < len(word_tokens):
        end, total = start, 0
        while end < len(word_tokens) and total + word_tokens[end] <= max_tokens:
            total += word_tokens[end]
            end += 1
        end = max(end, start + 1)  # a single oversized word still forms a window
        windows.append((start, end))
        if end == len(word_tokens):
            break
        back, kept = end, 0  # step back to keep ~overlap tokens of context
        while back > start + 1 and kept + word_tokens[back - 1] <= overlap:
            back -= 1
            kept += word_tokens[back]
        start = back
    return windows


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
chunks = []
with PAGES_FILE.open(encoding="utf-8") as pages:
    for line in pages:
        page = json.loads(line)
        words = page["text"].split()
        word_tokens = [len(ids) for ids in tokenizer(words, add_special_tokens=False)["input_ids"]]
        for index, (start, end) in enumerate(split_by_tokens(word_tokens)):
            chunks.append({
                "chunk_id": f"{page['arxiv_id']}_p{page['page']}_c{index}",
                "arxiv_id": page["arxiv_id"],
                "page": page["page"],
                "text": " ".join(words[start:end]),
            })

with CHUNKS_FILE.open("w", encoding="utf-8") as out:
    for chunk in chunks:
        out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

lengths = [len(chunk["text"].split()) for chunk in chunks]
print(f"{len(chunks)} chunks written to {CHUNKS_FILE.name}")
print(f"Words per chunk: min {min(lengths)}, median {statistics.median(lengths)}, max {max(lengths)}")