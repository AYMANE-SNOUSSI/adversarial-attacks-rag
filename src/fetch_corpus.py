"""Fetch arXiv metadata for corpus/papers.txt and download the pinned PDF versions."""
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PAPERS_FILE = ROOT / "corpus" / "papers.txt"
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
METADATA_FILE = DATA_DIR / "metadata.json"

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{}"
ATOM = "{http://www.w3.org/2005/Atom}"
HEADERS = {"User-Agent": "adversarial-attacks-rag/0.1 (https://github.com/AYMANE-SNOUSSI)"}
DELAY_S = 3  # arXiv asks for at most one request every 3 seconds
RETRY_STATUS = {406, 429, 500, 502, 503, 504}


def read_ids(path):
    """Return arXiv IDs from papers.txt, skipping blank and comment-only lines."""
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        arxiv_id = line.split("#")[0].strip()
        if arxiv_id:
            ids.append(arxiv_id)
    return ids


def get(url, params=None, attempts=5):
    """GET with exponential backoff: arXiv intermittently answers 406/429/5xx."""
    for attempt in range(1, attempts + 1):
        response = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if response.status_code == 200:
            return response
        if response.status_code not in RETRY_STATUS or attempt == attempts:
            response.raise_for_status()
        wait = DELAY_S * 2**attempt
        print(f"  HTTP {response.status_code}, retry in {wait}s")
        time.sleep(wait)


def fetch_metadata(ids):
    """Return {arxiv_id: {title, authors}} for every ID arXiv recognises."""
    response = get(API_URL, params={"id_list": ",".join(ids), "max_results": len(ids)})
    root = ET.fromstring(response.content)
    metadata = {}
    for entry in root.findall(f"{ATOM}entry"):
        arxiv_id = entry.find(f"{ATOM}id").text.split("/abs/")[-1]
        metadata[arxiv_id] = {
            "title": " ".join(entry.find(f"{ATOM}title").text.split()),
            "authors": [name.text for name in entry.findall(f"{ATOM}author/{ATOM}name")],
        }
    return metadata


def download_pdfs(ids):
    """Save each missing PDF as data/pdfs/<arxiv_id>.pdf."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for arxiv_id in ids:
        target = PDF_DIR / f"{arxiv_id}.pdf"
        if target.exists():
            continue
        time.sleep(DELAY_S)
        content = get(PDF_URL.format(arxiv_id)).content
        if not content.startswith(b"%PDF"):
            raise ValueError(f"{arxiv_id}: response is not a PDF")
        target.write_bytes(content)
        print(f"  downloaded {target.name}")


ids = read_ids(PAPERS_FILE)
metadata = fetch_metadata(ids)

missing = [arxiv_id for arxiv_id in ids if arxiv_id not in metadata]
for arxiv_id in ids:
    if arxiv_id in metadata:
        paper = metadata[arxiv_id]
        print(f"OK      {arxiv_id:14} {paper['authors'][0]} | {paper['title']}")
    else:
        print(f"ABSENT  {arxiv_id}")
print(f"\n{len(ids) - len(missing)} found, {len(missing)} absent")

if missing:
    raise SystemExit("Fix papers.txt before downloading.")

METADATA_FILE.parent.mkdir(exist_ok=True)
METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
download_pdfs(ids)
print(f"\n{len(list(PDF_DIR.glob('*.pdf')))} PDFs in {PDF_DIR}")