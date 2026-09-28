"""Print the raw text PyMuPDF extracts from one page of a corpus PDF."""
import sys
from pathlib import Path

import pymupdf

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"

arxiv_id, page_number = sys.argv[1], int(sys.argv[2])
with pymupdf.open(PDF_DIR / f"{arxiv_id}.pdf") as doc:
    print(f"{arxiv_id}: {doc.page_count} pages")
    print("-" * 70)
    print(doc[page_number - 1].get_text())