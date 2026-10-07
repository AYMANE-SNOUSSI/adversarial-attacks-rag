# Design and evaluation

Details behind the [README](../README.md): how the system is built, how it was measured, and where it fails.

## Corpus

44 papers on adversarial attacks and defenses in computer vision, mostly universal adversarial perturbations, transferability and attacks or protections involving diffusion models, plus a few reference papers (ResNet, ImageNet). Each arXiv ID is pinned to a version in [`corpus/papers.txt`](../corpus/papers.txt), so a citation to "page 4" stays valid. The PDFs are downloaded by script and never redistributed.

## Design choices

| Stage | Choice | Reason |
|---|---|---|
| Extraction | PyMuPDF, page by page; ligatures expanded, end-of-line hyphenation rejoined | Keeps the page number of every passage (735 pages) |
| Chunking | ≤400 tokens counted with the embedding model's own tokenizer, 80-token overlap, never across pages | A word-based first version silently truncated 335 of 2,261 chunks (15%) at the model's 512-token limit; the token-based version truncates 0 of 2,465 |
| Embeddings | `intfloat/multilingual-e5-small` (384-d), paper title prepended to each chunk | Multilingual, so French questions work on English papers; the title prefix matched the larger e5-base at a third of the compute |
| Vector store | NumPy matrix, one matrix product per query | 2,465 vectors do not need an approximate index |
| Generation | `gemini-3.5-flash`, low thinking level, rules in the system instruction | The model must cite `[n]` after every claim and reply `NOT_FOUND` when the passages do not contain the answer |
| Citations | Parsed from the answer and mapped to paper, page and arXiv link by the code | The LLM never writes a title or page number; citations to passages that were not provided are flagged |

## Experiment 0: an LLM alone as a source

Gemini (`gemini-3.8-flash`) was asked for 5 highly cited and 5 niche papers on universal adversarial perturbations; every reference was checked on arXiv ([`results/exp00_llm_references.md`](../results/exp00_llm_references.md)).

| Topic | Fully correct references |
|---|---|
| Well-known (universal adversarial perturbations) | 3 / 5 |
| Niche (perturbations in the latent space of the Stable Diffusion VAE) | 1 / 5 |

No paper was invented outright. The errors were details: an arXiv ID one digit off (`2302.04478` for `2302.04578`), a first author borrowed from the next reference in the same list, rewritten titles, and descriptions that contradict the papers' own abstracts.

## Evaluation set

[`eval/questions.jsonl`](../eval/questions.jsonl): 30 questions written from the text of four papers (UAP, Fast-UAP, FGSM, DeepFool) and spot-checked against the PDFs.

- 25 answerable questions, each labelled with the paper and page that contain the answer
- 5 unanswerable trap questions
- 5 questions in French

Retrieval searches all 2,465 chunks, so the other 40 papers act as distractors.

## Retrieval

A hit means a retrieved chunk comes from the labelled page.

| Experiment | Embedding model | recall@1 | recall@3 | recall@5 | recall@10 | MRR | recall@5 (FR, 4 q.) | Encoding time, laptop CPU |
|---|---|---|---|---|---|---|---|---|
| exp01 | e5-small | 0.52 | 0.68 | 0.76 | 0.84 | 0.63 | 0.50 | ~9 min |
| exp02 | e5-base | 0.56 | 0.72 | 0.80 | 0.92 | 0.67 | 0.75 | ~28 min |
| **exp03** | **e5-small + title prefix** | 0.48 | 0.68 | **0.80** | **0.92** | 0.62 | 0.75 | ~14 min |

With 25 questions, one question is 4 points: these differences are within noise. exp03 is kept because it reaches e5-base's recall@5 and recall@10 with a model three times smaller. The title prefix fixed the question it targeted (q001, a page that never names its own paper: not retrieved → rank 5).

## End to end (exp04)

| Metric | Result |
|---|---|
| Unanswerable questions correctly refused | 5 / 5 |
| Answerable questions answered correctly | 19 / 25 |
| Answers citing the labelled page | 18 / 25 |
| Citations to passages that were not provided | 0 |
| Mean tokens per question | ~2,200 |

Correctness is judged by a second model (`gemini-3.1-flash-lite`) against the reference answer; every verdict and its reason are in [`results/exp04_generation_details.jsonl`](../results/exp04_generation_details.jsonl).

Where the six failures come from:

| Labelled page in the 5 passages given to the LLM? | Questions | Outcome |
|---|---|---|
| Yes (20) | 18 | Correct, cited answer |
| | 2 (q017, q020) | Refused |
| No (5) | 4 | Refused |
| | 1 (q013) | Answered, judged correct |

The two refusals with the "right page" retrieved are a measurement artefact, confirmed for q017: page 5 of the FGSM paper is split into three chunks, the answer ("fell to 17.9%") is in chunk `p5_c2`, and only chunks `c0` and `c1` were retrieved. The model was right to refuse. Page-level labels therefore overestimate retrieval quality.

## Limitations

- **Small evaluation set.** 30 questions from 4 of the 44 papers, only 5 in French. Differences of one or two questions are not significant.
- **Page-level relevance labels** overestimate recall (q017). Labelling an exact evidence string per question would measure retrieval at chunk level.
- **Single runs.** The LLM does not give the same answer twice; each configuration was run once.
- **LLM-as-judge.** Correctness is decided by another LLM, spot-checked but not fully reviewed by hand.
- **PDF extraction.** Formulas are flattened (`k̂(x+v) ≠ k̂(x)` becomes `ˆk(x + v)̸ = ˆk(x)`), and one line overlapping an algorithm box was dropped.
- **Chunks never cross pages**, so an idea split by a page break is split in two chunks.
- **Free tier.** 20 requests per day per model bounds the evaluation (exp04 ran over two days, resuming where it stopped) and any public demo. Prompts sent on the free tier may be used by Google: only public papers are sent.

## Reproduce the experiments

```powershell
python src/fetch_corpus.py                                  # check IDs on arXiv, download the pinned PDFs
python src/extract_text.py                                  # data/pages.jsonl
python src/chunk.py                                         # data/chunks.jsonl
python src/embed.py intfloat/multilingual-e5-small --titles # data/embeddings.npy
python src/eval_retrieval.py exp03_e5_small_titles          # retrieval metrics
python src/eval_generation.py exp04_generation              # end-to-end metrics (resumable)
```

## Repository layout

```
corpus/papers.txt         44 pinned arXiv IDs
eval/questions.jsonl      evaluation set
results/                  one file per experiment (exp00 to exp04)
src/
  fetch_corpus.py         arXiv metadata check and PDF download
  extract_text.py         per-page text extraction and cleaning
  inspect_pdf.py          print the extracted text of one page
  chunk.py                token-bounded, page-bounded chunking
  embed.py                chunk embeddings
  search.py               dense retrieval
  answer.py               retrieval + generation + citation mapping
  eval_retrieval.py       recall@k and MRR
  eval_generation.py      refusals, citations and correctness
exploration/              first API calls and the exp00 script
```
