# Stage 1: dependencies and embedding model, shared by the two next stages.
FROM python:3.12-slim AS base
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time so containers start fast and offline.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# Stage 2: build the index from arXiv, so the paper text is never stored in the repository.
FROM base AS index
COPY corpus/ corpus/
COPY src/ src/
RUN python src/fetch_corpus.py \
 && python src/extract_text.py \
 && python src/chunk.py \
 && python src/embed.py --titles

# Stage 3: runtime image, with the index but without the PDFs or the extracted pages.
FROM base
# Hugging Face Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
COPY src/ src/
COPY start.sh .
COPY --from=index /app/data/chunks.jsonl /app/data/embeddings.npy \
     /app/data/embeddings_info.json /app/data/metadata.json data/

ENV HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user
USER user

# Default: the whole app (API + web interface on port 7860). compose.yaml runs each part separately.
EXPOSE 7860
CMD ["sh", "start.sh"]