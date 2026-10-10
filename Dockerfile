FROM python:3.12-slim

WORKDIR /app

# The extra index provides CPU-only PyTorch wheels; the default Linux wheel bundles
# several GB of CUDA libraries that this app never uses.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Download the embedding model at build time so containers start fast and offline.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY src/ src/
COPY data/chunks.jsonl data/embeddings.npy data/embeddings_info.json data/metadata.json data/

ENV HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]