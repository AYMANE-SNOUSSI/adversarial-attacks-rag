"""Download the prebuilt index from the private Hugging Face dataset (used by the hosted app)."""
from huggingface_hub import snapshot_download

from search import DATA

INDEX_REPO = "AymaneGAZI/adversarial-attacks-rag-index"
INDEX_FILES = ["chunks.jsonl", "embeddings.npy", "embeddings_info.json", "metadata.json"]


def download_index():
    """Fetch the index into data/ unless it is already there. The token is read from HF_TOKEN."""
    if all((DATA / name).exists() for name in INDEX_FILES):
        return
    snapshot_download(repo_id=INDEX_REPO, repo_type="dataset", local_dir=DATA, allow_patterns=INDEX_FILES)


if __name__ == "__main__":
    download_index()
    print(f"Index ready in {DATA}")