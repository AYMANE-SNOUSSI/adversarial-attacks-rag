"""Question-answering engine: index, embedding model and Gemini client, loaded once."""
import json
import os

from dotenv import load_dotenv


class Engine:
    """Index, embedding model and Gemini client, loaded once."""

    def __init__(self):
        # Heavy imports stay here so modules importing Engine (and their tests) load without them.
        from google import genai
        from sentence_transformers import SentenceTransformer

        from search import DATA, load_index

        load_dotenv()
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        model_name, self.embeddings, self.chunks = load_index()
        self.embedder = SentenceTransformer(model_name)
        self.metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))

    def ask(self, question):
        from answer import answer

        return answer(question, self.embedder, self.embeddings, self.chunks, self.metadata, self.client)