from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingModel:
    def __init__(self, model_path: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"🔧 Loading embedding model: {model_path}")
        self.model = SentenceTransformer(model_path, device="cpu")
        self.embedding_size = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model ready. Dim: {self.embedding_size}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()