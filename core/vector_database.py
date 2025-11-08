import chromadb
from typing import List, Dict, Any
import os

class VectorDatabase:
    def __init__(self, persist_directory: str = "./chroma_db"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector DB ready")

    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{}] * len(texts),
            ids=ids
        )
        print(f"Stored {len(texts)} docs")

    def search_similar(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        formatted = []
        for i in range(len(results['documents'][0])):
            formatted.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': float(results['distances'][0][i]),
                'id': results['ids'][0][i]
            })
        return formatted

    def get_collection_info(self) -> Dict[str, Any]:
        return {"total_documents": self.collection.count()}

    def clear_collection(self):
        """Delete all documents in the collection"""
        self.collection.delete(where={})