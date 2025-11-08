from pdf_processor import PDFProcessor
from embedding_model import EmbeddingModel
from vector_database import VectorDatabase
from local_llm import LocalLLM
from typing import List, Dict, Any
import os

class RAGSystem:
    def __init__(
            self,
            pdf_path: str,
            embedding_model_path: str,
            llm_model_path: str,
            chunk_size: int = 400,
            chunk_overlap: int = 50,
            max_length: int = 256,
            temperature: float = 0.3,
            vector_db_dir: str = "./chroma_db",
            top_k: int = 2
    ):
        print("Initializing Edge RAG System...")
        print("=" * 60)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k

        self.pdf_processor = PDFProcessor()
        self.embedding_model = EmbeddingModel(embedding_model_path)
        self.vector_db = VectorDatabase(persist_directory=vector_db_dir)
        self.llm = LocalLLM(llm_model_path, max_length=max_length)

        self._setup_knowledge_base(pdf_path)
        print("Edge RAG ready!")
        print("=" * 60)

    def _setup_knowledge_base(self, pdf_path: str):
        text_chunks = self.pdf_processor.load_pdf(
            pdf_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        if not text_chunks:
            raise ValueError("No text extracted from PDF!")

        embeddings = self.embedding_model.create_embeddings(text_chunks)
        metadatas = [{"chunk_id": i, "source": os.path.basename(pdf_path)} for i in range(len(text_chunks))]
        self.vector_db.add_documents(text_chunks, embeddings, metadatas)

        print(f"Knowledge Base Summary:")
        print(f"   - Chunks: {len(text_chunks)}")
        print(f"   - Embed Dim: {self.embedding_model.embedding_size}")

    def ask_question(self, question: str) -> Dict[str, Any]:  # top_k now from config
        print(f"\n❓ {question}")
        print("-" * 50)

        try:
            q_emb = self.embedding_model.create_embeddings([question])[0]
            similar = self.vector_db.search_similar(q_emb, top_k=self.top_k)

            print(f"📚 Retrieved {len(similar)} context chunks.")
            context = "\n\n".join([doc['document'] for doc in similar])
            prompt = self._create_prompt(question, context)
            answer = self.llm.generate_response(prompt, temperature=self.temperature)

            print(f"💡 Answer:\n{answer}")
            print("-" * 50)
            return {"question": question, "answer": answer, "context_chunks": similar}

        except Exception as e:
            print(f"Error: {e}")
            return {"error": str(e)}

    def _create_prompt(self, question: str, context: str) -> str:
        return f"""Use only the following context to answer the question. If unsure, say "I don't know."

Context:
{context}

Question: {question}

Answer:""".strip()