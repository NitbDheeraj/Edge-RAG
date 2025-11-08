from PyPDF2 import PdfReader
from typing import List, Dict, Any
import os


class PDFProcessor:
    def __init__(self):
        self.text_chunks = []

    def load_pdf(self, pdf_path: str, chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
        print(f"Loading PDF: {pdf_path}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    full_text += f"Page {page_num + 1}: {page_text}\n"

            print(f"Extracted text from {len(reader.pages)} pages")
            self.text_chunks = self._split_text(full_text, chunk_size, chunk_overlap)
            print(f"Created {len(self.text_chunks)} chunks")
            return self.text_chunks
        except Exception as e:
            print(f"PDF load error: {e}")
            return []

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        step = chunk_size - chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def get_chunk_info(self) -> Dict[str, Any]:
        if not self.text_chunks:
            return {"total_chunks": 0, "sample_chunk": ""}
        return {
            "total_chunks": len(self.text_chunks),
            "sample_chunk": self.text_chunks[0][:200] + "..."
        }