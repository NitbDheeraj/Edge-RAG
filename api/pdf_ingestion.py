# api/pdf_ingestion.py
import os
import shutil
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pathlib import Path

# Import your core modules
from core.pdf_processor import PDFProcessor
from core.embedding_model import EmbeddingModel
from core.vector_database import VectorDatabase


# Load shared config
def load_config():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.yaml not found")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()

# Initialize components (shared across requests)
pdf_processor = PDFProcessor()
embedding_model = EmbeddingModel(config["embedding_model_path"])
vector_db = VectorDatabase(persist_directory=config["vector_db_dir"])

app = FastAPI(title="Edge RAG PDF Ingestion API", version="1.0")


@app.post("/upload-pdf", summary="Upload and process a new PDF")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Save uploaded file temporarily
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Process PDF
        chunks = pdf_processor.load_pdf(
            str(file_path),
            chunk_size=config.get("chunk_size", 400),
            chunk_overlap=config.get("chunk_overlap", 50)
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")

        # Generate embeddings
        embeddings = embedding_model.create_embeddings(chunks)

        # Clear existing DB and add new docs
        vector_db.collection.delete(where={})  # Delete all entries
        metadatas = [{"chunk_id": i, "source": file.filename} for i in range(len(chunks))]
        vector_db.add_documents(chunks, embeddings, metadatas)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "PDF processed successfully",
                "chunks_created": len(chunks),
                "filename": file.filename
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        # Clean up uploaded file
        if file_path.exists():
            os.remove(file_path)


@app.delete("/delete-knowledge", summary="Delete entire knowledge base")
async def delete_knowledge():
    try:
        vector_db.collection.delete(where={})  # Clear all
        # Optional: also clear persist directory (hard reset)
        # shutil.rmtree(config["vector_db_dir"], ignore_errors=True)
        return {"message": "Knowledge base cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.get("/status", summary="Get current knowledge base status")
async def get_status():
    try:
        count = vector_db.collection.count()
        return {
            "documents_in_db": count,
            "vector_db_path": config["vector_db_dir"],
            "embedding_model": config["embedding_model_path"]
        }
    except Exception as e:
        return {"error": str(e)}