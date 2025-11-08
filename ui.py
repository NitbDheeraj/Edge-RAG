# ui.py

import os
import yaml
import shutil
from pathlib import Path
import gradio as gr

# Import your core modules
from core.pdf_processor import PDFProcessor
from core.embedding_model import EmbeddingModel
from core.vector_database import VectorDatabase
from core.local_llm import LocalLLM

# ----------------------------
# Load config
# ----------------------------
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("config.yaml not found!")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ----------------------------
# Initialize shared components
# ----------------------------
pdf_processor = PDFProcessor()
embedding_model = EmbeddingModel(config["embedding_model_path"])
vector_db = VectorDatabase(persist_directory=config["vector_db_dir"])
llm = LocalLLM(
    model_path=config["llm_model_path"],
    max_length=config.get("max_length", 256)
)


# ----------------------------
# Helper Functions
# ----------------------------
def upload_pdf(file):
    if not file or not file.name.endswith(".pdf"):
        return "Please upload a valid PDF file.", "", []

    try:
        # Process PDF
        chunks = pdf_processor.load_pdf(
            file.name,
            chunk_size=config.get("chunk_size", 400),
            chunk_overlap=config.get("chunk_overlap", 50)
        )
        if not chunks:
            return "No text could be extracted from the PDF.", "", []

        # Generate embeddings
        embeddings = embedding_model.create_embeddings(chunks)

        # Clear old DB and add new
        vector_db.collection.delete(where={})
        metadatas = [{"chunk_id": i, "source": os.path.basename(file.name)} for i in range(len(chunks))]
        vector_db.add_documents(chunks, embeddings, metadatas)

        status = f"PDF '{os.path.basename(file.name)}' processed successfully!\nChunks: {len(chunks)}"
        return status, "", []

    except Exception as e:
        return f"Error: {str(e)}", "", []


def clear_knowledge():
    try:
        vector_db.collection.delete(where={})
        # Optional: delete persist directory for full reset
        # if os.path.exists(config["vector_db_dir"]):
        #     shutil.rmtree(config["vector_db_dir"])
        return "Knowledge base cleared!", "", []
    except Exception as e:
        return f"Error clearing knowledge: {str(e)}", "", []


def ask_question(question, history):
    if not question.strip():
        return history, "Please ask a question."

    try:
        # Check if any documents exist
        if vector_db.collection.count() == 0:
            return history, "Knowledge base is empty. Please upload a PDF first."

        # Embed query
        q_emb = embedding_model.create_embeddings([question])[0]

        # Retrieve context
        top_k = config.get("top_k", 2)
        similar = vector_db.search_similar(q_emb, top_k=top_k)

        # Build context
        context = "\n\n".join([doc['document'] for doc in similar])

        # Generate prompt
        prompt = f"""Use only the following context to answer the question. If unsure, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

        # Generate answer
        temperature = config.get("temperature", 0.3)
        answer = llm.generate_response(prompt, temperature=temperature)

        # Format retrieved context for display
        context_display = "\n\n".join([
            f"Chunk {i + 1} (Dist: {doc['distance']:.3f}):\n{doc['document'][:300]}..."
            for i, doc in enumerate(similar)
        ])

        # Update chat history
        history.append((question, answer))
        return history, f"🔍 Retrieved Context:\n{context_display}"

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append((question, error_msg))
        return history, error_msg


# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks(title="Edge RAG - Local Document Q&A") as demo:
    gr.Markdown("## Edge RAG: Ask Questions About Your PDFs (Offline & Private)")

    with gr.Tab(" Document Management"):
        with gr.Row():
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        with gr.Row():
            upload_btn = gr.Button("Process PDF")
            clear_btn = gr.Button("🗑️ Clear Knowledge Base")
        status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("💬 Chat"):
        chatbot = gr.Chatbot(height=400, label="Conversation")
        question_input = gr.Textbox(label="Ask a question", placeholder="e.g., What is backpropagation?")
        context_output = gr.Textbox(label="Retrieved Context (for transparency)", lines=5, max_lines=10,
                                    interactive=False)
        ask_btn = gr.Button("💬 Ask")

    # Event bindings
    upload_btn.click(upload_pdf, inputs=pdf_input, outputs=[status_output, context_output, chatbot])
    clear_btn.click(clear_knowledge, outputs=[status_output, context_output, chatbot])
    ask_btn.click(ask_question, inputs=[question_input, chatbot], outputs=[chatbot, context_output])
    question_input.submit(ask_question, inputs=[question_input, chatbot], outputs=[chatbot, context_output])

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public link (not recommended for private docs)
        inbrowser=True  # Opens browser automatically
    )