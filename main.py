# main.py
import yaml
import os
from rag_system import RAGSystem

def load_config(config_path: str = "config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("config.yaml")

    # Validate required paths
    required_paths = ["pdf_path", "embedding_model_path", "llm_model_path"]
    for path_key in required_paths:
        if not os.path.exists(config[path_key]):
            print(f"Required path does not exist: {config[path_key]} ({path_key})")
            return

    try:
        rag = RAGSystem(
            pdf_path=config["pdf_path"],
            embedding_model_path=config["embedding_model_path"],
            llm_model_path=config["llm_model_path"],
            chunk_size=config.get("chunk_size", 400),
            chunk_overlap=config.get("chunk_overlap", 50),
            max_length=config.get("max_length", 256),
            temperature=config.get("temperature", 0.3),
            vector_db_dir=config.get("vector_db_dir", "./chroma_db"),
            top_k=config.get("top_k", 2)
        )

        print("\nAsk questions about your document. Type 'quit' to exit.\n")
        while True:
            user_input = input("Your question: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if user_input:
                rag.ask_question(user_input)

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()