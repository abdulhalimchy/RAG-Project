# =============================================================
# main.py
# -------------------------------------------------------------
# Entry point for the RAG Sales Analysis project.
#
# How to run:
#   python main.py            -> reuses saved chunks if available
#   python main.py --rebuild  -> forces rebuild of all chunks
#
# Requirements:
#   - Ollama running with llama3.2:3b pulled
#   - superstore.csv in data/ folder
#   - Virtual environment activated
# =============================================================

import sys
import os
from rag.data_preparation import load_data, create_chunks, save_chunks, load_chunks
from rag.vector_store import load_embedding_model, init_chromadb, store_chunks
from rag.rag_pipeline import load_llm, rag_query
from config import CHUNKS_SAVE_PATH


def main():
    rebuild = '--rebuild' in sys.argv

    print("=" * 60)
    print("  RAG SALES ANALYSIS")
    print("=" * 60)

    # STEP 1: Data Preparation
    if os.path.exists(CHUNKS_SAVE_PATH) and not rebuild:
        print("\nLoading saved chunks...")
        chunks = load_chunks()
    else:
        print("\nBuilding chunks from dataset...")
        df = load_data()
        chunks = create_chunks(df)
        save_chunks(chunks)

    # STEP 2: Vector Database
    print("Setting up vector database...")
    embedder = load_embedding_model()
    collection = init_chromadb()
    store_chunks(collection, chunks, embedder)

    # STEP 3: LLM
    print("Loading LLM...")
    load_llm()

    # STEP 4: Interactive Mode
    print("\n" + "=" * 60)
    print("  Ready! Ask anything about the sales data.")
    print("  Type 'exit' to quit.")
    print("=" * 60)

    while True:
        print()
        user_input = input("Your question: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        rag_query(user_input, collection, embedder)


if __name__ == "__main__":
    main()