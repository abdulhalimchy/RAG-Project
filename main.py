# =============================================================
# main.py
# -------------------------------------------------------------
# Entry point for the RAG Sales Analysis project.
#
# What it does:
#   1. Loads and prepares the dataset
#   2. Sets up the vector database
#   3. Runs the required 5 analytical queries
#   4. Evaluates accuracy
#   5. Starts interactive query mode
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

import time
import sys
import os
from rag.data_preparation import load_data, create_chunks, save_chunks, load_chunks
from rag.vector_store import load_embedding_model, init_chromadb, store_chunks
from rag.rag_pipeline import load_llm, run_analysis, rag_query
from rag.evaluator import evaluate
from config import CHUNKS_SAVE_PATH


# -------------------------------------------------------------
# INTERACTIVE MODE
# -------------------------------------------------------------

def interactive_mode(collection, embedder):
    """
    Starts an interactive query session where the user can
    ask any number of questions about the sales data.

    The session continues until the user types 'exit' or 'quit'.

    Args:
        collection : ChromaDB collection
        embedder   : loaded SentenceTransformer model
    """

    print("\n" + "=" * 60)
    print("INTERACTIVE QUERY MODE")
    print("-" * 60)
    print("You can now ask any question about the sales data.")
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'help' to see example questions.")
    print("=" * 60)

    # Example questions shown when user types 'help'
    example_questions = [
        "Which region has the best sales performance?",
        "What is the sales trend over the 4-year period?",
        "Which product category generates the most revenue?",
        "Which months show the highest sales?",
        "Compare Technology vs Furniture sales.",
        "Which sub-category has the highest profit margin?",
        "How does the West region compare to the East in profit?",
        "Which customer segment generates the most revenue?",
        "What is the impact of discounts on profit?",
        "Which shipping mode is most commonly used?",
        "Which state has the highest sales?",
        "Which city is the top performer?",
    ]

    session_queries = []

    while True:
        print()
        user_input = input("Your question: ").strip()

        # Exit condition
        if user_input.lower() in ['exit', 'quit']:
            print("\nExiting interactive mode.")
            break

        # Help command
        if user_input.lower() == 'help':
            print("\nExample questions:")
            for i, q in enumerate(example_questions, 1):
                print(f"  {i}. {q}")
            continue

        # Skip empty input
        if not user_input:
            print("Please enter a question.")
            continue

        # Run RAG query
        result = rag_query(user_input, collection, embedder)
        session_queries.append(result)

    # Summary of session
    if session_queries:
        print(f"\nSession complete. Total questions asked: {len(session_queries)}")

    return session_queries


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    """
    Main function that runs the full RAG pipeline.

    Steps:
        1. Load and prepare data
        2. Setup vector database
        3. Run required 5 analytical queries
        4. Evaluate accuracy
        5. Start interactive query mode
    """

    start_time = time.time()

    # Check if --rebuild flag is passed
    rebuild = '--rebuild' in sys.argv

    print("=" * 60)
    print("RAG SALES ANALYSIS PROJECT")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: DATA PREPARATION
    # ---------------------------------------------------------
    print("\nSTEP 1: Data Preparation")
    print("-" * 60)

    if os.path.exists(CHUNKS_SAVE_PATH) and not rebuild:
        print("Saved chunks found. Loading from file...")
        print("To rebuild chunks run: python main.py --rebuild")
        chunks = load_chunks()
        df     = load_data()
    else:
        if rebuild:
            print("Rebuild flag detected. Rebuilding all chunks...")
        else:
            print("No saved chunks found. Creating from scratch...")
        df     = load_data()
        chunks = create_chunks(df)
        save_chunks(chunks)

    # ---------------------------------------------------------
    # STEP 2: VECTOR DATABASE SETUP
    # ---------------------------------------------------------
    print("\nSTEP 2: Vector Database Setup")
    print("-" * 60)

    embedder   = load_embedding_model()
    collection = init_chromadb()
    store_chunks(collection, chunks, embedder)

    # ---------------------------------------------------------
    # STEP 3: LLM SETUP
    # ---------------------------------------------------------
    print("\nSTEP 3: LLM Setup")
    print("-" * 60)

    load_llm()

    # ---------------------------------------------------------
    # STEP 4: RUN REQUIRED ANALYSIS QUERIES
    # ---------------------------------------------------------
    print("\nSTEP 4: Running Required Analysis Queries")
    print("-" * 60)

    rag_results = run_analysis(collection, embedder)

    # ---------------------------------------------------------
    # STEP 5: EVALUATION
    # ---------------------------------------------------------
    print("\nSTEP 5: Evaluation")
    print("-" * 60)

    evaluate(df, collection, embedder, rag_results)

    # ---------------------------------------------------------
    # STEP 6: INTERACTIVE MODE
    # ---------------------------------------------------------
    print("\nSTEP 6: Interactive Query Mode")
    print("-" * 60)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"Pipeline completed in {minutes}m {seconds}s")
    print(f"Results saved to: results/rag_results.json")

    interactive_mode(collection, embedder)


if __name__ == "__main__":
    main()