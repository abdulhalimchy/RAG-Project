# =============================================================
# main.py
# -------------------------------------------------------------
# Entry point for the RAG Sales Analysis project.
#
# What it does:
#   1. Loads and prepares the dataset
#   2. Sets up the vector database
#   3. Runs all 5 analytical queries
#   4. Evaluates accuracy
#   5. Saves results
#
# How to run:
#   python main.py
#
# Requirements:
#   - Ollama running with llama3.2:3b pulled
#   - superstore.csv in data/ folder
#   - Virtual environment activated
# =============================================================

import time
from rag.data_preparation import load_data, create_chunks, save_chunks
from rag.vector_store import load_embedding_model, init_chromadb, store_chunks
from rag.rag_pipeline import load_llm, run_analysis
from rag.evaluator import evaluate


def main():
    """
    Main function that runs the full RAG pipeline.

    Steps:
        1. Load and prepare data
        2. Setup vector database
        3. Run analysis queries
        4. Evaluate accuracy
    """

    start_time = time.time()

    print("=" * 60)
    print("RAG SALES ANALYSIS PROJECT")
    print("Data Warehouse Course - University of Helsinki")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: DATA PREPARATION
    # ---------------------------------------------------------
    print("\nSTEP 1: Data Preparation")
    print("-" * 60)

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
    # STEP 4: RUN ANALYSIS QUERIES
    # ---------------------------------------------------------
    print("\nSTEP 4: Running Analysis Queries")
    print("-" * 60)

    rag_results = run_analysis(collection, embedder)

    # ---------------------------------------------------------
    # STEP 5: EVALUATION
    # ---------------------------------------------------------
    print("\nSTEP 5: Evaluation")
    print("-" * 60)

    evaluate(df, collection, embedder, rag_results)

    # ---------------------------------------------------------
    # DONE
    # ---------------------------------------------------------
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE")
    print(f"Total time : {minutes}m {seconds}s")
    print(f"Results saved to: results/rag_results.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()