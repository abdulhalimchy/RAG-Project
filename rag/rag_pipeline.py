# =============================================================
# rag/rag_pipeline.py
# -------------------------------------------------------------
# This module implements the full RAG pipeline.
#
# What it does:
#   1. Connects to the local Ollama LLM
#   2. Retrieves relevant chunks from ChromaDB
#   3. Builds a prompt with retrieved context
#   4. Sends prompt to LLM and returns the answer
#
# What is RAG?
#   RAG stands for Retrieval Augmented Generation.
#   Instead of relying on the LLM's training data alone,
#   we first retrieve relevant data from our vector database,
#   then pass it to the LLM as context.
#
#   This means the LLM answers based on YOUR data,
#   not just its general training knowledge.
#
# Flow:
#   User question
#       -> retrieve relevant chunks from ChromaDB
#       -> build prompt with chunks as context
#       -> send to Ollama LLM
#       -> return answer
# =============================================================

import ollama
import json
import os
from config import (
    LLM_MODEL,
    TOP_K_RESULTS,
    RESULTS_SAVE_PATH,
    ANALYSIS_QUERIES
)


# -------------------------------------------------------------
# LOAD LLM
# -------------------------------------------------------------

def load_llm():
    """
    Verifies that Ollama is running and the model is available.

    Why Ollama?
        Ollama allows running LLMs locally with no API costs.
        Everything runs on your own machine or server.

    Raises:
        ConnectionError: if Ollama is not running
        ValueError: if the required model is not pulled
    """

    print(f"Checking Ollama connection...")

    try:
        models = ollama.list()

        # Handle both old and new ollama library response formats
        model_list = models.get('models', models)
        available = []
        for m in model_list:
            if isinstance(m, dict):
                name = m.get('name') or m.get('model', '')
            else:
                name = getattr(m, 'model', '') or getattr(m, 'name', '')
            available.append(name)

        if not any(LLM_MODEL in m for m in available):
            raise ValueError(
                f"\nModel '{LLM_MODEL}' not found."
                f"\nRun: ollama pull {LLM_MODEL}"
            )

        print(f"Ollama is running")
        print(f"Model ready: {LLM_MODEL}")

    except Exception as e:
        if "Connection" in str(e):
            raise ConnectionError(
                f"\nOllama is not running."
                f"\nStart it with: ollama serve"
            )
        raise e


# -------------------------------------------------------------
# BUILD PROMPT
# -------------------------------------------------------------

def build_prompt(question, context_chunks):
    """
    Builds a prompt for the LLM combining the question and
    retrieved context chunks.

    Why prompt engineering matters?
        The way we phrase the prompt directly affects the
        quality of the LLM answer. Clear instructions and
        structured context help the LLM stay focused and
        produce accurate analytical responses.

    Args:
        question       : the user's question
        context_chunks : list of relevant text chunks

    Returns:
        str: formatted prompt ready to send to LLM
    """

    # Combine chunks into one context block
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a precise data analyst. You are analyzing
retail sales data from a superstore covering 2014 to 2017.

Use ONLY the context data provided below to answer the question.
Do not use any outside knowledge. Be direct, specific, and include
exact numbers from the context.

Rules:
- Compare ALL numbers before drawing conclusions
- The highest number wins for best or most questions
- Be concise and factual, maximum 150 words
- Start with the direct conclusion first

CONTEXT DATA:
{context}

QUESTION:
{question}

ANSWER:"""

    return prompt


# -------------------------------------------------------------
# RAG QUERY
# -------------------------------------------------------------

def rag_query(question, collection, embedder, n_results=None):
    """
    Runs the full RAG pipeline for a given question.

    Steps:
        1. Retrieve relevant chunks from ChromaDB
        2. Prioritize summary chunks over transaction chunks
        3. Build prompt with context
        4. Send to Ollama LLM
        5. Return answer

    Why prioritize summary chunks?
        Analytical questions like 'which region is best'
        are better answered by summary chunks that contain
        aggregated numbers rather than individual transactions.

    Args:
        question   : the user's question
        collection : ChromaDB collection
        embedder   : loaded SentenceTransformer model
        n_results  : number of chunks to retrieve

    Returns:
        dict with question, answer, and retrieved chunks
    """

    if n_results is None:
        n_results = TOP_K_RESULTS

    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print(f"{'=' * 60}")

    # Step 1: Retrieve relevant chunks
    from rag.vector_store import retrieve
    chunks = retrieve(question, collection, embedder, n_results=n_results)

    # Step 2: Prioritize summary chunks over transactions
    summaries    = [c for c in chunks if 'total sales' in c.lower()]
    transactions = [c for c in chunks if 'purchased' in c.lower()]
    ordered      = summaries + transactions
    context      = ordered[:8]

    print(f"Retrieved {len(chunks)} chunks ({len(summaries)} summaries, {len(transactions)} transactions)")

    # Step 3: Build prompt
    prompt = build_prompt(question, context)

    # Step 4: Send to LLM
    print(f"Generating answer with {LLM_MODEL}...")
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response['message']['content']

    print(f"\nAnswer:\n{answer}")
    print(f"{'=' * 60}\n")

    return {
        "question": question,
        "answer"  : answer,
        "chunks"  : context
    }


# -------------------------------------------------------------
# RUN ALL QUERIES
# -------------------------------------------------------------

def run_analysis(collection, embedder):
    """
    Runs all 5 required analytical queries through the RAG pipeline
    and saves results to a JSON file.

    The 5 queries cover:
        1. Sales trend analysis
        2. Category performance
        3. Regional performance
        4. Seasonality analysis
        5. Comparative analysis

    Args:
        collection : ChromaDB collection
        embedder   : loaded SentenceTransformer model

    Returns:
        results (list): list of question/answer dicts
    """

    print("\nRunning all analysis queries...")
    print(f"Total queries: {len(ANALYSIS_QUERIES)}")

    results = []

    for i, question in enumerate(ANALYSIS_QUERIES, 1):
        print(f"\nQuery {i} of {len(ANALYSIS_QUERIES)}")
        result = rag_query(question, collection, embedder)
        results.append(result)

    # Save results to file
    save_results(results)

    return results


# -------------------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------------------

def save_results(results):
    """
    Saves all query results to a JSON file.

    Args:
        results (list): list of question/answer dicts
    """

    os.makedirs(os.path.dirname(RESULTS_SAVE_PATH), exist_ok=True)

    # Remove chunks from saved results to keep file clean
    clean_results = [
        {"question": r["question"], "answer": r["answer"]}
        for r in results
    ]

    with open(RESULTS_SAVE_PATH, 'w') as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_SAVE_PATH}")