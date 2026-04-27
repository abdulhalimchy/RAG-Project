import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test for rag_pipeline.py

from rag.data_preparation import load_data, create_chunks
from rag.vector_store import load_embedding_model, init_chromadb, store_chunks
from rag.rag_pipeline import load_llm, build_prompt, rag_query

# Test 1: Check Ollama connection
print("=" * 50)
print("TEST 1: Ollama Connection")
print("=" * 50)
load_llm()
print("PASSED\n")

# Test 2: Build prompt
print("=" * 50)
print("TEST 2: Build Prompt")
print("=" * 50)
chunks = ["West region total sales $725,457", "East region total sales $678,781"]
prompt = build_prompt("Which region has best sales?", chunks)
assert "West region" in prompt, "Context missing from prompt"
assert "Which region" in prompt, "Question missing from prompt"
print("PASSED\n")

# Test 3: RAG query
print("=" * 50)
print("TEST 3: RAG Query")
print("=" * 50)
df       = load_data()
chunks   = create_chunks(df)
embedder = load_embedding_model()
collection = init_chromadb()
store_chunks(collection, chunks, embedder)

result = rag_query("Which region has best sales?", collection, embedder)
assert "question" in result, "question key missing"
assert "answer" in result, "answer key missing"
assert len(result["answer"]) > 0, "Empty answer"
print("PASSED\n")

print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)