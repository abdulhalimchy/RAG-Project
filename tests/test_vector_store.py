import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test for vector_store.py

from rag.data_preparation import load_data, create_chunks
from rag.vector_store import load_embedding_model, init_chromadb, store_chunks, retrieve

# Test 1: Load embedding model
print("=" * 50)
print("TEST 1: Load Embedding Model")
print("=" * 50)
embedder = load_embedding_model()
assert embedder is not None, "Embedder is None"
assert embedder.get_sentence_embedding_dimension() == 384, "Wrong dimensions"
print("PASSED\n")

# Test 2: Initialize ChromaDB
print("=" * 50)
print("TEST 2: Initialize ChromaDB")
print("=" * 50)
collection = init_chromadb()
assert collection is not None, "Collection is None"
print("PASSED\n")

# Test 3: Store chunks
print("=" * 50)
print("TEST 3: Store Chunks")
print("=" * 50)
df = load_data()
chunks = create_chunks(df)
store_chunks(collection, chunks, embedder)
assert collection.count() == len(chunks), "Chunk count mismatch"
print("PASSED\n")

# Test 4: Retrieve chunks
print("=" * 50)
print("TEST 4: Retrieve Chunks")
print("=" * 50)
results = retrieve("which region has best sales", collection, embedder)
assert len(results) > 0, "No results returned"
assert isinstance(results[0], str), "Result is not a string"
print(f"Sample result: {results[0][:100]}...")
print("PASSED\n")

# Test 5: Retrieve with filter
print("=" * 50)
print("TEST 5: Retrieve With Metadata Filter")
print("=" * 50)
results = retrieve(
    "category sales performance",
    collection,
    embedder,
    filters={"type": "category_summary"}
)
assert len(results) > 0, "No filtered results returned"
print(f"Sample filtered result: {results[0][:100]}...")
print("PASSED\n")

print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)