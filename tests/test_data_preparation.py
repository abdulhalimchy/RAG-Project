import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Quick test for data_preparation.py
from rag.data_preparation import load_data, create_chunks, save_chunks, load_chunks

# Test 1: Load data
print("=" * 50)
print("TEST 1: Load Data")
print("=" * 50)
df = load_data()
assert df is not None, "DataFrame is None"
assert len(df) == 9994, f"Expected 9994 rows, got {len(df)}"
assert 'Year' in df.columns, "Year column missing"
assert 'Month' in df.columns, "Month column missing"
print("PASSED\n")

# Test 2: Create chunks
print("=" * 50)
print("TEST 2: Create Chunks")
print("=" * 50)
chunks = create_chunks(df)
assert len(chunks) > 0, "No chunks created"
assert len(chunks) == 10049, f"Expected 10049 chunks, got {len(chunks)}"
assert 'text' in chunks[0], "text key missing in chunk"
assert 'type' in chunks[0], "type key missing in chunk"
print("PASSED\n")

# Test 3: Save chunks
print("=" * 50)
print("TEST 3: Save Chunks")
print("=" * 50)
save_chunks(chunks)
print("PASSED\n")

# Test 4: Load chunks
print("=" * 50)
print("TEST 4: Load Chunks")
print("=" * 50)
loaded = load_chunks()
assert len(loaded) == len(chunks), "Loaded chunks count mismatch"
print("PASSED\n")

print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)