import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test for evaluator.py

from rag.data_preparation import load_data, create_chunks
from rag.vector_store import load_embedding_model, init_chromadb, store_chunks
from rag.evaluator import get_ground_truth, print_ground_truth, evaluate_retrieval

# Test 1: Ground truth calculation
print("=" * 50)
print("TEST 1: Ground Truth Calculation")
print("=" * 50)
df           = load_data()
ground_truth = get_ground_truth(df)
assert ground_truth['best_region_by_sales']['answer'] == 'West', "Wrong best region"
assert ground_truth['best_category_by_revenue']['answer'] == 'Technology', "Wrong best category"
assert ground_truth['tech_vs_furniture']['answer'] == 'Technology', "Wrong tech vs furniture"
print("PASSED\n")

# Test 2: Print ground truth
print("=" * 50)
print("TEST 2: Print Ground Truth")
print("=" * 50)
print_ground_truth(ground_truth)
print("PASSED\n")

# Test 3: Retrieval evaluation
print("=" * 50)
print("TEST 3: Retrieval Evaluation")
print("=" * 50)
chunks     = create_chunks(df)
embedder   = load_embedding_model()
collection = init_chromadb()
store_chunks(collection, chunks, embedder)
result     = evaluate_retrieval(collection, embedder)
assert 'accuracy' in result, "Accuracy missing from result"
print("PASSED\n")

print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)