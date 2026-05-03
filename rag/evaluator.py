# =============================================================
# rag/evaluator.py
# -------------------------------------------------------------
# Evaluates RAG pipeline accuracy against 30 ground truth
# questions calculated directly from the dataset.
#
# How it works:
#   1. Calculate ground truth answers from pandas
#   2. Run each question through the RAG pipeline
#   3. Check if the expected answer appears in RAG response
#   4. Report accuracy and optionally save results
#
# Run standalone:
#   python -m rag.evaluator
# =============================================================

import json
import os
import pandas as pd
from config import DATA_PATH


# -------------------------------------------------------------
# GROUND TRUTH
# -------------------------------------------------------------

def build_ground_truth(df):
    """
    Calculates 30 ground truth Q&A pairs directly from the
    dataset using pandas. These are 100% accurate reference
    answers used to evaluate the RAG pipeline.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of dicts with question, expected answer, keyword
    """

    gt = []

    # --- Region ---
    region_sales  = df.groupby('Region')['Sales'].sum()
    region_profit = df.groupby('Region')['Profit'].sum()
    best_region_sales  = region_sales.idxmax()
    best_region_profit = region_profit.idxmax()
    worst_region_sales = region_sales.idxmin()

    gt.append({"question": "Which region has the highest total sales?",         "expected": best_region_sales})
    gt.append({"question": "Which region has the highest total profit?",         "expected": best_region_profit})
    gt.append({"question": "Which region has the lowest total sales?",           "expected": worst_region_sales})

    # --- Category ---
    cat_sales  = df.groupby('Category')['Sales'].sum()
    cat_profit = df.groupby('Category')['Profit'].sum()
    best_cat_sales  = cat_sales.idxmax()
    best_cat_profit = cat_profit.idxmax()
    worst_cat_profit = cat_profit.idxmin()

    gt.append({"question": "Which product category generates the most revenue?",      "expected": best_cat_sales})
    gt.append({"question": "Which product category generates the highest profit?",    "expected": best_cat_profit})
    gt.append({"question": "Which product category has the lowest profit?",           "expected": worst_cat_profit})

    # --- Sub-Category ---
    subcat_profit = df.groupby('Sub-Category')['Profit'].sum()
    subcat_sales  = df.groupby('Sub-Category')['Sales'].sum()
    best_subcat_profit = subcat_profit.idxmax()
    best_subcat_sales  = subcat_sales.idxmax()
    worst_subcat_profit = subcat_profit.idxmin()

    gt.append({"question": "Which sub-category has the highest profit?",              "expected": best_subcat_profit})
    gt.append({"question": "Which sub-category has the highest sales?",               "expected": best_subcat_sales})
    gt.append({"question": "Which sub-category has the lowest profit?",               "expected": worst_subcat_profit})

    # --- Month / Seasonality ---
    month_sales = df.groupby('Month')['Sales'].sum()
    best_month  = month_sales.idxmax()
    worst_month = month_sales.idxmin()

    gt.append({"question": "Which month has the highest total sales?",               "expected": best_month})
    gt.append({"question": "Which month has the lowest total sales?",                "expected": worst_month})

    # --- Year / Trend ---
    yearly_sales = df.groupby('Year')['Sales'].sum().sort_index()
    best_year    = str(yearly_sales.idxmax())
    worst_year   = str(yearly_sales.idxmin())
    trend        = "increasing" if yearly_sales.iloc[-1] > yearly_sales.iloc[0] else "decreasing"

    gt.append({"question": "Which year had the highest total sales?",                "expected": best_year})
    gt.append({"question": "Which year had the lowest total sales?",                 "expected": worst_year})
    gt.append({"question": "What is the overall sales trend from 2014 to 2017?",     "expected": trend})

    # --- Segment ---
    seg_sales  = df.groupby('Segment')['Sales'].sum()
    seg_profit = df.groupby('Segment')['Profit'].sum()
    best_seg_sales  = seg_sales.idxmax()
    best_seg_profit = seg_profit.idxmax()

    gt.append({"question": "Which customer segment generates the most revenue?",     "expected": best_seg_sales})
    gt.append({"question": "Which customer segment generates the highest profit?",   "expected": best_seg_profit})

    # --- State / City ---
    state_sales = df.groupby('State')['Sales'].sum()
    city_sales  = df.groupby('City')['Sales'].sum()
    best_state  = state_sales.idxmax()
    best_city   = city_sales.idxmax()

    gt.append({"question": "Which state has the highest total sales?",               "expected": best_state})
    gt.append({"question": "Which city has the highest total sales?",                "expected": best_city})

    # --- Shipping ---
    ship_sales  = df.groupby('Ship Mode')['Sales'].sum()
    ship_orders = df.groupby('Ship Mode')['Order ID'].count()
    best_ship_sales  = ship_sales.idxmax()
    most_used_ship   = ship_orders.idxmax()

    gt.append({"question": "Which shipping mode generates the most sales?",          "expected": best_ship_sales})
    gt.append({"question": "Which shipping mode is used most often?",                "expected": most_used_ship})

    # --- Technology vs Furniture ---
    tech_sales = df[df['Category'] == 'Technology']['Sales'].sum()
    furn_sales = df[df['Category'] == 'Furniture']['Sales'].sum()
    tech_profit = df[df['Category'] == 'Technology']['Profit'].sum()
    furn_profit = df[df['Category'] == 'Furniture']['Profit'].sum()
    better_sales  = 'Technology' if tech_sales  > furn_sales  else 'Furniture'
    better_profit = 'Technology' if tech_profit > furn_profit else 'Furniture'

    gt.append({"question": "Between Technology and Furniture, which has higher sales?",  "expected": better_sales})
    gt.append({"question": "Between Technology and Furniture, which has higher profit?", "expected": better_profit})

    # --- West vs East ---
    west_profit = df[df['Region'] == 'West']['Profit'].sum()
    east_profit = df[df['Region'] == 'East']['Profit'].sum()
    better_region = 'West' if west_profit > east_profit else 'East'

    gt.append({"question": "Between West and East, which region has higher profit?",     "expected": better_region})

    # --- Discount impact ---
    high_disc_profit = df[df['Discount'] >= 0.30]['Profit'].sum()
    low_disc_profit  = df[df['Discount'] <  0.30]['Profit'].sum()
    better_discount  = 'low' if low_disc_profit > high_disc_profit else 'high'

    gt.append({"question": "Do high discounts or low discounts result in better profit?", "expected": better_discount})

    # --- Office Supplies ---
    office_sales  = df[df['Category'] == 'Office Supplies']['Sales'].sum()
    office_profit = df[df['Category'] == 'Office Supplies']['Profit'].sum()
    gt.append({"question": "What is the total sales of Office Supplies category?",       "expected": str(int(office_sales))})
    gt.append({"question": "Does the Office Supplies category generate positive profit?", "expected": "positive" if office_profit > 0 else "negative"})

    # --- Q4 Seasonality ---
    df['Quarter'] = df['Order Date'].dt.quarter
    quarter_sales = df.groupby('Quarter')['Sales'].sum()
    best_quarter  = f"Q{quarter_sales.idxmax()}"

    gt.append({"question": "Which quarter has the highest sales?",                       "expected": best_quarter})

    # --- Top product ---
    product_sales = df.groupby('Product Name')['Sales'].sum()
    top_product   = product_sales.idxmax()

    gt.append({"question": "Which product has the highest total sales?",                 "expected": top_product})

    return gt


# -------------------------------------------------------------
# EVALUATE
# -------------------------------------------------------------

def evaluate(df, collection, embedder, save=True):
    """
    Runs 30 ground truth questions through the RAG pipeline
    and measures accuracy.

    A question is marked correct if the expected keyword
    appears anywhere in the RAG answer (case-insensitive).

    Args:
        df         : cleaned pandas DataFrame
        collection : ChromaDB collection
        embedder   : loaded SentenceTransformer model
        save       : if True, saves results to results/evaluation.json

    Returns:
        dict with results and accuracy score
    """

    from rag.rag_pipeline import rag_query

    ground_truth = build_ground_truth(df)

    print("\n" + "=" * 60)
    print("EVALUATION — 30 Ground Truth Questions")
    print("=" * 60)

    correct = 0
    results = []

    for i, item in enumerate(ground_truth, 1):
        result    = rag_query(item['question'], collection, embedder)
        answer    = result['answer']
        expected  = str(item['expected'])
        is_correct = expected.lower() in answer.lower()
        correct   += 1 if is_correct else 0
        status     = "CORRECT" if is_correct else "WRONG"

        print(f"\n[{i:02d}] {item['question']}")
        print(f"     Expected : {expected}")
        print(f"     Status   : {status}")

        results.append({
            "question" : item['question'],
            "expected" : expected,
            "rag_answer": answer[:300],
            "correct"  : is_correct,
            "status"   : status
        })

    accuracy = (correct / len(ground_truth)) * 100

    print("\n" + "=" * 60)
    print(f"Result: {correct}/{len(ground_truth)} correct — Accuracy: {accuracy:.0f}%")
    print("=" * 60)

    report = {
        "total"    : len(ground_truth),
        "correct"  : correct,
        "accuracy" : round(accuracy, 2),
        "results"  : results
    }

    if save:
        path = os.path.join(os.path.dirname(DATA_PATH), '..', 'results', 'evaluation.json')
        path = os.path.normpath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Results saved to: {path}")

    return report


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from rag.data_preparation import load_data, load_chunks, create_chunks, save_chunks
    from rag.vector_store import load_embedding_model, init_chromadb, store_chunks
    from rag.rag_pipeline import load_llm
    from config import CHUNKS_SAVE_PATH

    save = '--nosave' not in sys.argv

    print("=" * 60)
    print("  RAG EVALUATION")
    print("=" * 60)

    if os.path.exists(CHUNKS_SAVE_PATH):
        print("\nLoading saved chunks...")
        chunks = load_chunks()
    else:
        print("\nBuilding chunks from dataset...")
        chunks = create_chunks(load_data())
        save_chunks(chunks)

    df = load_data()

    print("Setting up vector database...")
    embedder   = load_embedding_model()
    collection = init_chromadb()
    store_chunks(collection, chunks, embedder)

    print("Loading LLM...")
    load_llm()

    evaluate(df, collection, embedder, save=save)