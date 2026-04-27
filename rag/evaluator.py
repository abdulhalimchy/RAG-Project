# =============================================================
# rag/evaluator.py
# -------------------------------------------------------------
# This module evaluates the accuracy of the RAG pipeline.
#
# What it does:
#   1. Calculates ground truth answers directly from pandas
#   2. Checks retrieval quality (are right chunks found?)
#   3. Compares RAG answers against ground truth
#   4. Saves evaluation report to results/ folder
#
# Evaluation approach:
#   Ground truth is calculated from the dataset using pandas.
#   This gives 100% accurate reference answers.
#   RAG answers are compared against these references.
#
# Note:
#   Only the 5 required analytical queries are evaluated
#   against ground truth. Interactive queries are saved
#   but not evaluated since ground truth is not known
#   in advance for arbitrary user questions.
# =============================================================

import json
import os
import pandas as pd
from config import DATA_PATH, RESULTS_SAVE_PATH


# -------------------------------------------------------------
# GROUND TRUTH
# -------------------------------------------------------------

def get_ground_truth(df):
    """
    Calculates correct answers directly from the dataset
    using pandas aggregations.

    These answers are 100% accurate and serve as reference
    to evaluate the RAG pipeline answers.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        dict: ground truth answers with values and descriptions
    """

    print("Calculating ground truth from dataset...")

    # Best region by total sales
    region_sales      = df.groupby('Region')['Sales'].sum()
    best_region_sales = region_sales.idxmax()
    best_region_value = region_sales.max()

    # Best region by total profit
    region_profit      = df.groupby('Region')['Profit'].sum()
    best_region_profit = region_profit.idxmax()

    # Best category by total revenue
    category_sales        = df.groupby('Category')['Sales'].sum()
    best_category_revenue = category_sales.idxmax()
    best_category_value   = category_sales.max()

    # Best category by total profit
    category_profit      = df.groupby('Category')['Profit'].sum()
    best_category_profit = category_profit.idxmax()

    # Best month by total sales across all years
    month_sales      = df.groupby('Month')['Sales'].sum()
    best_month       = month_sales.idxmax()
    best_month_value = month_sales.max()

    # Sales trend yearly totals
    yearly_sales    = df.groupby('Year')['Sales'].sum().sort_index()
    trend_direction = (
        "increasing"
        if yearly_sales.iloc[-1] > yearly_sales.iloc[0]
        else "decreasing"
    )

    # Technology vs Furniture
    tech_sales      = df[df['Category'] == 'Technology']['Sales'].sum()
    furn_sales      = df[df['Category'] == 'Furniture']['Sales'].sum()
    better_category = 'Technology' if tech_sales > furn_sales else 'Furniture'

    ground_truth = {
        "best_region_by_sales": {
            "answer"     : best_region_sales,
            "value"      : round(best_region_value, 2),
            "description": "Region with highest total sales"
        },
        "best_region_by_profit": {
            "answer"     : best_region_profit,
            "description": "Region with highest total profit"
        },
        "best_category_by_revenue": {
            "answer"     : best_category_revenue,
            "value"      : round(best_category_value, 2),
            "description": "Category with highest total revenue"
        },
        "best_category_by_profit": {
            "answer"     : best_category_profit,
            "description": "Category with highest total profit"
        },
        "best_month_by_sales": {
            "answer"     : best_month,
            "value"      : round(best_month_value, 2),
            "description": "Month with highest total sales"
        },
        "sales_trend": {
            "answer"     : trend_direction,
            "description": "Overall sales trend 2014 to 2017",
            "yearly"     : {
                str(k): round(v, 2)
                for k, v in yearly_sales.to_dict().items()
            }
        },
        "tech_vs_furniture": {
            "answer"     : better_category,
            "tech_sales" : round(tech_sales, 2),
            "furn_sales" : round(furn_sales, 2),
            "description": "Category with higher total sales"
        }
    }

    print("Ground truth calculated successfully")
    return ground_truth


# -------------------------------------------------------------
# PRINT GROUND TRUTH
# -------------------------------------------------------------

def print_ground_truth(ground_truth):
    """
    Prints all ground truth answers in a readable format.

    Args:
        ground_truth (dict): ground truth answers
    """

    print("\n" + "=" * 60)
    print("GROUND TRUTH (calculated from dataset)")
    print("=" * 60)

    for key, value in ground_truth.items():
        print(f"\n{value['description']}:")
        print(f"  Answer : {value['answer']}")
        if 'value' in value:
            print(f"  Value  : ${value['value']:,.2f}")
        if 'yearly' in value:
            for year, sales in value['yearly'].items():
                print(f"  {year}   : ${sales:,.2f}")
        if 'tech_sales' in value:
            print(f"  Technology : ${value['tech_sales']:,.2f}")
            print(f"  Furniture  : ${value['furn_sales']:,.2f}")

    print("=" * 60)


# -------------------------------------------------------------
# EVALUATE RETRIEVAL
# -------------------------------------------------------------

def evaluate_retrieval(collection, embedder):
    """
    Checks if ChromaDB retrieves the correct chunks for
    each of the 5 required analytical queries.

    A retrieval is considered correct if the expected
    keyword appears in any of the retrieved chunks.

    Args:
        collection : ChromaDB collection
        embedder   : loaded SentenceTransformer model

    Returns:
        dict: retrieval evaluation results and accuracy
    """

    from rag.vector_store import retrieve

    print("\n" + "=" * 60)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 60)

    tests = [
        {
            "query"   : "which region has best sales performance",
            "expected": "West",
            "label"   : "Best region query"
        },
        {
            "query"   : "which category generates most revenue",
            "expected": "Technology",
            "label"   : "Best category query"
        },
        {
            "query"   : "highest sales month seasonality",
            "expected": "December",
            "label"   : "Best month query"
        },
        {
            "query"   : "technology vs furniture comparison",
            "expected": "Technology",
            "label"   : "Tech vs Furniture query"
        },
        {
            "query"   : "sales trend 2014 2015 2016 2017",
            "expected": "total sales",
            "label"   : "Sales trend query"
        }
    ]

    correct = 0
    results = []

    for test in tests:
        chunks = retrieve(test['query'], collection, embedder)
        found  = any(
            test['expected'].lower() in c.lower()
            for c in chunks
        )
        correct += 1 if found else 0
        status  = "PASS" if found else "FAIL"

        print(f"\n{test['label']}:")
        print(f"  Query    : {test['query']}")
        print(f"  Expected : {test['expected']}")
        print(f"  Found    : {found}")
        print(f"  Status   : {status}")

        results.append({
            "label"   : test['label'],
            "found"   : found,
            "status"  : status
        })

    accuracy = (correct / len(tests)) * 100
    print(f"\nRetrieval Accuracy: {correct}/{len(tests)} = {accuracy:.0f}%")
    print("=" * 60)

    return {
        "results" : results,
        "accuracy": accuracy
    }


# -------------------------------------------------------------
# EVALUATE REQUIRED QUERIES
# -------------------------------------------------------------

def evaluate_answers(rag_results, ground_truth):
    """
    Compares the 5 required RAG query answers against
    ground truth answers.

    A RAG answer is considered correct if the expected
    keyword appears anywhere in the answer text.

    Only the 5 required analytical queries are evaluated.
    Interactive queries are excluded since ground truth
    is not known in advance for arbitrary questions.

    Args:
        rag_results  (list): list of question/answer dicts
        ground_truth (dict): ground truth answers

    Returns:
        dict: answer evaluation results and accuracy
    """

    print("\n" + "=" * 60)
    print("ANSWER ACCURACY EVALUATION")
    print("=" * 60)

    # Define the 5 required queries and their expected answers
    tests = [
        {
            "question": "Which region has the best sales performance?",
            "expected": ground_truth['best_region_by_sales']['answer'],
            "label"   : "Best region"
        },
        {
            "question": "Which product category generates the most revenue and profit?",
            "expected": ground_truth['best_category_by_revenue']['answer'],
            "label"   : "Best category"
        },
        {
            "question": "Which months show the highest sales? Is there seasonality?",
            "expected": ground_truth['best_month_by_sales']['answer'],
            "label"   : "Best month"
        },
        {
            "question": "Compare Technology vs Furniture sales and profit performance.",
            "expected": ground_truth['tech_vs_furniture']['answer'],
            "label"   : "Tech vs Furniture"
        },
        {
            "question": "What is the sales trend over the 4-year period from 2014 to 2017?",
            "expected": ground_truth['sales_trend']['answer'],
            "label"   : "Sales trend"
        }
    ]

    correct = 0
    results = []

    for test in tests:
        # Find matching RAG answer by partial question match
        rag_answer = ""
        for r in rag_results:
            if any(
                word in r['question'].lower()
                for word in test['question'].lower().split()
                if len(word) > 4
            ):
                rag_answer = r['answer']
                break

        # Check if expected answer appears in RAG response
        is_correct = test['expected'].lower() in rag_answer.lower()
        correct   += 1 if is_correct else 0
        status     = "CORRECT" if is_correct else "WRONG"

        print(f"\n{test['label']}:")
        print(f"  Expected : {test['expected']}")
        print(f"  RAG said : {rag_answer[:120]}...")
        print(f"  Status   : {status}")

        results.append({
            "label"     : test['label'],
            "expected"  : test['expected'],
            "rag_answer": rag_answer[:300],
            "correct"   : is_correct,
            "status"    : status
        })

    accuracy = (correct / len(tests)) * 100
    print(f"\nAnswer Accuracy: {correct}/{len(tests)} = {accuracy:.0f}%")
    print("=" * 60)

    return {
        "results" : results,
        "accuracy": accuracy
    }


# -------------------------------------------------------------
# SAVE EVALUATION REPORT
# -------------------------------------------------------------

def save_evaluation_report(report):
    """
    Saves the full evaluation report to a JSON file.

    Why save?
        Provides a record for the technical report.
        Screenshots of this file can be included in
        the project submission.

    Args:
        report (dict): full evaluation report
    """

    report_path = os.path.join(
        os.path.dirname(RESULTS_SAVE_PATH),
        'evaluation_report.json'
    )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Convert any non-serializable values
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        if isinstance(obj, float):
            return round(obj, 4)
        return obj

    with open(report_path, 'w') as f:
        json.dump(clean(report), f, indent=2)

    print(f"\nEvaluation report saved to: {report_path}")


# -------------------------------------------------------------
# FULL EVALUATION
# -------------------------------------------------------------

def evaluate(df, collection, embedder, rag_results):
    """
    Runs the full evaluation and prints a final report.

    Evaluates only the 5 required analytical queries.
    Interactive queries are excluded from evaluation.

    Args:
        df          : cleaned pandas DataFrame
        collection  : ChromaDB collection
        embedder    : loaded SentenceTransformer model
        rag_results : list of question/answer dicts from
                      the required analysis queries

    Returns:
        dict: full evaluation report
    """

    # Calculate ground truth
    ground_truth = get_ground_truth(df)
    print_ground_truth(ground_truth)

    # Evaluate retrieval quality
    retrieval_eval = evaluate_retrieval(collection, embedder)

    # Evaluate answer accuracy
    answer_eval = evaluate_answers(rag_results, ground_truth)

    # Final report
    overall = (
        retrieval_eval['accuracy'] + answer_eval['accuracy']
    ) / 2

    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT")
    print("=" * 60)
    print(f"Retrieval Accuracy : {retrieval_eval['accuracy']:.0f}%")
    print(f"Answer Accuracy    : {answer_eval['accuracy']:.0f}%")
    print(f"Overall Accuracy   : {overall:.0f}%")
    print("=" * 60)

    report = {
        "ground_truth"   : ground_truth,
        "retrieval_eval" : retrieval_eval,
        "answer_eval"    : answer_eval,
        "overall_accuracy": overall
    }

    # Save report to file
    save_evaluation_report(report)

    return report