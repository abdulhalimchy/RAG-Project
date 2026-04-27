# =============================================================
# rag/data_preparation.py
# -------------------------------------------------------------
# This module handles all data loading, cleaning, and chunking.
#
# What it does:
#   1. Loads the Superstore CSV dataset
#   2. Cleans and prepares the data
#   3. Converts each row to a natural language sentence
#   4. Creates aggregated summaries (monthly, category, regional)
#   5. Saves all chunks to a JSON file
#
# Why we convert to text?
#   Vector databases work with text, not tables.
#   Each chunk becomes a searchable unit of information.
# =============================================================

import pandas as pd
import json
import os
from tqdm import tqdm
from config import DATA_PATH, CHUNKS_SAVE_PATH, MIN_CHUNK_LENGTH


# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------

def load_data():
    """
    Loads and cleans the Superstore CSV dataset.

    Returns:
        df (DataFrame): cleaned pandas DataFrame

    Raises:
        FileNotFoundError: if CSV file is not found in data/ folder
    """

    # Check if file exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found at: {DATA_PATH}"
            f"\nDownload from: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final"
            f"\nPlace it in the data/ folder"
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, encoding='latin1')

    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()

    # Convert dates to datetime format
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])

    # Extract year and month for analysis
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month_name()
    df['Month_Num'] = df['Order Date'].dt.month

    print(f"Dataset loaded: {len(df)} transactions")
    print(f"Date range : {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
    print(f"Regions    : {sorted(df['Region'].unique())}")
    print(f"Categories : {sorted(df['Category'].unique())}")

    return df


# -------------------------------------------------------------
# CONVERT ROW TO TEXT
# -------------------------------------------------------------

def row_to_text(row):
    """
    Converts a single CSV row into a natural language sentence.

    Why?
        AI models understand sentences, not table rows.
        Each sentence becomes one searchable chunk.

    Args:
        row: a single pandas DataFrame row

    Returns:
        str: natural language description of the transaction
    """

    return (
        f"On {row['Order Date'].strftime('%B %d, %Y')}, "
        f"customer {row['Customer Name']} ({row['Segment']} segment) "
        f"purchased {row['Quantity']} unit(s) of '{row['Product Name']}' "
        f"from the {row['Category']} category "
        f"(sub-category: {row['Sub-Category']}) "
        f"at a price of ${row['Sales']:.2f}. "
        f"A discount of {row['Discount']*100:.0f}% was applied, "
        f"resulting in a profit of ${row['Profit']:.2f}. "
        f"The order was shipped via {row['Ship Mode']} to "
        f"{row['City']}, {row['State']} in the {row['Region']} region."
    )


# -------------------------------------------------------------
# CREATE CHUNKS
# -------------------------------------------------------------

def create_chunks(df):
    """
    Creates all text chunks from the dataset.

    Four types of chunks are created:
        1. Transaction chunks  - one per CSV row (9,994 chunks)
        2. Monthly summaries   - one per month/year (48 chunks)
        3. Category summaries  - one per category (3 chunks)
        4. Regional summaries  - one per region (4 chunks)

    Why multiple chunk types?
        Different questions need different data granularity.
        'Which region is best?' needs regional summary chunks.
        'Who bought chairs?' needs transaction chunks.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        chunks (list): list of dicts with text and metadata
    """

    chunks = []

    # ---------------------------------------------------------
    # 1. TRANSACTION CHUNKS (one per row)
    # ---------------------------------------------------------
    print("\nCreating transaction chunks...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transactions"):
        text = row_to_text(row)

        # Skip chunks that are too short
        if len(text) < MIN_CHUNK_LENGTH:
            continue

        chunks.append({
            "text"    : text,
            "type"    : "transaction",
            "year"    : str(int(row['Year'])),
            "month"   : row['Month'],
            "category": row['Category'],
            "region"  : row['Region']
        })

    print(f"Transaction chunks created: {len(chunks)}")

    # ---------------------------------------------------------
    # 2. MONTHLY SUMMARY CHUNKS
    # ---------------------------------------------------------
    print("\nCreating monthly summary chunks...")

    monthly = df.groupby(['Year', 'Month', 'Month_Num']).agg(
        total_sales  =('Sales', 'sum'),
        total_profit =('Profit', 'sum'),
        total_orders =('Order ID', 'count')
    ).reset_index().sort_values(['Year', 'Month_Num'])

    for _, row in monthly.iterrows():
        text = (
            f"In {row['Month']} {int(row['Year'])}, "
            f"total sales were ${row['total_sales']:.2f} "
            f"with a profit of ${row['total_profit']:.2f} "
            f"across {row['total_orders']} orders."
        )
        chunks.append({
            "text" : text,
            "type" : "monthly_summary",
            "year" : str(int(row['Year'])),
            "month": row['Month']
        })

    print(f"Monthly summary chunks created: {len(monthly)}")

    # ---------------------------------------------------------
    # 3. CATEGORY SUMMARY CHUNKS
    # ---------------------------------------------------------
    print("\nCreating category summary chunks...")

    category = df.groupby('Category').agg(
        total_sales  =('Sales', 'sum'),
        total_profit =('Profit', 'sum'),
        total_orders =('Order ID', 'count')
    ).reset_index()

    for _, row in category.iterrows():
        text = (
            f"The {row['Category']} category generated "
            f"total sales of ${row['total_sales']:.2f} "
            f"with a profit of ${row['total_profit']:.2f} "
            f"across {row['total_orders']} orders "
            f"over the entire dataset period (2014-2017)."
        )
        chunks.append({
            "text"    : text,
            "type"    : "category_summary",
            "category": row['Category']
        })

    print(f"Category summary chunks created: {len(category)}")

    # ---------------------------------------------------------
    # 4. REGIONAL SUMMARY CHUNKS
    # ---------------------------------------------------------
    print("\nCreating regional summary chunks...")

    regional = df.groupby('Region').agg(
        total_sales  =('Sales', 'sum'),
        total_profit =('Profit', 'sum'),
        total_orders =('Order ID', 'count')
    ).reset_index()

    for _, row in regional.iterrows():
        text = (
            f"The {row['Region']} region achieved "
            f"total sales of ${row['total_sales']:.2f} "
            f"with a profit of ${row['total_profit']:.2f} "
            f"across {row['total_orders']} orders."
        )
        chunks.append({
            "text"  : text,
            "type"  : "regional_summary",
            "region": row['Region']
        })

    print(f"Regional summary chunks created: {len(regional)}")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print(f"\nTotal chunks created: {len(chunks)}")

    return chunks


# -------------------------------------------------------------
# SAVE CHUNKS
# -------------------------------------------------------------

def save_chunks(chunks):
    """
    Saves all chunks to a JSON file for reuse.

    Why save chunks?
        So we do not need to recreate them every time.
        Just load from file on next run to save time.

    Args:
        chunks (list): list of chunk dicts
    """

    # Make sure results folder exists
    os.makedirs(os.path.dirname(CHUNKS_SAVE_PATH), exist_ok=True)

    with open(CHUNKS_SAVE_PATH, 'w') as f:
        json.dump(chunks, f, indent=2)

    print(f"\nChunks saved to: {CHUNKS_SAVE_PATH}")


# -------------------------------------------------------------
# LOAD CHUNKS
# -------------------------------------------------------------

def load_chunks():
    """
    Loads previously saved chunks from JSON file.

    Returns:
        chunks (list): list of chunk dicts

    Raises:
        FileNotFoundError: if chunks.json does not exist
    """

    if not os.path.exists(CHUNKS_SAVE_PATH):
        raise FileNotFoundError(
            f"\nChunks file not found at: {CHUNKS_SAVE_PATH}"
            f"\nRun main.py first to generate chunks"
        )

    with open(CHUNKS_SAVE_PATH, 'r') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from: {CHUNKS_SAVE_PATH}")
    return chunks