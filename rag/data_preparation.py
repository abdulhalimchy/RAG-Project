# =============================================================
# rag/data_preparation.py
# -------------------------------------------------------------
# This module handles all data loading, cleaning, and chunking.
#
# What it does:
#   1. Loads the Superstore CSV dataset
#   2. Cleans and prepares the data
#   3. Converts each row to a natural language sentence
#   4. Creates aggregated summaries of multiple types
#   5. Saves all chunks to a JSON file
#
# Chunk types created:
#   - transaction        : one per CSV row (9,994 chunks)
#   - overall_summary    : one summary of entire dataset
#   - yearly_summary     : one per year (4 chunks)
#   - monthly_summary    : one per month/year (48 chunks)
#   - category_summary   : one per category (3 chunks)
#   - subcategory_summary: one per sub-category (17 chunks)
#   - regional_summary   : one per region (4 chunks)
#   - category_year      : category performance per year
#   - region_year        : region performance per year
#   - top_states         : top 10 states by sales
#   - top_cities         : top 10 cities by sales
#   - top_performers     : explicit best performers
#   - segment_summary    : one per customer segment
#   - ship_mode_summary  : one per shipping mode
#   - discount_summary   : high vs low discount analysis
#   - growth_summary     : year over year growth
#   - product_summary    : most discounted products
#
# Why so many chunk types?
#   Different questions need different data granularity.
#   More chunk types means better retrieval accuracy
#   and more accurate LLM answers.
# =============================================================

import pandas as pd
import json
import os
from tqdm import tqdm
from config import DATA_PATH, CHUNKS_SAVE_PATH


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

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found at: {DATA_PATH}"
            f"\nDownload from: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final"
            f"\nPlace it in the data/ folder"
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, encoding='latin1')

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert dates to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date']  = pd.to_datetime(df['Ship Date'])

    # Extract time dimensions
    df['Year']      = df['Order Date'].dt.year
    df['Month']     = df['Order Date'].dt.month_name()
    df['Month_Num'] = df['Order Date'].dt.month
    df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)

    # Calculate profit margin per row
    df['Profit_Margin'] = (df['Profit'] / df['Sales'] * 100).round(2)

    print(f"Dataset loaded: {len(df)} transactions")
    print(f"Date range  : {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
    print(f"Regions     : {sorted(df['Region'].unique())}")
    print(f"Categories  : {sorted(df['Category'].unique())}")
    print(f"Segments    : {sorted(df['Segment'].unique())}")

    return df


# -------------------------------------------------------------
# HELPER
# -------------------------------------------------------------

def _margin(profit, sales):
    """
    Calculates profit margin percentage safely.

    Args:
        profit : total profit value
        sales  : total sales value

    Returns:
        float: profit margin as percentage
    """
    return round((profit / sales) * 100, 2) if sales else 0.0


def _make_chunk(text, metadata):
    """
    Creates a chunk dict from text and metadata.
    Strips extra whitespace from text.

    Args:
        text     : the chunk text
        metadata : dict of metadata fields

    Returns:
        dict: chunk with text and metadata
    """
    clean_text = ' '.join(text.split())
    return {"text": clean_text, **metadata}


# -------------------------------------------------------------
# TRANSACTION CHUNKS
# -------------------------------------------------------------

def create_transaction_chunks(df):
    """
    Creates one text chunk per CSV row.

    Each transaction is converted to a natural language
    sentence describing the full order details.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transaction chunks"):
        text = (
            f"Order {row['Order ID']} was placed on "
            f"{row['Order Date'].strftime('%B %d, %Y')} by customer "
            f"{row['Customer Name']} from the {row['Segment']} segment "
            f"in {row['City']}, {row['State']} ({row['Region']} region). "
            f"The customer purchased {row['Quantity']} unit(s) of "
            f"'{row['Product Name']}' from the {row['Category']} category "
            f"(sub-category: {row['Sub-Category']}) at a price of "
            f"${row['Sales']:.2f}. A discount of "
            f"{row['Discount']*100:.0f}% was applied, resulting in a "
            f"profit of ${row['Profit']:.2f}. The order was shipped via "
            f"{row['Ship Mode']}."
        )

        chunks.append(_make_chunk(text, {
            "type"        : "transaction",
            "year"        : str(int(row['Year'])),
            "month"       : row['Month'],
            "category"    : row['Category'],
            "sub_category": row['Sub-Category'],
            "region"      : row['Region'],
            "state"       : row['State'],
            "city"        : row['City'],
            "segment"     : row['Segment'],
            "ship_mode"   : row['Ship Mode']
        }))

    print(f"Transaction chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# OVERALL SUMMARY
# -------------------------------------------------------------

def create_overall_summary(df):
    """
    Creates one summary chunk for the entire dataset.

    This gives the LLM a high-level view of the dataset
    when answering general questions.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list with one chunk dict
    """

    total_sales    = df['Sales'].sum()
    total_profit   = df['Profit'].sum()
    total_quantity = df['Quantity'].sum()
    avg_discount   = df['Discount'].mean()
    margin         = _margin(total_profit, total_sales)

    text = (
        f"The dataset contains {len(df)} transactions from 2014 to 2017. "
        f"Total sales were ${total_sales:.2f}, total profit was "
        f"${total_profit:.2f}, total quantity sold was {int(total_quantity)}, "
        f"average discount was {avg_discount*100:.1f}%, and overall profit "
        f"margin was {margin:.2f}%."
    )

    print("Overall summary: 1 chunk")
    return [_make_chunk(text, {"type": "overall_summary"})]


# -------------------------------------------------------------
# YEARLY SUMMARY
# -------------------------------------------------------------

def create_yearly_summary(df):
    """
    Creates one summary chunk per year (2014-2017).

    Yearly summaries help answer trend and growth questions
    by providing aggregated annual performance data.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks  = []
    yearly  = df.groupby('Year').agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum'),
        avg_discount  =('Discount', 'mean'),
        total_orders  =('Order ID', 'count')
    ).reset_index().sort_values('Year')

    for _, row in yearly.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"In {int(row['Year'])}, total sales were ${row['total_sales']:.2f}, "
            f"total profit was ${row['total_profit']:.2f}, "
            f"total quantity sold was {int(row['total_quantity'])}, "
            f"average discount was {row['avg_discount']*100:.1f}%, "
            f"profit margin was {margin:.2f}%, "
            f"and total orders were {row['total_orders']}."
        )
        chunks.append(_make_chunk(text, {
            "type": "yearly_summary",
            "year": str(int(row['Year']))
        }))

    print(f"Yearly summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# MONTHLY SUMMARY
# -------------------------------------------------------------

def create_monthly_summary(df):
    """
    Creates one summary chunk per month per year (48 chunks).

    Monthly summaries help answer seasonality questions
    and identify peak sales periods.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks  = []
    monthly = df.groupby(['Year', 'Month', 'Month_Num']).agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum'),
        avg_discount  =('Discount', 'mean'),
        total_orders  =('Order ID', 'count')
    ).reset_index().sort_values(['Year', 'Month_Num'])

    for _, row in monthly.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"In {row['Month']} {int(row['Year'])}, total sales were "
            f"${row['total_sales']:.2f}, total profit was "
            f"${row['total_profit']:.2f}, total quantity sold was "
            f"{int(row['total_quantity'])}, average discount was "
            f"{row['avg_discount']*100:.1f}%, profit margin was "
            f"{margin:.2f}%, and total orders were {row['total_orders']}."
        )
        chunks.append(_make_chunk(text, {
            "type" : "monthly_summary",
            "year" : str(int(row['Year'])),
            "month": row['Month']
        }))

    print(f"Monthly summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# CATEGORY SUMMARY
# -------------------------------------------------------------

def create_category_summary(df):
    """
    Creates one summary chunk per product category.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks   = []
    category = df.groupby('Category').agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum'),
        avg_discount  =('Discount', 'mean'),
        total_orders  =('Order ID', 'count')
    ).reset_index()

    for _, row in category.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"The {row['Category']} category generated total sales of "
            f"${row['total_sales']:.2f}, total profit of "
            f"${row['total_profit']:.2f}, total quantity sold of "
            f"{int(row['total_quantity'])}, average discount of "
            f"{row['avg_discount']*100:.1f}%, profit margin of "
            f"{margin:.2f}%, and total orders of {row['total_orders']} "
            f"over the entire dataset period (2014-2017)."
        )
        chunks.append(_make_chunk(text, {
            "type"    : "category_summary",
            "category": row['Category']
        }))

    print(f"Category summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# SUB-CATEGORY SUMMARY
# -------------------------------------------------------------

def create_subcategory_summary(df):
    """
    Creates one summary chunk per product sub-category.

    Sub-category summaries help answer detailed product
    performance questions such as profit margins per
    sub-category and discount patterns.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks      = []
    subcategory = df.groupby(['Category', 'Sub-Category']).agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum'),
        avg_discount  =('Discount', 'mean'),
        total_orders  =('Order ID', 'count')
    ).reset_index()

    for _, row in subcategory.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"The {row['Sub-Category']} sub-category (under {row['Category']}) "
            f"generated total sales of ${row['total_sales']:.2f}, "
            f"total profit of ${row['total_profit']:.2f}, "
            f"total quantity sold of {int(row['total_quantity'])}, "
            f"average discount of {row['avg_discount']*100:.1f}%, "
            f"and profit margin of {margin:.2f}%."
        )
        chunks.append(_make_chunk(text, {
            "type"        : "subcategory_summary",
            "category"    : row['Category'],
            "sub_category": row['Sub-Category']
        }))

    print(f"Sub-category summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# REGIONAL SUMMARY
# -------------------------------------------------------------

def create_regional_summary(df):
    """
    Creates one summary chunk per region.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks   = []
    regional = df.groupby('Region').agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum'),
        avg_discount  =('Discount', 'mean'),
        total_orders  =('Order ID', 'count')
    ).reset_index()

    for _, row in regional.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"The {row['Region']} region achieved total sales of "
            f"${row['total_sales']:.2f}, total profit of "
            f"${row['total_profit']:.2f}, total quantity sold of "
            f"{int(row['total_quantity'])}, average discount of "
            f"{row['avg_discount']*100:.1f}%, profit margin of "
            f"{margin:.2f}%, and total orders of {row['total_orders']}."
        )
        chunks.append(_make_chunk(text, {
            "type"  : "regional_summary",
            "region": row['Region']
        }))

    print(f"Regional summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# CATEGORY PER YEAR SUMMARY
# -------------------------------------------------------------

def create_category_year_summary(df):
    """
    Creates one summary chunk per category per year.

    These chunks help answer questions about how category
    performance changed over time.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks        = []
    category_year = df.groupby(['Year', 'Category']).agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum'),
        avg_discount  =('Discount', 'mean')
    ).reset_index()

    for _, row in category_year.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"In {int(row['Year'])}, the {row['Category']} category "
            f"had sales of ${row['total_sales']:.2f}, profit of "
            f"${row['total_profit']:.2f}, quantity sold of "
            f"{int(row['total_quantity'])}, average discount of "
            f"{row['avg_discount']*100:.1f}%, and profit margin of "
            f"{margin:.2f}%."
        )
        chunks.append(_make_chunk(text, {
            "type"    : "category_year_summary",
            "year"    : str(int(row['Year'])),
            "category": row['Category']
        }))

    print(f"Category-year summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# REGION PER YEAR SUMMARY
# -------------------------------------------------------------

def create_region_year_summary(df):
    """
    Creates one summary chunk per region per year.

    These chunks help answer questions about regional
    performance trends over time.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks      = []
    region_year = df.groupby(['Year', 'Region']).agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum'),
        avg_discount  =('Discount', 'mean')
    ).reset_index()

    for _, row in region_year.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"In {int(row['Year'])}, the {row['Region']} region "
            f"had sales of ${row['total_sales']:.2f}, profit of "
            f"${row['total_profit']:.2f}, quantity sold of "
            f"{int(row['total_quantity'])}, average discount of "
            f"{row['avg_discount']*100:.1f}%, and profit margin of "
            f"{margin:.2f}%."
        )
        chunks.append(_make_chunk(text, {
            "type"  : "region_year_summary",
            "year"  : str(int(row['Year'])),
            "region": row['Region']
        }))

    print(f"Region-year summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# TOP STATES SUMMARY
# -------------------------------------------------------------

def create_top_states_summary(df):
    """
    Creates one summary chunk per top 10 state by sales.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks     = []
    top_states = df.groupby('State').agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum')
    ).reset_index().sort_values('total_sales', ascending=False).head(10)

    for _, row in top_states.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"The state of {row['State']} is one of the top states "
            f"by sales. Total sales were ${row['total_sales']:.2f}, "
            f"total profit was ${row['total_profit']:.2f}, total "
            f"quantity sold was {int(row['total_quantity'])}, and "
            f"profit margin was {margin:.2f}%."
        )
        chunks.append(_make_chunk(text, {
            "type" : "top_state_summary",
            "state": row['State']
        }))

    print(f"Top states summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# TOP CITIES SUMMARY
# -------------------------------------------------------------

def create_top_cities_summary(df):
    """
    Creates one summary chunk per top 10 city by sales.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks     = []
    top_cities = df.groupby('City').agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum')
    ).reset_index().sort_values('total_sales', ascending=False).head(10)

    for _, row in top_cities.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"The city of {row['City']} is one of the top cities "
            f"by sales. Total sales were ${row['total_sales']:.2f}, "
            f"total profit was ${row['total_profit']:.2f}, total "
            f"quantity sold was {int(row['total_quantity'])}, and "
            f"profit margin was {margin:.2f}%."
        )
        chunks.append(_make_chunk(text, {
            "type": "top_city_summary",
            "city": row['City']
        }))

    print(f"Top cities summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# TOP PERFORMERS SUMMARY
# -------------------------------------------------------------

def create_top_performers_summary(df):
    """
    Creates explicit top performer summary chunks.

    These chunks directly state who is the best performer
    in each dimension, helping the LLM answer comparative
    questions accurately without needing to compare numbers.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks = []

    # Category performers
    cat = df.groupby('Category').agg(
        total_sales =('Sales', 'sum'),
        total_profit=('Profit', 'sum')
    ).reset_index()

    top_cat_sales  = cat.sort_values('total_sales', ascending=False).iloc[0]
    top_cat_profit = cat.sort_values('total_profit', ascending=False).iloc[0]

    chunks.append(_make_chunk(
        f"The category with the highest total sales was "
        f"{top_cat_sales['Category']} with sales of "
        f"${top_cat_sales['total_sales']:.2f}. The category with "
        f"the highest total profit was {top_cat_profit['Category']} "
        f"with profit of ${top_cat_profit['total_profit']:.2f}.",
        {"type": "top_performer_summary", "dimension": "category"}
    ))

    # Region performers
    reg = df.groupby('Region').agg(
        total_sales =('Sales', 'sum'),
        total_profit=('Profit', 'sum')
    ).reset_index()

    top_reg_sales  = reg.sort_values('total_sales', ascending=False).iloc[0]
    top_reg_profit = reg.sort_values('total_profit', ascending=False).iloc[0]

    chunks.append(_make_chunk(
        f"The region with the highest total sales was "
        f"{top_reg_sales['Region']} with sales of "
        f"${top_reg_sales['total_sales']:.2f}. The region with "
        f"the highest total profit was {top_reg_profit['Region']} "
        f"with profit of ${top_reg_profit['total_profit']:.2f}.",
        {"type": "top_performer_summary", "dimension": "region"}
    ))

    # Sub-category performers
    subcat = df.groupby('Sub-Category').agg(
        total_profit=('Profit', 'sum')
    ).reset_index()

    top_subcat = subcat.sort_values('total_profit', ascending=False).iloc[0]

    chunks.append(_make_chunk(
        f"The sub-category with the highest total profit was "
        f"{top_subcat['Sub-Category']} with profit of "
        f"${top_subcat['total_profit']:.2f}.",
        {"type": "top_performer_summary", "dimension": "sub_category"}
    ))

    print(f"Top performer summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# SEGMENT SUMMARY
# -------------------------------------------------------------

def create_segment_summary(df):
    """
    Creates one summary chunk per customer segment.

    Customer segments are Consumer, Corporate, and Home Office.
    These chunks help answer questions about customer behavior.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks  = []
    segment = df.groupby('Segment').agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum'),
        avg_discount  =('Discount', 'mean'),
        total_orders  =('Order ID', 'count')
    ).reset_index()

    for _, row in segment.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"The {row['Segment']} customer segment generated "
            f"total sales of ${row['total_sales']:.2f}, total "
            f"profit of ${row['total_profit']:.2f}, total quantity "
            f"sold of {int(row['total_quantity'])}, average discount "
            f"of {row['avg_discount']*100:.1f}%, profit margin of "
            f"{margin:.2f}%, and total orders of {row['total_orders']}."
        )
        chunks.append(_make_chunk(text, {
            "type"   : "segment_summary",
            "segment": row['Segment']
        }))

    print(f"Segment summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# SHIP MODE SUMMARY
# -------------------------------------------------------------

def create_ship_mode_summary(df):
    """
    Creates one summary chunk per shipping mode.

    Ship mode summaries help answer questions about delivery
    preferences and their impact on sales and profit.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks    = []
    ship_mode = df.groupby('Ship Mode').agg(
        total_sales   =('Sales', 'sum'),
        total_profit  =('Profit', 'sum'),
        total_quantity=('Quantity', 'sum'),
        total_orders  =('Order ID', 'count')
    ).reset_index()

    for _, row in ship_mode.iterrows():
        margin = _margin(row['total_profit'], row['total_sales'])
        text   = (
            f"Orders shipped via {row['Ship Mode']} had total sales "
            f"of ${row['total_sales']:.2f}, total profit of "
            f"${row['total_profit']:.2f}, total quantity sold of "
            f"{int(row['total_quantity'])}, profit margin of "
            f"{margin:.2f}%, and total orders of {row['total_orders']}."
        )
        chunks.append(_make_chunk(text, {
            "type"     : "ship_mode_summary",
            "ship_mode": row['Ship Mode']
        }))

    print(f"Ship mode summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# DISCOUNT ANALYSIS SUMMARY
# -------------------------------------------------------------

def create_discount_summary(df):
    """
    Creates summary chunks comparing high vs low discount impact.

    High discount is defined as 30% or more.
    Low discount is defined as below 30%.

    These chunks help answer questions about discount strategies
    and their effect on profit.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks        = []
    high_discount = df[df['Discount'] >= 0.30]
    low_discount  = df[df['Discount'] <  0.30]

    text = (
        f"Transactions with high discounts of 30% or more had "
        f"total sales of ${high_discount['Sales'].sum():.2f} and "
        f"total profit of ${high_discount['Profit'].sum():.2f}. "
        f"Transactions with discounts below 30% had total sales of "
        f"${low_discount['Sales'].sum():.2f} and total profit of "
        f"${low_discount['Profit'].sum():.2f}. High discounts "
        f"significantly reduce profit margins."
    )
    chunks.append(_make_chunk(text, {"type": "discount_summary"}))

    # Most discounted products
    product_discount = df.groupby('Product Name').agg(
        avg_discount=('Discount', 'mean'),
        total_sales =('Sales', 'sum'),
        total_profit=('Profit', 'sum')
    ).reset_index().sort_values('avg_discount', ascending=False).head(5)

    for _, row in product_discount.iterrows():
        text = (
            f"The product '{row['Product Name']}' had an average "
            f"discount of {row['avg_discount']*100:.1f}%, total "
            f"sales of ${row['total_sales']:.2f}, and total profit "
            f"of ${row['total_profit']:.2f}."
        )
        chunks.append(_make_chunk(text, {
            "type"   : "product_discount_summary",
            "product": row['Product Name']
        }))

    print(f"Discount summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# YEAR OVER YEAR GROWTH SUMMARY
# -------------------------------------------------------------

def create_growth_summary(df):
    """
    Creates year over year growth summary chunks.

    Growth summaries help the LLM answer trend questions
    with specific percentage growth figures rather than
    just raw sales numbers.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        list of chunk dicts
    """

    chunks       = []
    yearly_sales = df.groupby('Year')['Sales'].sum().sort_index()
    years        = list(yearly_sales.index)

    for i in range(1, len(years)):
        prev_year  = years[i - 1]
        curr_year  = years[i]
        prev_sales = yearly_sales[prev_year]
        curr_sales = yearly_sales[curr_year]
        growth     = ((curr_sales - prev_sales) / prev_sales) * 100

        direction = "increased" if growth > 0 else "decreased"

        text = (
            f"Sales {direction} from {prev_year} to {curr_year} "
            f"by {abs(growth):.1f}%. Sales in {prev_year} were "
            f"${prev_sales:.2f} and in {curr_year} were "
            f"${curr_sales:.2f}."
        )
        chunks.append(_make_chunk(text, {
            "type"     : "growth_summary",
            "from_year": str(prev_year),
            "to_year"  : str(curr_year)
        }))

    # Overall growth
    first_year  = years[0]
    last_year   = years[-1]
    first_sales = yearly_sales[first_year]
    last_sales  = yearly_sales[last_year]
    total_growth = ((last_sales - first_sales) / first_sales) * 100
    direction    = "increased" if total_growth > 0 else "decreased"

    text = (
        f"Overall, sales {direction} by {abs(total_growth):.1f}% "
        f"from {first_year} to {last_year}. Sales in {first_year} "
        f"were ${first_sales:.2f} and in {last_year} were "
        f"${last_sales:.2f}."
    )
    chunks.append(_make_chunk(text, {
        "type"     : "growth_summary",
        "from_year": str(first_year),
        "to_year"  : str(last_year)
    }))

    print(f"Growth summary chunks: {len(chunks)}")
    return chunks


# -------------------------------------------------------------
# CREATE ALL CHUNKS
# -------------------------------------------------------------

def create_chunks(df):
    """
    Creates all chunk types from the dataset.

    Calls all individual chunk creation functions and
    combines their results into one list.

    Args:
        df (DataFrame): cleaned pandas DataFrame

    Returns:
        chunks (list): list of all chunk dicts
    """

    print("\nCreating all chunks...")
    print("-" * 60)

    chunks = []
    chunks += create_transaction_chunks(df)
    chunks += create_overall_summary(df)
    chunks += create_yearly_summary(df)
    chunks += create_monthly_summary(df)
    chunks += create_category_summary(df)
    chunks += create_subcategory_summary(df)
    chunks += create_regional_summary(df)
    chunks += create_category_year_summary(df)
    chunks += create_region_year_summary(df)
    chunks += create_top_states_summary(df)
    chunks += create_top_cities_summary(df)
    chunks += create_top_performers_summary(df)
    chunks += create_segment_summary(df)
    chunks += create_ship_mode_summary(df)
    chunks += create_discount_summary(df)
    chunks += create_growth_summary(df)

    print("-" * 60)
    print(f"Total chunks created: {len(chunks)}")

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

    os.makedirs(os.path.dirname(CHUNKS_SAVE_PATH), exist_ok=True)

    with open(CHUNKS_SAVE_PATH, 'w') as f:
        json.dump(chunks, f, indent=2)

    print(f"Chunks saved to: {CHUNKS_SAVE_PATH}")


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