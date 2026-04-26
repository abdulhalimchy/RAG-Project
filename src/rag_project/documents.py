from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document


def _prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["Order Date"] = pd.to_datetime(prepared["Order Date"])
    prepared["Year"] = prepared["Order Date"].dt.year
    prepared["YearMonth"] = prepared["Order Date"].dt.to_period("M").astype(str)
    return prepared


def create_transaction_documents(df: pd.DataFrame) -> list[Document]:
    documents: list[Document] = []
    prepared = _prepare_dates(df)

    for _, row in prepared.iterrows():
        text = f"""
        Order {row['Order ID']} was placed on {row['Order Date'].date()} by {row['Customer Name']}
        from the {row['Segment']} segment in {row['City']}, {row['State']} ({row['Region']} region).
        The customer purchased {row['Product Name']} from the {row['Category']} category,
        sub-category {row['Sub-Category']}. Sales were {row['Sales']:.2f}, quantity was
        {row['Quantity']}, discount was {row['Discount']:.2f}, and profit was {row['Profit']:.2f}.
        """
        documents.append(
            Document(
                page_content=" ".join(text.split()),
                metadata={
                    "type": "transaction",
                    "order_id": row["Order ID"],
                    "year": int(row["Year"]),
                    "region": row["Region"],
                    "state": row["State"],
                    "city": row["City"],
                    "category": row["Category"],
                    "sub_category": row["Sub-Category"],
                },
            )
        )

    return documents


def create_summary_documents(df: pd.DataFrame) -> list[Document]:
    documents: list[Document] = []
    prepared = _prepare_dates(df)

    def add_doc(text: str, metadata: dict) -> None:
        documents.append(Document(page_content=" ".join(text.split()), metadata=metadata))

    total_sales = prepared["Sales"].sum()
    total_profit = prepared["Profit"].sum()
    total_quantity = prepared["Quantity"].sum()
    avg_discount = prepared["Discount"].mean()
    profit_margin = (total_profit / total_sales) * 100 if total_sales else 0.0

    add_doc(
        f"""
        Overall, the dataset contains {len(prepared)} transactions. Total sales were {total_sales:.2f},
        total profit was {total_profit:.2f}, total quantity sold was {total_quantity}, average discount
        was {avg_discount:.2f}, and overall profit margin was {profit_margin:.2f}%.
        """,
        {"type": "overall_summary"},
    )

    yearly = (
        prepared.groupby("Year")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum"),
            Discount=("Discount", "mean"),
        )
        .reset_index()
    )
    for _, row in yearly.iterrows():
        margin = (row["Profit"] / row["Sales"]) * 100 if row["Sales"] else 0.0
        add_doc(
            f"""
            In {int(row['Year'])}, total sales were {row['Sales']:.2f}, total profit was
            {row['Profit']:.2f}, total quantity sold was {int(row['Quantity'])}, average discount
            was {row['Discount']:.2f}, and profit margin was {margin:.2f}%.
            """,
            {"type": "yearly_summary", "year": int(row["Year"])},
        )

    monthly = (
        prepared.groupby("YearMonth")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum"),
            Discount=("Discount", "mean"),
        )
        .reset_index()
    )
    for _, row in monthly.iterrows():
        margin = (row["Profit"] / row["Sales"]) * 100 if row["Sales"] else 0.0
        add_doc(
            f"""
            In {row['YearMonth']}, total sales were {row['Sales']:.2f}, total profit was
            {row['Profit']:.2f}, total quantity sold was {int(row['Quantity'])}, average discount
            was {row['Discount']:.2f}, and profit margin was {margin:.2f}%.
            """,
            {"type": "monthly_summary", "year_month": row["YearMonth"]},
        )

    for dimension in ("Category", "Sub-Category", "Region"):
        grouped = (
            prepared.groupby(dimension)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
                Quantity=("Quantity", "sum"),
                Discount=("Discount", "mean"),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            margin = (row["Profit"] / row["Sales"]) * 100 if row["Sales"] else 0.0
            value = row[dimension]
            key = dimension.lower().replace("-", "_")
            add_doc(
                f"""
                The {value} {dimension.lower()} generated total sales of {row['Sales']:.2f},
                total profit of {row['Profit']:.2f}, total quantity sold of {int(row['Quantity'])},
                average discount of {row['Discount']:.2f}, and profit margin of {margin:.2f}%.
                """,
                {"type": f"{key}_summary", key: value},
            )

    for dimension in ("Category", "Region"):
        grouped = (
            prepared.groupby(["Year", dimension])
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
                Quantity=("Quantity", "sum"),
                Discount=("Discount", "mean"),
            )
            .reset_index()
        )
        for _, row in grouped.iterrows():
            margin = (row["Profit"] / row["Sales"]) * 100 if row["Sales"] else 0.0
            value = row[dimension]
            key = dimension.lower()
            add_doc(
                f"""
                In {int(row['Year'])}, the {value} {dimension.lower()} had sales of {row['Sales']:.2f},
                profit of {row['Profit']:.2f}, quantity sold of {int(row['Quantity'])}, average discount
                of {row['Discount']:.2f}, and profit margin of {margin:.2f}%.
                """,
                {"type": f"{key}_year_summary", "year": int(row["Year"]), key: value},
            )

    top_states = (
        prepared.groupby("State")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum"))
        .reset_index()
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    for _, row in top_states.iterrows():
        add_doc(
            f"""
            The state of {row['State']} is one of the top states by sales. Total sales were
            {row['Sales']:.2f}, total profit was {row['Profit']:.2f}, and total quantity sold
            was {int(row['Quantity'])}.
            """,
            {"type": "top_state_summary", "state": row["State"]},
        )

    top_cities = (
        prepared.groupby("City")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum"))
        .reset_index()
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    for _, row in top_cities.iterrows():
        add_doc(
            f"""
            The city of {row['City']} is one of the top cities by sales. Total sales were
            {row['Sales']:.2f}, total profit was {row['Profit']:.2f}, and total quantity sold
            was {int(row['Quantity'])}.
            """,
            {"type": "top_city_summary", "city": row["City"]},
        )

    category = (
        prepared.groupby("Category").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    )
    region = (
        prepared.groupby("Region").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    )
    sub_category = prepared.groupby("Sub-Category").agg(Profit=("Profit", "sum")).reset_index()

    top_category_sales = category.sort_values("Sales", ascending=False).iloc[0]
    top_category_profit = category.sort_values("Profit", ascending=False).iloc[0]
    top_region_sales = region.sort_values("Sales", ascending=False).iloc[0]
    top_region_profit = region.sort_values("Profit", ascending=False).iloc[0]
    top_subcat_profit = sub_category.sort_values("Profit", ascending=False).iloc[0]

    add_doc(
        f"""
        The category with the highest total sales was {top_category_sales['Category']} with sales of
        {top_category_sales['Sales']:.2f}. The category with the highest total profit was
        {top_category_profit['Category']} with profit of {top_category_profit['Profit']:.2f}.
        """,
        {"type": "top_performer_summary", "dimension": "category"},
    )
    add_doc(
        f"""
        The region with the highest total sales was {top_region_sales['Region']} with sales of
        {top_region_sales['Sales']:.2f}. The region with the highest total profit was
        {top_region_profit['Region']} with profit of {top_region_profit['Profit']:.2f}.
        """,
        {"type": "top_performer_summary", "dimension": "region"},
    )
    add_doc(
        f"""
        The sub-category with the highest total profit was {top_subcat_profit['Sub-Category']}
        with profit of {top_subcat_profit['Profit']:.2f}.
        """,
        {"type": "top_performer_summary", "dimension": "sub_category"},
    )

    high_discount = prepared[prepared["Discount"] >= 0.30]
    low_discount = prepared[prepared["Discount"] < 0.30]
    add_doc(
        f"""
        Transactions with high discounts of 30% or more had total sales of
        {high_discount['Sales'].sum():.2f} and total profit of {high_discount['Profit'].sum():.2f}.
        Transactions with discounts below 30% had total sales of {low_discount['Sales'].sum():.2f}
        and total profit of {low_discount['Profit'].sum():.2f}.
        """,
        {"type": "discount_profit_summary"},
    )

    return documents


def build_documents(df: pd.DataFrame) -> list[Document]:
    return create_transaction_documents(df) + create_summary_documents(df)
