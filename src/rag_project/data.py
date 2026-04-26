from __future__ import annotations

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1")


def describe_data(df: pd.DataFrame) -> str:
    lines = [
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "Available columns:",
    ]
    lines.extend(f"- {column}" for column in df.columns)
    return "\n".join(lines)
