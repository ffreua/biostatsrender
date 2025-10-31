from __future__ import annotations
import pandas as pd
import numpy as np
from .utils import safe_mode

def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    
    # Filtrar apenas colunas numéricas
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return pd.DataFrame()
    
    desc = df[numeric_cols].agg(["count","mean","std","min","median","max"])
    q = df[numeric_cols].quantile([0.25, 0.75])
    desc.loc["q1"] = q.loc[0.25]
    desc.loc["q3"] = q.loc[0.75]
    desc = desc.round(4)
    return desc

def describe_categorical(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
    tables = {}
    for c in cols:
        if c in df.columns:
            vc = df[c].astype("category")
            tab = vc.value_counts(dropna=False).to_frame("frequencia")
            tab["percentual_%"] = (tab["frequencia"] / len(df) * 100).round(2)
            tables[c] = tab
    return tables

def quick_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        try:
            row = {
                "coluna": c,
                "dtype": str(s.dtype),
                "n": int(s.shape[0]),
                "n_na": int(s.isna().sum()),
                "n_unique": int(s.nunique(dropna=True)),
                "exemplo": next((str(x) for x in s.dropna().head(3).tolist()), ""),
                "moda": safe_mode(s),
            }
        except Exception as e:
            row = {
                "coluna": c,
                "dtype": str(s.dtype),
                "n": int(s.shape[0]),
                "n_na": int(s.isna().sum()),
                "n_unique": 0,
                "exemplo": "",
                "moda": None,
            }
        rows.append(row)
    out = pd.DataFrame(rows)
    return out