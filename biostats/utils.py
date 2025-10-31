from __future__ import annotations
import numpy as np
import pandas as pd

def safe_mode(series: pd.Series):
    try:
        m = series.mode(dropna=True)
        if len(m) == 0:
            return None
        return m.iloc[0]
    except Exception:
        return None

def na_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "coluna": df.columns.tolist(),
        "faltantes": df.isna().sum().values,
        "percentual_%": (df.isna().mean().values * 100).round(2),
    })
    return out

def coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == "O":
        try:
            # Tentar converter para numérico, lidando com problemas comuns
            ser = pd.to_numeric(series.replace(",", ".", regex=False), errors="coerce")
            # Converter apenas se pelo menos 70% dos valores forem convertidos com sucesso
            if ser.notna().mean() > 0.7:
                return ser
        except Exception:
            pass
    return series

def plotly_template(theme_name: str | None = None) -> str:
    return "plotly_white"