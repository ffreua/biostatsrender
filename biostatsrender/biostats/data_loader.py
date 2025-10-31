from __future__ import annotations
import pandas as pd

def load_table(uploaded_file) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    
    # Tentar CSV primeiro, depois Excel
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        try:
            return pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"Erro ao ler arquivo: {e}")