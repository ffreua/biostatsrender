from __future__ import annotations
import pandas as pd
from typing import Tuple

DEFAULT_CAT_UNIQUE_RATIO = 0.05

def infer_types(df: pd.DataFrame, unique_ratio: float = DEFAULT_CAT_UNIQUE_RATIO) -> Tuple[list, list]:
    categorical, numeric = [], []
    for col in df.columns:
        s = df[col]
        dt = s.dtype
        
        # Checar o tipo de variável
        if str(dt) in ("object", "category", "bool"):
            categorical.append(col)
        else:
            try:
                # Checar se é numérica
                if pd.api.types.is_numeric_dtype(s):
                    n_unique = s.nunique(dropna=True)
                    ratio = n_unique / max(1, len(s.dropna()))
                    if ratio <= unique_ratio:
                        categorical.append(col)
                    else:
                        numeric.append(col)
                else:
                    categorical.append(col)
            except Exception:
                categorical.append(col)
    
    # Remove duplicatas
    categorical = list(dict.fromkeys(categorical))
    numeric = [c for c in df.columns if c not in categorical]
    return categorical, numeric