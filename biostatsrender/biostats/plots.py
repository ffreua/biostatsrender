from __future__ import annotations
import pandas as pd
import plotly.express as px

def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, template: str = "plotly"):
    return px.scatter(df, x=x, y=y, color=color, title=f"Dispersão: {x} vs {y}", template=template)

def histogram(df: pd.DataFrame, x: str, color: str | None = None, nbins: int = 30, template: str = "plotly"):
    return px.histogram(df, x=x, color=color, nbins=nbins, title=f"Histograma: {x}", template=template)

def boxplot(df: pd.DataFrame, x: str | None, y: str, color: str | None = None, template: str = "plotly"):
    return px.box(df, x=x, y=y, color=color, title=f"Boxplot: {y}" + (f" por {x}" if x else ""), template=template)

def violin(df: pd.DataFrame, x: str | None, y: str, color: str | None = None, points: str = "outliers", template: str = "plotly"):
    return px.violin(df, x=x, y=y, color=color, box=True, points=points, title=f"Violin: {y}" + (f" por {x}" if x else ""), template=template)

def bar_count(df: pd.DataFrame, x: str, color: str | None = None, template: str = "plotly"):
    if color == x:
        color = None
    if color is None:
        counts = df[x].value_counts().reset_index()
        counts.columns = [x, 'count']
        return px.bar(counts, x=x, y='count', title=f"Contagem: {x}", template=template)
    else:
        counts = df.groupby([x, color]).size().to_frame('_temp_agg_count_').reset_index()
        counts = counts.rename(columns={'_temp_agg_count_': 'count'})
        return px.bar(counts, x=x, y='count', color=color, title=f"Contagem: {x} por {color}", template=template)