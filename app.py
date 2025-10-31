import streamlit as st
import pandas as pd
import numpy as np
import tempfile, os

from biostats.data_loader import load_table
from biostats.typing import infer_types
from biostats.describe import describe_numeric, describe_categorical, quick_overview
from biostats.plots import scatter, histogram, boxplot, violin, bar_count
from biostats.tests import (
    t_test_one_sample, t_test_independent, t_test_paired, anova_oneway,
    mannwhitney, wilcoxon_signed, kruskal, friedman,
    chi2_or_fisher, 
    correlation, linear_regression, logistic_regression
)
from biostats.utils import na_summary, coerce_numeric, plotly_template
from biostats.pdf_report import save_plotly_figure, build_pdf

def display_test_result(test_name, result):
    st.markdown(f"### Resultados para: **{test_name}**")
    
    if not result:
        st.warning("Nenhum resultado para exibir.")
        return

    # Formatar valor-p
    p_value = result.get('p')
    if p_value is not None:
        p_formatted = f"{p_value:.4f}"
        if p_value < 0.0001:
            p_formatted = "< 0.0001"
        
        cols = st.columns(3)
        cols[0].metric("Valor-p", p_formatted)
        
        alpha = 0.05
        if p_value < alpha:
            cols[1].markdown(f"**Conclusão:** <span style='color: #28a745;'>Significativo (p < {alpha})</span>", unsafe_allow_html=True)
        else:
            cols[1].markdown(f"**Conclusão:** <span style='color: #dc3545;'>Não significativo (p ≥ {alpha})</span>", unsafe_allow_html=True)

    # Exibir outras estatísticas
    other_stats = {k: v for k, v in result.items() if k not in ['p', 'tabela', 'summary', 'grupos', 'encoding']}
    
    # Exibir métricas principais
    main_metrics = {}
    if "t" in other_stats: main_metrics["Estatística t"] = other_stats.pop("t")
    if "F" in other_stats: main_metrics["Estatística F"] = other_stats.pop("F")
    if "U" in other_stats: main_metrics["Estatística U"] = other_stats.pop("U")
    if "W" in other_stats: main_metrics["Estatística W"] = other_stats.pop("W")
    if "H" in other_stats: main_metrics["Estatística H"] = other_stats.pop("H")
    if "Q" in other_stats: main_metrics["Estatística Q"] = other_stats.pop("Q")
    if "chi2" in other_stats: main_metrics["Qui-quadrado"] = other_stats.pop("chi2")
    if "r" in other_stats: main_metrics[f"Correlação ({result.get('metodo', '')})"] = other_stats.pop("r")
    if "odds_ratio" in other_stats: main_metrics["Odds Ratio"] = other_stats.pop("odds_ratio")
    
    if main_metrics:
        num_metrics = len(main_metrics)
        cols = st.columns(num_metrics)
        for i, (label, value) in enumerate(main_metrics.items()):
            cols[i].metric(label, f"{value:.4f}")

    # Exibir o resto em um expander
    if other_stats:
        with st.expander("Ver todas as estatísticas"):
            st.json(other_stats)
            
    # Exibir sumários de regressão
    if "summary" in result:
        st.text(result["summary"])
    
    # Exibir tabela de contingência
    if "tabela" in result:
        st.write("Tabela de Contingência:")
        st.dataframe(result["tabela"])


st.set_page_config(page_title="BioStats Render v2.1", page_icon="📊", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📊 BioStats Render")
    st.info("WebApp para análise de dados bioestatísticos")
    fullscreen = st.checkbox("🖥️ Tela cheia (esconder sidebar)")
    st.markdown("---")
    st.info("Desenvolvido por **Dr Fernando Freua**")

# Fullscreen
if fullscreen:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"]{ display:none !important; }
    div[data-testid="stToolbar"]{ display:none !important; }
    </style>
    """, unsafe_allow_html=True)

PLOTLY_TEMPLATE = plotly_template()

# ---------- Session State ----------
defaults = {
    "df": None,
    "categorical": [],
    "numeric": [],
    "last_plot_fig": None,
    "last_test_result": None,
    "last_test_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Tabs ----------
tabs = st.tabs(["⬆️ Importar Dados", "🧩 Explorar & Visualizar", "📈 Testes Bioestatísticos", "🧾 Relatório (PDF)"])

# ============ TAB 1: Importar Dados ============
with tabs[0]:
    st.subheader("⬆️ Importar dados (CSV/Excel)")
    st.info("Dica: use cabeçalhos claros nas colunas. Após importar, confirme as variáveis **categóricas**.")
    up = st.file_uploader("Selecione um arquivo", type=["csv","xlsx","xls"], key="uploader")

    if up is not None:
        try:
            df = load_table(up)
            for c in df.columns:
                df[c] = coerce_numeric(df[c])
            st.session_state.df = df
            st.success(f"Arquivo carregado: **{up.name}** — {df.shape[0]} linhas × {df.shape[1]} colunas.")
            st.dataframe(df.head(30), use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")

    if st.session_state.df is not None:
        st.markdown("---")
        st.subheader("Definição de tipos")
        auto_cat, auto_num = infer_types(st.session_state.df)
        chosen_cats = st.multiselect(
            "Confirme as **categóricas** (as demais serão tratadas como numéricas):",
            options=list(st.session_state.df.columns),
            default=auto_cat,
            help="Ajuste se necessário. Essa escolha impacta os gráficos e testes disponíveis."
        )
        st.session_state.categorical = chosen_cats
        st.session_state.numeric = [c for c in st.session_state.df.columns if c not in chosen_cats]
        st.success("Tipos confirmados. Você pode seguir para as próximas abas sem reimportar.")

# ============ TAB 2: Explorar & Visualizar ============
with tabs[1]:
    st.subheader("🧩 Explorar & Visualizar")
    if st.session_state.df is None:
        st.warning("Importe um arquivo na aba **Importar Dados**.")
    else:
        df = st.session_state.df
        cats = st.session_state.categorical
        nums = [c for c in st.session_state.numeric if pd.api.types.is_numeric_dtype(df[c])]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Linhas", df.shape[0])
        c2.metric("Colunas", df.shape[1])
        c3.metric("Categóricas", len(cats))
        c4.metric("Numéricas", len(nums))

        st.markdown("### Estatísticas descritivas")
        desc_num = describe_numeric(df, nums)
        st.dataframe(desc_num, use_container_width=True)
        st.markdown("#### Frequências (categóricas)")
        if cats:
            tnames = [f"{c}" for c in cats]
            tabs_cats = st.tabs(tnames)
            for tab, col in zip(tabs_cats, cats):
                with tab:
                    # Obter a tabela de frequência
                    freq_df = describe_categorical(df, [col])[col]
                    # Resetar o índice para transformá-lo em uma coluna e renomear para clareza
                    freq_df = freq_df.reset_index()
                    freq_df = freq_df.rename(columns={'index': col})
                    st.dataframe(freq_df, use_container_width=True)
        else:
            st.info("Nenhuma categórica definida. Ajuste na aba **Importar Dados**.")

        st.markdown("### Valores faltantes")
        st.dataframe(na_summary(df), use_container_width=True)

        st.markdown("---")
        st.markdown("### Gráficos")
        gcol1, gcol2 = st.columns(2)
        graph = gcol1.selectbox("Tipo de gráfico", ["Dispersão", "Histograma", "Boxplot", "Violino", "Barras (contagem)"])
        color_opt = gcol2.selectbox("Cor (opcional)", options=["(nenhum)"] + list(df.columns))

        fig = None
        if graph == "Dispersão":
            x = st.selectbox("Eixo X (numérico)", options=nums, key="g_sc_x")
            y = st.selectbox("Eixo Y (numérico)", options=[c for c in nums if c != x], key="g_sc_y")
            fig = scatter(df, x, y, None if color_opt=="(nenhum)" else color_opt, template=PLOTLY_TEMPLATE)

        elif graph == "Histograma":
            x = st.selectbox("Variável (numérica)", options=nums, key="g_hist_x")
            bins = st.slider("Bins", 5, 100, 30, key="g_hist_bins")
            fig = histogram(df, x, None if color_opt=="(nenhum)" else color_opt, nbins=bins, template=PLOTLY_TEMPLATE)

        elif graph == "Boxplot":
            y = st.selectbox("Y (numérico)", options=nums, key="g_box_y")
            x = st.selectbox("X (categórico, opcional)", options=["(nenhum)"] + cats, key="g_box_x")
            fig = boxplot(df, None if x=="(nenhum)" else x, y, None if color_opt=="(nenhum)" else color_opt, template=PLOTLY_TEMPLATE)

        elif graph == "Violino":
            y = st.selectbox("Y (numérico)", options=nums, key="g_vio_y")
            x = st.selectbox("X (categórico, opcional)", options=["(nenhum)"] + cats, key="g_vio_x")
            fig = violin(df, None if x=="(nenhum)" else x, y, None if color_opt=="(nenhum)" else color_opt, template=PLOTLY_TEMPLATE)

        elif graph == "Barras (contagem)":
            x = st.selectbox("Variável categórica", options=cats, key="g_bar_x")
            fig = bar_count(df, x, None if color_opt=="(nenhum)" else color_opt, template=PLOTLY_TEMPLATE)

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.last_plot_fig = fig  # save for PDF

# ============ TAB 3: Testes Bioestatísticos ============
with tabs[2]:
    st.subheader("📈 Testes Bioestatísticos")
    if st.session_state.df is None:
        st.warning("Importe um arquivo na aba **Importar Dados**.")
    else:
        df = st.session_state.df
        cats = st.session_state.categorical
        nums = [c for c in st.session_state.numeric if pd.api.types.is_numeric_dtype(df[c])]

        test = st.selectbox("Escolha o teste", [
            "t (1 amostra)",
            "t (2 amostras independentes)",
            "t pareado",
            "ANOVA one-way",
            "Mann–Whitney",
            "Wilcoxon (pareado)",
            "Kruskal–Wallis",
            "Friedman (repetidas)",
            "Qui-quadrado / Fisher",
            "Correlação (Pearson)",
            "Correlação (Spearman)",
            "Regressão Linear (Y~X)",
            "Regressão Logística (Y binário ~ X)",
        ])

        st.markdown("> **Nota:** verifique suposições (normalidade, homocedasticidade, independência) quando aplicável.")

        res = None
        if test == "t (1 amostra)":
            if not nums:
                st.warning("Este teste requer pelo menos uma variável numérica.")
            else:
                col = st.selectbox("Coluna (numérica)", options=nums)
                mu = st.number_input("Média hipotética (μ₀)", value=0.0, step=0.1)
                if st.button("Rodar teste", type="primary"):
                    res = t_test_one_sample(df, col, mu)
                    display_test_result(test, res)
        elif test == "t (2 amostras independentes)":
            if not nums or not cats:
                st.warning("Este teste requer uma variável numérica e uma categórica (com 2 níveis).")
            else:
                y = st.selectbox("Y (numérico)", options=nums)
                group = st.selectbox("Grupo (2 níveis)", options=cats)
                if st.button("Rodar teste", type="primary"):
                    try:
                        res = t_test_independent(df, y, group)
                        display_test_result(test, res)
                    except Exception as e:
                        st.error(str(e))
        elif test == "t pareado":
            if len(nums) < 2:
                st.warning("Este teste requer pelo menos duas variáveis numéricas.")
            else:
                pre = st.selectbox("Pré", options=nums)
                pos = st.selectbox("Pós", options=[c for c in nums if c != pre])
                if st.button("Rodar teste", type="primary"):
                    res = t_test_paired(df, pre, pos)
                    display_test_result(test, res)
        elif test == "ANOVA one-way":
            if not nums or not cats:
                st.warning("Este teste requer uma variável numérica e uma categórica.")
            else:
                y = st.selectbox("Y (numérico)", options=nums)
                group = st.selectbox("Grupo (≥2 níveis)", options=cats)
                if st.button("Rodar teste", type="primary"):
                    res = anova_oneway(df, y, group)
                    display_test_result(test, res)
        elif test == "Mann–Whitney":
            if not nums or not cats:
                st.warning("Este teste requer uma variável numérica/ordinal e uma categórica (com 2 níveis).")
            else:
                y = st.selectbox("Y (ordinal/contínua)", options=nums)
                group = st.selectbox("Grupo (2 níveis)", options=cats)
                if st.button("Rodar teste", type="primary"):
                    try:
                        res = mannwhitney(df, y, group)
                        display_test_result(test, res)
                    except Exception as e:
                        st.error(str(e))
        elif test == "Wilcoxon (pareado)":
            if len(nums) < 2:
                st.warning("Este teste requer pelo menos duas variáveis numéricas.")
            else:
                pre = st.selectbox("Pré", options=nums, key="w_pre")
                pos = st.selectbox("Pós", options=[c for c in nums if c != pre], key="w_pos")
                if st.button("Rodar teste", type="primary"):
                    res = wilcoxon_signed(df, pre, pos)
                    display_test_result(test, res)
        elif test == "Kruskal–Wallis":
            if not nums or not cats:
                st.warning("Este teste requer uma variável numérica/ordinal e uma categórica.")
            else:
                y = st.selectbox("Y (ordinal/contínua)", options=nums)
                group = st.selectbox("Grupo (≥2 níveis)", options=cats)
                if st.button("Rodar teste", type="primary"):
                    res = kruskal(df, y, group)
                    display_test_result(test, res)
        elif test == "Friedman (repetidas)":
            if len(nums) < 3:
                st.warning("Este teste requer pelo menos três variáveis numéricas.")
            else:
                cols = st.multiselect("Selecione 3+ colunas (mesmos sujeitos)", options=nums)
                if st.button("Rodar teste", type="primary"):
                    if len(cols) < 3:
                        st.error("Selecione ao menos 3 colunas para Friedman.")
                    else:
                        res = friedman(df, cols)
                        display_test_result(test, res)
        elif test == "Qui-quadrado / Fisher":
            if len(cats) < 2:
                st.warning("Este teste requer pelo menos duas variáveis categóricas.")
            else:
                x = st.selectbox("Categórica 1", options=cats)
                y = st.selectbox("Categórica 2", options=[c for c in cats if c != x])
                if st.button("Rodar teste", type="primary"):
                    res = chi2_or_fisher(df, x, y)
                    display_test_result(test, res)
        elif test == "Correlação (Pearson)":
            if len(nums) < 2:
                st.warning("Este teste requer pelo menos duas variáveis numéricas.")
            else:
                x = st.selectbox("X (numérico)", options=nums, key="c_p_x")
                y = st.selectbox("Y (numérico)", options=[c for c in nums if c != x], key="c_p_y")
                if st.button("Calcular correlação", type="primary"):
                    res = correlation(df, x, y, method="pearson")
                    display_test_result(test, res)
        elif test == "Correlação (Spearman)":
            if len(nums) < 2:
                st.warning("Este teste requer pelo menos duas variáveis numéricas/ordinais.")
            else:
                x = st.selectbox("X (numérico/ordinal)", options=nums, key="c_s_x")
                y = st.selectbox("Y (numérico/ordinal)", options=[c for c in nums if c != x], key="c_s_y")
                if st.button("Calcular correlação", type="primary"):
                    res = correlation(df, x, y, method="spearman")
                    display_test_result(test, res)
        elif test == "Regressão Linear (Y~X)":
            if len(nums) < 2:
                st.warning("Este teste requer pelo menos duas variáveis numéricas.")
            else:
                y = st.selectbox("Y (numérico)", options=nums, key="rl_y")
                x = st.selectbox("X (numérico)", options=[c for c in nums if c != y], key="rl_x")
                if st.button("Rodar regressão", type="primary"):
                    res = linear_regression(df, y, x)
                    display_test_result(test, res)
        elif test == "Regressão Logística (Y binário ~ X)":
            if not cats:
                st.warning("Este teste requer pelo menos uma variável categórica para Y.")
            else:
                y = st.selectbox("Y (binário/categórico com 2 níveis)", options=cats, key="logit_y")
                x = st.selectbox("X (numérico/categórico)", options=[c for c in df.columns if c != y], key="logit_x")
                if st.button("Rodar regressão", type="primary"):
                    try:
                        res = logistic_regression(df, y, x)
                        display_test_result(test, res)
                    except Exception as e:
                        st.error(str(e))

        if res is not None:
            st.session_state.last_test_result = res
            st.session_state.last_test_name = test

# ============ TAB 4: PDF ============
with tabs[3]:
    st.subheader("🧾 Relatório (PDF)")
    if st.session_state.df is None:
        st.warning("Importe um arquivo na aba **Importar Dados** e gere pelo menos um gráfico ou estatística.")
    else:
        df = st.session_state.df
        cats = st.session_state.categorical
        nums = [c for c in st.session_state.numeric if pd.api.types.is_numeric_dtype(df[c])]

        st.write("Este relatório inclui:")
        st.markdown("- Sumário geral (linhas, colunas, #cat, #num)\n- Estatísticas descritivas numéricas\n- Tabela de faltantes\n- **Último gráfico** gerado\n- **Último teste** executado (com resultados)")

        # Preparar dados para PDF
        desc = describe_numeric(df, nums)
        na = na_summary(df)

        # Selecionar último gráfico
        fig = st.session_state.last_plot_fig

        # Botão gerar PDF
        if st.button("Gerar PDF", type="primary"):
            with tempfile.TemporaryDirectory() as td:
                plot_path = None
                if fig is not None:
                    plot_path = os.path.join(td, "grafico.png")
                    try:
                        save_plotly_figure(fig, plot_path)
                    except Exception as e:
                        plot_path = None
                        st.warning(f"Não foi possível exportar o gráfico (kaleido ausente?): {e}")

                # Tabelas em CSV (texto) para incorporar
                stats_tables = {
                    "Descritivas numéricas (CSV)": desc.to_csv(index=True),
                    "Valores faltantes (CSV)": na.to_csv(index=False)
                }

                output_pdf = os.path.join(td, "BioStats_Render_relatorio.pdf")
                summary_text = f"""Linhas: {df.shape[0]}
Colunas: {df.shape[1]}
Categóricas: {len(cats)}
Numéricas: {len(nums)}
"""

                build_pdf(
                    output_path=output_pdf,
                    title="BioStats Render — Relatório",
                    summary_text=summary_text,
                    stats_tables=stats_tables,
                    plot_path=plot_path,
                    test_result=st.session_state.last_test_result
                )

                with open(output_pdf, "rb") as f:
                    st.download_button("⬇️ Baixar PDF", f, file_name="BioStats_Render_relatorio.pdf", mime="application/pdf")

st.markdown("---")
st.markdown("**BioStats Render v2.1** — © Desenvolvido por **Dr Fernando Freua**.")