from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import proportion as prop

def cohens_d(x, y):
    x = pd.Series(x).dropna().values
    y = pd.Series(y).dropna().values
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled = np.sqrt(((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / (nx+ny-2))
    if pooled == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled

def odds_ratio_2x2(a, b, c, d):
    or_ = (a*d) / (b*c) if (b*c) != 0 else np.inf
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_low = np.exp(np.log(or_) - 1.96*se)
    ci_high = np.exp(np.log(or_) + 1.96*se)
    return or_, (ci_low, ci_high)

def t_test_one_sample(df: pd.DataFrame, col: str, mu: float) -> Dict[str, Any]:
    t, p = stats.ttest_1samp(pd.to_numeric(df[col], errors="coerce").dropna(), mu)
    return {"t": float(t), "p": float(p), "mu0": float(mu)}

def t_test_independent(df: pd.DataFrame, y: str, group: str) -> Dict[str, Any]:
    levels = [lv for lv in df[group].dropna().unique() if lv == lv]
    if len(levels) != 2:
        raise ValueError("A variável de grupo deve ter exatamente 2 níveis para t independente.")
    g1, g2 = levels
    x1 = df.loc[df[group] == g1, y]
    x2 = df.loc[df[group] == g2, y]
    t, p = stats.ttest_ind(x1, x2, nan_policy="omit", equal_var=False)
    d = cohens_d(x1, x2)
    return {"t": float(t), "p": float(p), "cohens_d": float(d), "grupos": [str(g1), str(g2)]}

def t_test_paired(df: pd.DataFrame, col_pre: str, col_pos: str) -> Dict[str, Any]:
    t, p = stats.ttest_rel(df[col_pre], df[col_pos], nan_policy="omit")
    return {"t": float(t), "p": float(p)}

def anova_oneway(df: pd.DataFrame, y: str, group: str) -> Dict[str, Any]:
    groups = [grp.dropna().values for _, grp in df.groupby(group)[y]]
    f, p = stats.f_oneway(*groups)
    model = smf.ols(f"{y} ~ C({group})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    ss_between = anova_table.loc[f"C({group})", "sum_sq"]
    ss_total = ss_between + anova_table.loc["Residual", "sum_sq"]
    eta2 = ss_between / ss_total if ss_total != 0 else np.nan
    return {"F": float(f), "p": float(p), "eta2": float(eta2)}

def mannwhitney(df: pd.DataFrame, y: str, group: str) -> Dict[str, Any]:
    levels = [lv for lv in df[group].dropna().unique() if lv == lv]
    if len(levels) != 2:
        raise ValueError("A variável de grupo deve ter exatamente 2 níveis para Mann–Whitney.")
    g1, g2 = levels
    x1 = df.loc[df[group] == g1, y]
    x2 = df.loc[df[group] == g2, y]
    u, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    return {"U": float(u), "p": float(p), "grupos": [str(g1), str(g2)]}

def wilcoxon_signed(df: pd.DataFrame, col_pre: str, col_pos: str) -> Dict[str, Any]:
    w, p = stats.wilcoxon(df[col_pre], df[col_pos], zero_method="wilcox", alternative="two-sided", nan_policy="omit")
    return {"W": float(w), "p": float(p)}

def kruskal(df: pd.DataFrame, y: str, group: str) -> Dict[str, Any]:
    groups = [grp.dropna().values for _, grp in df.groupby(group)[y]]
    h, p = stats.kruskal(*groups)
    return {"H": float(h), "p": float(p)}

def friedman(df: pd.DataFrame, cols: list[str]) -> Dict[str, Any]:
    arrays = [pd.to_numeric(df[c], errors="coerce").dropna() for c in cols]
    min_len = min(map(len, arrays)) if arrays else 0
    arrays = [a.iloc[:min_len] for a in arrays]
    stat, p = stats.friedmanchisquare(*arrays)
    return {"Q": float(stat), "p": float(p)}

def chi2_or_fisher(df: pd.DataFrame, x: str, y: str) -> Dict[str, Any]:
    tab = pd.crosstab(df[x], df[y])
    if tab.shape == (2,2) and (tab.values < 5).any():
        from scipy.stats import fisher_exact
        (a,b),(c,d) = tab.values
        or_, p = fisher_exact([[a,b],[c,d]])
        or_ci = odds_ratio_2x2(a,b,c,d)[1]
        return {"teste": "fisher", "p": float(p), "odds_ratio": float(or_), "or_95ci": [float(or_ci[0]), float(or_ci[1])], "tabela": tab}
    else:
        chi2, p, dof, expected = stats.chi2_contingency(tab)
        return {"teste": "chi2", "chi2": float(chi2), "p": float(p), "graus_lib": int(dof), "esperada_min": float(expected.min()), "tabela": tab}

def one_proportion(df: pd.DataFrame, col: str, success, prop0: float):
    x = (df[col] == success).sum()
    n = df[col].notna().sum()
    stat, p = prop.proportions_ztest(count=x, nobs=n, value=prop0)
    ci_low, ci_high = prop.proportion_confint(count=x, nobs=n, method="wilson")
    return {"success": str(success), "x": int(x), "n": int(n),
            "p_hat": float(x/n), "z": float(stat), "p": float(p),
            "ci_95": [float(ci_low), float(ci_high)]}


def two_proportions(df: pd.DataFrame, col: str, group: str, success, g1, g2):
    sub = df[[col, group]].dropna()
    sub = sub[sub[group].isin([g1, g2])]
    x1 = (sub[sub[group]==g1][col] == success).sum()
    n1 = sub[sub[group]==g1][col].shape[0]
    x2 = (sub[sub[group]==g2][col] == success).sum()
    n2 = sub[sub[group]==g2][col].shape[0]
    stat, p = prop.proportions_ztest(count=[x1, x2], nobs=[n1, n2])
    (l1, u1) = prop.proportion_confint(x1, n1, method="wilson")
    (l2, u2) = prop.proportion_confint(x2, n2, method="wilson")
    return {"grupo1": str(g1), "grupo2": str(g2),
            "x1": int(x1), "n1": int(n1),
            "x2": int(x2), "n2": int(n2),
            "z": float(stat), "p": float(p),
            "ci95_g1": [float(l1), float(u1)],
            "ci95_g2": [float(l2), float(u2)]}


def correlation(df: pd.DataFrame, x: str, y: str, method: str = "pearson"):
    s1 = pd.to_numeric(df[x], errors="coerce")
    s2 = pd.to_numeric(df[y], errors="coerce")
    if method == "pearson":
        r, p = stats.pearsonr(s1.dropna(), s2.dropna())
    else:
        r, p = stats.spearmanr(s1, s2, nan_policy="omit")
    return {"metodo": method, "r": float(r), "p": float(p)}


def linear_regression(df: pd.DataFrame, y: str, x: str):
    # Handle categorical variables by creating dummy variables
    if df[x].dtype == 'object' or df[x].dtype.name == 'category':
        # For categorical X, use C() notation
        formula = f"{y} ~ C({x})"
    else:
        # For numeric X, use as is
        formula = f"{y} ~ {x}"
    
    model = smf.ols(formula, data=df).fit()
    return {"summary": str(model.summary()), "params": model.params.to_dict(), "r2": float(model.rsquared)}


def logistic_regression(df: pd.DataFrame, y: str, x: str):
    y_series = df[y].dropna()
    levels = y_series.unique()
    if len(levels) != 2:
        raise ValueError("A variável resposta deve ser binária para regressão logística.")
    mapping = {levels[0]: 0, levels[1]: 1}
    df2 = df[[y, x]].dropna().copy()
    df2[y] = df2[y].map(mapping)
    
    # Lidar com variáveis categóricas criando variáveis dummy
    if df2[x].dtype == 'object' or df2[x].dtype.name == 'category':
        # For categorical X, use C() notation
        formula = f"{y} ~ C({x})"
    else:
        # For numeric X, use as is
        formula = f"{y} ~ {x}"
    
    model = smf.logit(formula, data=df2).fit(disp=False)
    return {"summary": str(model.summary()), "params": model.params.to_dict(),
            "pseudo_r2": float(model.prsquared), "encoding": mapping}
