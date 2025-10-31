# BioStats Render v2.1

## Rodando
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
Verifique Plotly Chrome instalado: $ plotly_get_chrome (no terminal)

## Observações
- Para exportar gráficos ao PDF, usamos o `kaleido` (instalado via requirements).
- O PDF é gerado a partir do conteúdo em memória (estatísticas + último gráfico gerado + último teste realizado).
