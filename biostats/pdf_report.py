from __future__ import annotations
import os, datetime, json, tempfile
from typing import Optional, Dict, Any
from fpdf import FPDF

def save_plotly_figure(fig, path_png: str):
    # requer kaleido
    try:
        fig.write_image(path_png, scale=2)
    except Exception as e:
        raise RuntimeError(f"Erro ao salvar gráfico: {e}. Verifique se o kaleido está instalado.")

def build_pdf(output_path: str,
              title: str,
              summary_text: str,
              stats_tables: Dict[str, str] | None = None,
              plot_path: Optional[str] = None,
              test_result: Optional[Dict[str, Any]] = None):
    """
    stats_tables: dict{name: csv_string} — cada tabela já em texto CSV para mostrar no PDF.
    """
    try:
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, title, ln=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100,100,100)
        pdf.cell(0, 6, f"Geração: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1)
        pdf.set_text_color(0,0,0)
        pdf.ln(2)
    except Exception as e:
        raise RuntimeError(f"Erro ao inicializar PDF: {e}")

    # Resumo
    try:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Resumo", ln=1)
        pdf.set_font("Helvetica", "", 10)
        for line in summary_text.splitlines():
            pdf.multi_cell(0, 5, line)
        pdf.ln(2)
    except Exception as e:
        pdf.cell(0, 6, f"Erro ao incluir resumo: {e}", ln=1)

    # Gráfico
    if plot_path and os.path.exists(plot_path):
        try:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Gráfico", ln=1)
            pdf.image(plot_path, w=170)
            pdf.ln(4)
        except Exception as e:
            pdf.cell(0, 6, f"Erro ao incluir gráfico: {e}", ln=1)

    # Tabelas
    if stats_tables:
        for name, csv_text in stats_tables.items():
            try:
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, name, ln=1)
                pdf.set_font("Helvetica", "", 8)
                for ln in csv_text.strip().splitlines():
                    pdf.multi_cell(0, 4, ln)
                pdf.ln(2)
            except Exception as e:
                pdf.cell(0, 6, f"Erro ao incluir tabela {name}: {e}", ln=1)

    # Teste
    if test_result:
        try:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Último teste executado", ln=1)
            pdf.set_font("Helvetica", "", 9)
            try:
                pretty = json.dumps(test_result, ensure_ascii=False, indent=2)
            except Exception:
                pretty = str(test_result)
            for ln in pretty.splitlines():
                pdf.multi_cell(0, 4, ln)
            pdf.ln(2)
        except Exception as e:
            pdf.cell(0, 6, f"Erro ao incluir resultado do teste: {e}", ln=1)

    try:
        pdf.output(output_path)
    except Exception as e:
        raise RuntimeError(f"Erro ao gerar PDF: {e}")