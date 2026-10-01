# AI Business Intelligence Copilot (TEC 406 Capstone)

Ask a business question in plain English, e.g. **"Why did sales decline in the South region?"**.
The copilot analyses the data, retrieves company context, explains the trend, suggests actions and shows a dashboard.

## Run it (3 commands)
```bash
pip install -r requirements.txt
streamlit run app.py
# optional, for the LLM-assisted mode:  set ANTHROPIC_API_KEY or paste the key in the sidebar
```
No API key? It still works: a built-in template writer produces the explanation (offline mode).

## How it works
1. **Understand** - LLM router (optional) or rule-based parser -> intent, metric, filters, period (output is validated).
2. **Analyse** - pandas tools compute every number: KPIs, driver breakdown (category/city/channel), price vs volume effect, hotspots.
3. **Retrieve (RAG)** - TF-IDF search over `knowledge/business_events.md` and `knowledge/metric_glossary.md`.
4. **Explain** - LLM (optional) or template writes the answer using ONLY the verified facts.
5. **Audit** - every money/percent/unit figure in the answer is checked against the verified facts; unverified figures are flagged.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit prototype |
| `copilot/engine.py` | Core engine (parsing, analysis tools, retrieval, narration, audit) |
| `generate_data.py` / `sales_data.csv` | Synthetic dataset (3,360 rows, Jan 2025 - Sep 2026) with a planted South-region decline |
| `knowledge/` | Business events log + metric glossary (the RAG knowledge base) |
| `evaluate.py` | 21-question evaluation + guardrail tests -> `eval_results.csv`, `eval_summary.csv` |
| `report_figures.py` | Generates the figures used in the report |

## Evaluate
```bash
python evaluate.py          # deterministic offline run; set ANTHROPIC_API_KEY to evaluate the LLM mode too
```
