"""AI Business Intelligence Copilot - Streamlit prototype.   Run:  streamlit run app.py"""
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from copilot.engine import Copilot, DEFAULT_MODEL, fmt_metric, public_facts

st.set_page_config(page_title="AI BI Copilot", page_icon="📊", layout="wide")

EXAMPLES = [
    "Why did sales decline in the South region?",
    "Which region declined the most?",
    "Why did profit fall in Bengaluru?",
    "Show the monthly sales trend for the West region",
    "Compare regions by sales",
    "Top categories in the North region",
    "What is net price per unit?",
    "Forecast sales for next quarter",
]


@st.cache_resource
def get_copilot(api_key: str, model: str) -> Copilot:
    return Copilot(api_key=api_key or None, model=model)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    key_default = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.text_input("Anthropic API key (optional)", value=key_default, type="password",
                            help="Leave blank to run in offline template mode. With a key, Claude routes the "
                                 "question and writes the explanation. All numbers are still computed by Python.")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.caption("Mode: **LLM-assisted**" if api_key else "Mode: **Offline template** (no API key)")
    st.divider()
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state["question"] = ex
    st.divider()
    st.caption("Data is synthetic (Jan 2025 - Sep 2026). Company events are fictional.")

copilot = get_copilot(api_key, model)

# ------------------------------------------------------------------ header
st.title("📊 AI Business Intelligence Copilot")
st.caption("Ask a business question in plain English. Python computes the numbers, retrieval adds company context, "
           "and every figure in the answer is audited against the data.")

question = st.text_input("Ask a question", key="question",
                         placeholder="e.g. Why did sales decline in the South region?")

if not question:
    st.info("Pick an example from the sidebar or type your own question.")
    st.stop()

with st.spinner("Analysing..."):
    ans = copilot.ask(question)

q, f = ans.query, ans.facts

# ------------------------------------------------------------------ answer
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Answer")
    st.markdown(ans.text)
    if ans.audit.get("passed"):
        st.success(f"Number audit passed: all {ans.audit['figures_found']} figures match the verified data.")
    else:
        st.warning(f"Number audit: {len(ans.audit['unverified'])} figure(s) could not be verified against the data: "
                   f"{ans.audit['unverified']}. Treat the narrative with caution.")
    st.caption(f"Interpreted as: intent = **{q.intent}**, metric = **{q.metric}**, "
               f"scope = **{f.get('scope', '-')}** | router: {ans.router} | writer: {ans.mode}")

with right:
    st.subheader("Dashboard")
    if "kpi" in f:
        k = f["kpi"]
        c1, c2 = st.columns(2)
        c1.metric(f"{q.metric.title()} ({f['period']['current']})", k["current"], k["change_pct"])
        c2.metric("Avg discount", k["discount_current"], delta=None)
        st.caption(f"vs {f['period']['previous']}: {k['previous']} | margin {k['margin_previous']} -> {k['margin_current']}")

    T = ans.tables
    m = q.metric

    def bar(df, dim, title):
        d = df.sort_values("change")
        fig = go.Figure(go.Bar(x=d["change"] / 1e5 if m != "units" else d["change"], y=d[dim], orientation="h",
                               marker_color=["#c0392b" if v < 0 else "#2e8b57" for v in d["change"]]))
        fig.update_layout(title=title, height=230, margin=dict(l=0, r=0, t=40, b=0),
                          xaxis_title="Change (₹ lakh)" if m != "units" else "Change (units)")
        return fig

    if "trend_by_region" in T:
        d = T["trend_by_region"]
        fig = px.line(d, x="month", y=m, color="region", title=f"Monthly {m} by region")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    elif "trend" in T:
        fig = px.line(T["trend"], x="month", y=m, title=f"Monthly {m}: {f.get('scope', '')}")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    for name, dim in [("by_region", "region"), ("by_category", "category"), ("by_city", "city"), ("by_channel", "channel")]:
        if name in T:
            st.plotly_chart(bar(T[name], dim, f"Change in {m} by {dim}"), width="stretch")
    if "ranking" in T:
        dim = f["ranking"]["dimension"]
        fig = px.bar(T["ranking"], x=dim, y="current", title=f"{dim.title()} ranking: {m} ({f['period']['current']})")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    if "regions" in T:
        fig = px.bar(T["regions"], x="region", y=["previous", "current"], barmode="group", title=f"{m.title()} by region")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ transparency tabs
tab1, tab2, tab3 = st.tabs(["🔎 Evidence tables", "📚 Retrieved context (RAG)", "🧾 Verified facts (JSON)"])
with tab1:
    shown = False
    for name, t in ans.tables.items():
        if name.startswith("trend"):
            continue
        st.markdown(f"**{name.replace('_', ' ').title()}**")
        st.dataframe(t.round(1), width="stretch")
        shown = True
    if not shown:
        st.write("No evidence tables for this question type.")
with tab2:
    if f.get("context"):
        for c in f["context"]:
            st.markdown(f"- **{c['date']}** ({c['region']}, {c['area']}): {c['note']}")
        st.caption("Retrieved with TF-IDF similarity, boosted by region and time window. Context is background, not proof.")
    elif f.get("definitions"):
        for d in f["definitions"]:
            st.markdown(f"- **{d['term']}**: {d['meaning']}")
    else:
        st.write("No retrieval was needed for this question.")
with tab3:
    st.json(json.loads(json.dumps(public_facts(f), default=str)))
