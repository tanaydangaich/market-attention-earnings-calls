"""
Market Attention Shifts Around Earnings Calls
Streamlit dashboard.

Run: streamlit run src/viz/app.py
  or: python main.py --app
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(
    page_title="Market Attention Shifts Around Earnings Calls",
    page_icon="📈",
    layout="wide",
)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    processed = ROOT / "data" / "processed"
    required = ["event_study.parquet", "tone_scores.parquet", "edgar_daily_top100.parquet"]
    missing = [f for f in required if not (processed / f).exists()]
    if missing:
        return None, None, None, None

    event_df = pd.read_parquet(processed / "event_study.parquet")
    tone_df = pd.read_parquet(processed / "tone_scores.parquet")
    edgar_df = pd.read_parquet(processed / "edgar_daily_top100.parquet")
    edgar_df["date"] = pd.to_datetime(edgar_df["date"])
    event_df["call_date"] = pd.to_datetime(event_df["call_date"])
    tone_df["call_date"] = pd.to_datetime(tone_df["call_date"])

    reg_path = processed / "regression_results.json"
    reg_results = json.loads(reg_path.read_text()) if reg_path.exists() else {}

    return event_df, tone_df, edgar_df, reg_results


@st.cache_data
def load_rag_index():
    """Lazy-load RAG index only when query tab is opened."""
    try:
        import chromadb
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore

        chroma_path = ROOT / "chroma_db"
        if not chroma_path.exists():
            return None

        chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        collection = chroma_client.get_or_create_collection("transcripts")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        return index.as_query_engine()
    except Exception:
        return None


# ── Helper: event window chart ────────────────────────────────────────────────

def plot_event_window(edgar_df, cik, call_date, company_name, lm_tone):
    """
    Plot daily downloads for a firm with the earnings call marked.
    Shows [-30, +60] day window around the call.
    """
    window_start = call_date - pd.Timedelta(days=30)
    window_end = call_date + pd.Timedelta(days=60)

    firm_data = edgar_df[
        (edgar_df["cik"] == str(cik))
        & (edgar_df["date"] >= window_start)
        & (edgar_df["date"] <= window_end)
    ]

    if firm_data.empty:
        return None

    daily = firm_data.groupby("date")["nr_total"].sum().reset_index()
    daily.columns = ["date", "downloads"]

    tone_color = (
        "#ef4444" if lm_tone < -0.5
        else "#f97316" if lm_tone < 0
        else "#22c55e"
    )
    tone_label = (
        "Negative" if lm_tone < -0.5
        else "Slightly Negative" if lm_tone < 0
        else "Positive"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["downloads"],
        mode="lines", name="Daily Downloads",
        line=dict(color="#6366f1", width=1.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
    ))
    fig.add_vline(
        x=call_date.timestamp() * 1000,
        line_dash="dash", line_color=tone_color, line_width=2,
        annotation_text=f"Earnings Call ({tone_label})",
        annotation_font_color=tone_color,
    )
    fig.update_layout(
        title=f"{company_name} — EDGAR Downloads Around {call_date.strftime('%b %d, %Y')} Call",
        xaxis_title="Date",
        yaxis_title="Daily Downloads",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def plot_average_event_study(event_df):
    """
    Average abnormal download % by relative day, split by tone quartile.
    The signature chart of the project.
    """
    event_df = event_df.dropna(subset=["lm_tone", "cad_30"])
    event_df["tone_quartile"] = pd.qcut(
        event_df["lm_tone"], q=4,
        labels=["Q1 (Most Negative)", "Q2", "Q3", "Q4 (Most Positive)"]
    )

    # compute average abnormal % by quartile (using cad_30_pct as proxy)
    summary = event_df.groupby("tone_quartile")["cad_30_pct"].agg(["mean", "sem"]).reset_index()
    summary.columns = ["tone_quartile", "mean_cad_pct", "se"]

    colors = ["#ef4444", "#f97316", "#84cc16", "#22c55e"]
    fig = go.Figure()
    for i, row in summary.iterrows():
        fig.add_trace(go.Bar(
            x=[row["tone_quartile"]],
            y=[row["mean_cad_pct"]],
            error_y=dict(type="data", array=[row["se"] * 1.96], visible=True),
            name=row["tone_quartile"],
            marker_color=colors[i],
        ))

    fig.update_layout(
        title="30-Day Cumulative Abnormal Downloads by Earnings Call Tone",
        xaxis_title="Tone Quartile",
        yaxis_title="Avg Abnormal Downloads (%)",
        showlegend=False,
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    return fig


def plot_crisis_comparison(event_df):
    """Crisis vs non-crisis: does the tone → attention effect change?"""
    event_df = event_df.dropna(subset=["lm_tone", "cad_30_pct"])
    event_df["period"] = event_df["is_crisis"].map(
        {True: "Crisis (2007-2009)", False: "Non-Crisis"}
    )
    event_df["tone_quartile"] = pd.qcut(
        event_df["lm_tone"], q=4,
        labels=["Q1 (Most Negative)", "Q2", "Q3", "Q4 (Most Positive)"]
    )

    summary = (
        event_df.groupby(["tone_quartile", "period"])["cad_30_pct"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        summary,
        x="tone_quartile", y="cad_30_pct", color="period",
        barmode="group",
        color_discrete_map={
            "Crisis (2007-2009)": "#ef4444",
            "Non-Crisis": "#6366f1",
        },
        title="Tone → Attention Effect: Crisis vs Non-Crisis Period",
        labels={"cad_30_pct": "Avg Abnormal Downloads (%)", "tone_quartile": "Tone Quartile"},
        height=380,
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40))
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Market Attention Shifts Around Earnings Calls")
    st.caption(
        "Do EDGAR filing download spikes follow negative earnings call tone? "
        "Analysis of 100 large-cap firms, 2003–2015 (includes 2008 financial crisis)."
    )

    event_df, tone_df, edgar_df, reg_results = load_data()

    if event_df is None:
        st.error(
            "Processed data not found. Run the pipeline first:\n\n"
            "```bash\npython main.py --mock\n```"
        )
        st.stop()

    is_mock = "is_mock" in tone_df.columns and tone_df["is_mock"].any()
    if is_mock:
        st.info(
            "Running on **synthetic data** for development. "
            "Replace with real SRAF + WRDS data and re-run `python main.py --real`.",
            icon="🧪",
        )

    # ── Sidebar ──
    st.sidebar.header("Filters")
    firms = sorted(event_df["company_name"].dropna().unique())
    selected_firm = st.sidebar.selectbox("Company", ["All"] + list(firms))

    years = sorted(event_df["fiscal_year"].dropna().unique().astype(int))
    year_range = st.sidebar.slider(
        "Year range", min_value=int(min(years)), max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )

    show_crisis = st.sidebar.checkbox("Highlight crisis period (2007–2009)", value=True)

    # ── Filter data ──
    filtered = event_df[
        (event_df["fiscal_year"] >= year_range[0])
        & (event_df["fiscal_year"] <= year_range[1])
    ]
    if selected_firm != "All":
        filtered = filtered[filtered["company_name"] == selected_firm]

    # ── Top metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Earnings Calls Analyzed", f"{len(filtered):,}")
    col2.metric("Firms", f"{filtered['cik'].nunique()}")
    neg_calls = filtered[filtered["lm_tone"] < 0]
    pos_calls = filtered[filtered["lm_tone"] >= 0]
    col3.metric(
        "Avg 30-Day Abnormal Downloads (Negative Calls)",
        f"{neg_calls['cad_30_pct'].mean():.1f}%",
        delta=f"{neg_calls['cad_30_pct'].mean() - pos_calls['cad_30_pct'].mean():.1f}% vs positive",
    )
    col4.metric(
        "Crisis Period Calls",
        f"{filtered['is_crisis'].sum():,}",
        delta=f"{filtered['is_crisis'].mean()*100:.0f}% of total",
    )

    st.divider()

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Event Study", "🏢 Firm Explorer", "⚠️ 2008 Crisis", "🤖 Ask the Transcripts"
    ])

    # Tab 1: Average event study chart
    with tab1:
        st.subheader("Do negative earnings calls drive EDGAR filing scrutiny?")
        st.plotly_chart(plot_average_event_study(filtered), use_container_width=True)

        st.caption(
            "Bars show average cumulative abnormal downloads in 30 days following an earnings call, "
            "grouped by tone quartile. Negative-tone calls (Q1) should show significantly higher "
            "post-call download activity than positive-tone calls (Q4)."
        )

        # Regression results
        if reg_results:
            st.subheader("Regression Results")
            m2 = reg_results.get("model_2_with_FE", {})
            rc1, rc2, rc3 = st.columns(3)
            coef = m2.get("coef_lm_tone")
            pval = m2.get("pval_lm_tone")
            r2 = m2.get("r_squared")
            n = m2.get("n_obs")
            if coef is not None:
                rc1.metric("LM Tone Coefficient", f"{coef:.4f}", help="CAD_30 ~ lm_tone + controls + firm FE + year FE")
                rc2.metric("p-value", f"{pval:.4f}", delta="significant" if pval < 0.05 else "not significant")
                rc3.metric("R² (with FE)", f"{r2:.3f}", help=f"N = {n:,} earnings calls")

            st.caption(
                "Model: `CAD_30 ~ lm_tone + lm_uncertain_pct + firm_FE + year_FE` "
                "with HC3 heteroskedasticity-robust standard errors. "
                "A negative coefficient on lm_tone confirms the hypothesis."
            )

    # Tab 2: Firm explorer
    with tab2:
        if selected_firm == "All":
            st.info("Select a specific company in the sidebar to explore individual calls.")
        else:
            firm_event = filtered[filtered["company_name"] == selected_firm].copy()
            firm_event = firm_event.sort_values("call_date")

            st.subheader(f"{selected_firm} — Earnings Call Timeline")

            # scatter: each call colored by tone
            fig_scatter = px.scatter(
                firm_event,
                x="call_date", y="cad_30_pct",
                color="lm_tone",
                color_continuous_scale="RdYlGn",
                hover_data=["fiscal_year", "fiscal_quarter", "lm_tone", "lm_uncertain_pct"],
                title=f"{selected_firm}: 30-Day Abnormal Downloads vs Call Tone",
                labels={"cad_30_pct": "30-Day Abnormal Downloads (%)", "call_date": "Call Date", "lm_tone": "LM Tone"},
                height=380,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # pick a specific call to drill into
            st.subheader("Drill into a specific call")
            firm_calls = firm_event[["call_date", "fiscal_year", "fiscal_quarter", "lm_tone", "cad_30_pct"]].copy()
            firm_calls["label"] = firm_calls.apply(
                lambda r: f"Q{int(r.fiscal_quarter)} {int(r.fiscal_year)} — Tone: {r.lm_tone:.2f}", axis=1
            )
            selected_call_label = st.selectbox("Select call", firm_calls["label"].tolist())
            selected_call = firm_calls[firm_calls["label"] == selected_call_label].iloc[0]
            call_date = pd.Timestamp(selected_call["call_date"])
            cik = event_df[event_df["company_name"] == selected_firm]["cik"].iloc[0]

            fig_window = plot_event_window(
                edgar_df, cik, call_date, selected_firm, selected_call["lm_tone"]
            )
            if fig_window:
                st.plotly_chart(fig_window, use_container_width=True)

            # show transcript snippet
            call_transcript = tone_df[
                (tone_df["company_name"] == selected_firm)
                & (pd.to_datetime(tone_df["call_date"]).dt.date == call_date.date())
            ]
            if not call_transcript.empty:
                with st.expander("Prepared remarks excerpt"):
                    text = call_transcript.iloc[0].get("prepared_text", "")
                    st.write(" ".join(str(text).split()[:200]) + "...")
                with st.expander("Q&A excerpt"):
                    text = call_transcript.iloc[0].get("qa_text", "")
                    st.write(" ".join(str(text).split()[:200]) + "...")

    # Tab 3: Crisis analysis
    with tab3:
        st.subheader("2008 Financial Crisis: Did the Effect Amplify?")
        st.plotly_chart(plot_crisis_comparison(filtered), use_container_width=True)
        st.caption(
            "During the 2008-2009 crisis, negative earnings calls may have driven "
            "disproportionately higher EDGAR scrutiny as investors and regulators sought "
            "to understand firm-level exposure. This chart tests whether the tone → attention "
            "effect differs in magnitude between crisis and non-crisis periods."
        )

        m3 = reg_results.get("model_3_crisis_interaction", {})
        if m3:
            c1, c2 = st.columns(2)
            c1.metric(
                "Crisis × Tone Interaction",
                f"{m3.get('coef_crisis_x_tone', 'N/A'):.4f}" if m3.get('coef_crisis_x_tone') else "N/A",
                help="A negative value means the effect of negative tone on downloads is larger during crisis"
            )
            c2.metric(
                "p-value (interaction)",
                f"{m3.get('pval_crisis_x_tone', 'N/A'):.4f}" if m3.get('pval_crisis_x_tone') else "N/A",
            )

    # Tab 4: RAG query
    with tab4:
        st.subheader("Ask questions about any earnings call")
        st.caption(
            "Powered by LlamaIndex + GPT-4o-mini over the full transcript corpus. "
            "Try: *'Which firms used the most uncertain language in 2008?'* or "
            "*'What did GE management say about liquidity in Q3 2008?'*"
        )

        query_engine = load_rag_index()

        if query_engine is None:
            st.warning(
                "RAG index not built yet. Run:\n\n"
                "```python\nfrom src.rag.indexer import build_index\nbuild_index()\n```"
            )
        else:
            query = st.text_input(
                "Query",
                placeholder="What did management say about credit risk in 2008?",
            )
            if query:
                with st.spinner("Searching transcripts..."):
                    try:
                        response = query_engine.query(query)
                        st.markdown(str(response))
                    except Exception as e:
                        st.error(f"Query failed: {e}")


if __name__ == "__main__":
    main()
