"""
Streamlit UI for the SHL Assessment Recommender.

Talks to the FastAPI service at API_BASE_URL (default http://localhost:8000).
Run with:  streamlit run streamlit_app.py
"""

import html
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from src.ui_helpers import (
    EXAMPLE_QUERIES,
    PIPELINE_STAGES,
    RECALL_AT_10,
    check_health,
    describe_reason,
    fetch_recommendations,
    format_duration,
    format_latency,
    to_rows,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

CSS = """
<style>
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .9rem 1.1rem; margin-bottom: .75rem;}
.card h4 {margin: 0 0 .35rem 0; font-size: 1.05rem;}
.card p {margin: .25rem 0 .5rem 0; opacity: .8; font-size: .9rem;}
.badge {display: inline-block; padding: .1rem .55rem; margin: 0 .3rem .3rem 0; border-radius: 999px;
        font-size: .78rem; border: 1px solid rgba(128,128,128,.35);}
.badge.on {background: rgba(46,160,67,.15); border-color: rgba(46,160,67,.5);}
</style>
"""


def render_sidebar() -> tuple:
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of results", 1, 20, 10)
        ratio = None
        if st.checkbox("Set Knowledge / Personality mix", help="Otherwise inferred from the query"):
            k = st.slider("Knowledge share", 0.0, 1.0, 0.6, 0.05)
            ratio = {"K": round(k, 2), "P": round(1 - k, 2)}

        st.divider()
        st.header("About")
        st.markdown("Recommends SHL assessments for a job description or hiring query.")
        st.markdown("**Pipeline**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(PIPELINE_STAGES, 1)))
        st.markdown("**Recall@10** (10 labelled queries, RRF fusion)")
        st.table(pd.DataFrame({"Recall@10": RECALL_AT_10}).style.format("{:.3f}"))
        st.caption("Only 10 labelled queries, so treat differences as directional. Details in analysis.md.")
        st.caption(f"API: {API_BASE_URL}")
    return top_k, ratio


def render_card(rank: int, a: dict) -> None:
    name = html.escape(a.get("name", "Unknown"))
    url = html.escape(a.get("url", "#"), quote=True)
    desc = a.get("description", "")
    desc = html.escape(desc[:220] + ("…" if len(desc) > 220 else ""))
    badges = [f'<span class="badge">⏱ {format_duration(a.get("duration"))}</span>']
    badges += [f'<span class="badge">{html.escape(t)}</span>' for t in a.get("test_type") or []]
    for label, key in (("Remote", "remote_support"), ("Adaptive", "adaptive_support")):
        on = a.get(key) == "Yes"
        badges.append(f'<span class="badge{" on" if on else ""}">{label}: {"Yes" if on else "No"}</span>')
    st.markdown(
        f'<div class="card"><h4>{rank}. <a href="{url}" target="_blank">{name}</a></h4>'
        f'<p>{desc}</p>{"".join(badges)}</div>',
        unsafe_allow_html=True,
    )


def render_results(result) -> None:
    if result.error:
        st.error(result.error)
        return
    if result.degraded_reasons:
        st.warning(
            "**Partial answer.** Some pipeline stages fell back, so quality may be lower:\n"
            + "\n".join(f"- {describe_reason(r)}" for r in result.degraded_reasons)
        )
    if not result.assessments:
        st.info("No matching assessments found. Try adding skills or the role level.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Results", len(result.assessments))
    c2.metric("Latency", format_latency(result.latency_ms))
    c3.metric("Status", "Degraded" if result.degraded_reasons else "Full")

    rows = to_rows(result.assessments)
    cards, table = st.tabs(["Cards", "Table"])
    with cards:
        for i, a in enumerate(result.assessments, 1):
            render_card(i, a)
    with table:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            column_config={"URL": st.column_config.LinkColumn("Link", display_text="Open")},
        )
    st.download_button(
        "Download CSV",
        pd.DataFrame(rows).to_csv(index=False),
        file_name=f"shl_recommendations_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="SHL Assessment Recommender", page_icon="🎯", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    top_k, ratio = render_sidebar()

    st.title("SHL Assessment Recommender")
    st.caption("Paste a job description or describe the role you are hiring for.")

    if not check_health(API_BASE_URL):
        st.error(
            f"The recommendation API at `{API_BASE_URL}` is not responding. "
            "Start it with `python run_api.py` (or set API_BASE_URL), then reload this page."
        )
        return

    st.session_state.setdefault("query", "")
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, (label, text) in zip(cols, EXAMPLE_QUERIES.items()):
        if col.button(label, use_container_width=True):
            st.session_state["query"] = text

    query = st.text_area("Job description or query", key="query", height=140,
                         placeholder="e.g. Mid-level Python developer with SQL and stakeholder communication")
    if st.button("Recommend assessments", type="primary"):
        if len(query.strip()) < 5:
            st.warning("Please enter at least a few words describing the role.")
        else:
            with st.spinner("Retrieving and ranking assessments…"):
                st.session_state["result"] = fetch_recommendations(
                    API_BASE_URL, query.strip(), top_k, ratio, api_key=os.getenv("API_KEY", "")
                )

    if "result" in st.session_state:
        render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
