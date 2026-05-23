"""
Streamlit UI for the Semantic Search Platform.
Run with: streamlit run streamlit_app.py
"""

import time

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="ArXiv Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────
st.markdown("""
<style>
.result-card {
    background: #f8f9fa;
    border-left: 4px solid #1a73e8;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    border-radius: 0 8px 8px 0;
}
.score-badge {
    background: #1a73e8;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.category-tag {
    background: #e8f0fe;
    color: #1a73e8;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    top_k = st.slider("Results to return", min_value=1, max_value=50, value=10)
    st.divider()

    st.subheader("Service Health")
    if st.button("Check Status", use_container_width=True):
        try:
            r = httpx.get(f"{API_BASE}/health", timeout=5)
            h = r.json()
            st.success(f"Status: {h['status']}")
            for svc, info in h["services"].items():
                st.text(f"• {svc}: {info}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

    st.divider()
    st.subheader("Index a Paper")
    with st.form("index_form"):
        pid = st.text_input("ArXiv ID", placeholder="2301.07041")
        title = st.text_input("Title")
        abstract = st.text_area("Abstract", height=100)
        submitted = st.form_submit_button("Index Paper")
        if submitted and pid and title and abstract:
            try:
                r = httpx.post(
                    f"{API_BASE}/index",
                    json={"paper_id": pid, "title": title, "abstract": abstract},
                    timeout=30,
                )
                st.success(r.json().get("message", "Indexed!"))
            except Exception as e:
                st.error(str(e))


# ── Main UI ───────────────────────────────────────────
st.title("🔍 ArXiv Semantic Search")
st.caption("Search 2M+ research papers by meaning — not keywords. Powered by SentenceTransformers + Milvus.")

query = st.text_input(
    "Search query",
    placeholder="e.g. neural networks that explain their own decisions",
    label_visibility="collapsed",
)

example_queries = [
    "transformers running on mobile devices",
    "detecting fake news automatically",
    "making AI explain its decisions",
    "graph neural networks for drug discovery",
    "diffusion models for image generation",
]

st.caption("Try: " + " · ".join(f"`{q}`" for q in example_queries))

if query:
    with st.spinner("Searching..."):
        t0 = time.monotonic()
        try:
            resp = httpx.post(
                f"{API_BASE}/search",
                json={"query": query, "top_k": top_k},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError:
            st.error("Cannot connect to API. Is the server running? (`uvicorn main:app --reload`)")
            st.stop()
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    results = data.get("results", [])
    took = data.get("took_ms", round((time.monotonic() - t0) * 1000))

    # ── Summary bar ───────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Results found", len(results))
    col2.metric("Time", f"{took}ms")
    col3.metric("Top score", f"{results[0]['score']:.3f}" if results else "—")

    if not results:
        st.info("No results found. Try indexing some papers first.")
        st.stop()

    # ── Score chart ───────────────────────────────────
    with st.expander("Score distribution", expanded=False):
        df = pd.DataFrame([{"title": r["title"][:50] + "…", "score": r["score"]} for r in results])
        fig = px.bar(df, x="score", y="title", orientation="h", color="score",
                     color_continuous_scale="Blues", height=300)
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Cosine similarity",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Results ───────────────────────────────────────
    for i, r in enumerate(results):
        with st.container():
            col_score, col_content = st.columns([1, 9])
            with col_score:
                st.markdown(
                    f'<div style="text-align:center; padding-top:8px">'
                    f'<span class="score-badge">{r["score"]:.3f}</span><br>'
                    f'<small style="color:#888">#{i+1}</small></div>',
                    unsafe_allow_html=True,
                )
            with col_content:
                st.markdown(f"**{r['title']}**")

                cats_html = " ".join(f'<span class="category-tag">{c}</span>' for c in r.get("categories", [])[:5])
                authors = ", ".join(r.get("authors", [])[:3])
                if len(r.get("authors", [])) > 3:
                    authors += " et al."
                meta = f'<small style="color:#888">{authors} · {r.get("published","")[:10]}</small>'
                if cats_html:
                    meta = cats_html + " &nbsp;" + meta
                st.markdown(meta, unsafe_allow_html=True)

                with st.expander("Abstract"):
                    st.write(r["abstract"])

                st.markdown(
                    f'<a href="https://arxiv.org/abs/{r["paper_id"]}" target="_blank" '
                    f'style="font-size:0.8rem">📄 View on ArXiv →</a>',
                    unsafe_allow_html=True,
                )
            st.divider()

else:
    st.markdown("""
    ### How it works
    1. Type a natural language query above
    2. Your query is embedded using `all-mpnet-base-v2`
    3. Milvus finds the most similar paper vectors (cosine similarity)
    4. Results are fetched from MongoDB and ranked by score

    **Semantic search means:** searching for *"AI that explains itself"* returns papers on
    explainability, LIME, SHAP — even if they don't contain those exact words.
    """)
