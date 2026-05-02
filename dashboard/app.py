"""
Real-time Claims Intelligence Platform — Streamlit Dashboard

Sections:
  1. Sidebar: Setup (load data, index policies)
  2. Overview: KPI metrics from CMS data
  3. Claims Explorer: Browse real inpatient/physician claims
  4. AI Analysis: Run RAG + LLM on any claim (the main learning feature)
  5. Analysis History: See all past analyses
  6. Knowledge Base: Explore what's in the vector store
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time

from app.data_ingestion import (
    init_db, ingest_inpatient, ingest_physician,
    get_inpatient_claims, get_physician_claims,
    get_stats, save_analysis, get_recent_analyses,
)
from app.rag_pipeline import load_and_index_policies, get_kb_stats
from app.llm_analysis import analyze_inpatient_claim, analyze_physician_claim, ask_question_about_claim

st.set_page_config(
    page_title="Claims Intelligence Platform",
    page_icon="🏥",
    layout="wide",
)

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f0f2f6;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.risk-low { color: #28a745; font-weight: bold; }
.risk-moderate { color: #ffc107; font-weight: bold; }
.risk-high { color: #fd7e14; font-weight: bold; }
.risk-critical { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────
def init_session():
    for k, v in {
        "data_loaded": False,
        "kb_indexed": False,
        "selected_claim": None,
        "selected_type": None,
        "last_analysis": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Claims Intelligence")
    st.caption("Powered by CMS Data + RAG + Groq LLM")

    st.divider()
    st.subheader("Setup")

    if st.button("1. Initialize Database", use_container_width=True):
        init_db()
        st.success("Database ready")

    if st.button("2. Load CMS Claims Data", use_container_width=True):
        with st.spinner("Fetching from CMS APIs..."):
            n_inp = ingest_inpatient(limit=300)
            n_phy = ingest_physician(limit=300)
        st.success(f"Loaded {n_inp} inpatient + {n_phy} physician claims")
        st.session_state.data_loaded = True

    if st.button("3. Index Policy Knowledge Base", use_container_width=True):
        with st.spinner("Embedding policy documents..."):
            n_chunks = load_and_index_policies(force_reload=True)
        st.success(f"Indexed {n_chunks} policy chunks")
        st.session_state.kb_indexed = True

    st.divider()
    kb_stats = get_kb_stats()
    db_stats = get_stats()
    st.caption(f"KB: {kb_stats['total_chunks']} chunks | {kb_stats['documents']} docs")
    st.caption(f"Claims: {db_stats['inpatient_count']} inpatient | {db_stats['physician_count']} physician")
    st.caption(f"Analyses run: {db_stats['analyses_count']}")


# ── Main Content ─────────────────────────────────────────────────────────────
st.title("Real-time Claims Intelligence Platform")
st.caption("Medicare claims analysis using Retrieval-Augmented Generation (RAG) + Groq LLaMA 3.3 70B")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 Claims Explorer", "🤖 AI Analysis", "📜 History", "📚 Knowledge Base"
])


# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    stats = get_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Inpatient Claims", f"{stats['inpatient_count']:,}")
    col2.metric("Physician Claims", f"{stats['physician_count']:,}")
    col3.metric("Avg Inpatient Charge", f"${stats['inpatient_avg_charge']:,.0f}")
    col4.metric("AI Analyses Run", f"{stats['analyses_count']:,}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top Inpatient Claims by Charge Ratio")
        claims = get_inpatient_claims(limit=20)
        if claims:
            df = pd.DataFrame(claims)
            fig = px.bar(
                df.head(15),
                x="drg_code",
                y="charge_to_payment_ratio",
                color="charge_to_payment_ratio",
                color_continuous_scale="RdYlGn_r",
                hover_data=["provider_name", "drg_description", "avg_submitted_charge"],
                labels={"charge_to_payment_ratio": "Charge/Payment Ratio", "drg_code": "DRG Code"},
                title="Charge-to-Payment Ratio by DRG (higher = more scrutiny needed)",
            )
            fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="High Risk Threshold (10x)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load CMS data first (sidebar Step 2)")

    with col_right:
        st.subheader("Physician Billing by Specialty")
        phy_claims = get_physician_claims(limit=100)
        if phy_claims:
            df_phy = pd.DataFrame(phy_claims)
            specialty_avg = df_phy.groupby("provider_type")["avg_submitted_charge"].mean().reset_index()
            specialty_avg = specialty_avg.sort_values("avg_submitted_charge", ascending=False).head(10)
            fig2 = px.bar(
                specialty_avg,
                x="avg_submitted_charge",
                y="provider_type",
                orientation="h",
                color="avg_submitted_charge",
                color_continuous_scale="Blues",
                title="Avg Submitted Charge by Specialty (Top 10)",
                labels={"avg_submitted_charge": "Avg Charge ($)", "provider_type": "Specialty"},
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Load CMS data first (sidebar Step 2)")

    # Recent analyses risk distribution
    analyses = get_recent_analyses(limit=50)
    if analyses:
        st.subheader("Risk Distribution of Analyzed Claims")
        risk_counts = {}
        for a in analyses:
            label = a.get("risk_label", "Unknown")
            risk_counts[label] = risk_counts.get(label, 0) + 1
        fig3 = px.pie(
            values=list(risk_counts.values()),
            names=list(risk_counts.keys()),
            color=list(risk_counts.keys()),
            color_discrete_map={"Low": "#28a745", "Moderate": "#ffc107", "High": "#fd7e14", "Critical": "#dc3545"},
            title="Claims Risk Distribution",
        )
        st.plotly_chart(fig3, use_container_width=True)


# ── Tab 2: Claims Explorer ────────────────────────────────────────────────────
with tab2:
    st.subheader("Explore Real CMS Claims Data")

    claim_type = st.radio("Claim Type", ["Inpatient", "Physician"], horizontal=True)

    if claim_type == "Inpatient":
        claims = get_inpatient_claims(limit=100)
        if not claims:
            st.info("Load CMS data first (sidebar Step 2)")
        else:
            df = pd.DataFrame(claims)
            display_cols = ["provider_name", "city", "state", "drg_code",
                            "drg_description", "total_discharges",
                            "avg_submitted_charge", "avg_medicare_payment",
                            "charge_to_payment_ratio"]

            # Risk flag column
            df["risk_flag"] = df["charge_to_payment_ratio"].apply(
                lambda x: "🔴 Critical" if x > 15 else ("🟠 High" if x > 10 else ("🟡 Moderate" if x > 6 else "🟢 Low"))
            )

            st.dataframe(
                df[display_cols + ["risk_flag"]].rename(columns={
                    "provider_name": "Hospital",
                    "drg_code": "DRG",
                    "drg_description": "Description",
                    "total_discharges": "Discharges",
                    "avg_submitted_charge": "Avg Charge ($)",
                    "avg_medicare_payment": "Medicare Payment ($)",
                    "charge_to_payment_ratio": "Ratio",
                }),
                use_container_width=True,
                height=400,
            )

            st.divider()
            st.subheader("Select a Claim to Analyze")
            options = [f"{r['provider_name']} | DRG {r['drg_code']} | Ratio: {r['charge_to_payment_ratio']}x" for r in claims]
            selected_idx = st.selectbox("Choose claim", range(len(options)), format_func=lambda i: options[i])

            if st.button("Send to AI Analysis →", type="primary"):
                st.session_state.selected_claim = claims[selected_idx]
                st.session_state.selected_type = "inpatient"
                st.success("Claim selected. Go to the AI Analysis tab.")

    else:  # Physician
        claims = get_physician_claims(limit=100)
        if not claims:
            st.info("Load CMS data first (sidebar Step 2)")
        else:
            df = pd.DataFrame(claims)
            display_cols = ["provider_name", "provider_type", "city", "state",
                            "hcpcs_code", "hcpcs_description",
                            "total_beneficiaries", "total_services",
                            "avg_submitted_charge", "avg_medicare_payment"]
            st.dataframe(
                df[display_cols].rename(columns={
                    "provider_name": "Provider",
                    "provider_type": "Specialty",
                    "hcpcs_code": "HCPCS",
                    "hcpcs_description": "Description",
                    "total_beneficiaries": "Beneficiaries",
                    "total_services": "Services",
                    "avg_submitted_charge": "Avg Charge ($)",
                    "avg_medicare_payment": "Medicare Payment ($)",
                }),
                use_container_width=True,
                height=400,
            )

            st.divider()
            st.subheader("Select a Claim to Analyze")
            options = [f"{r['provider_name']} ({r['provider_type']}) | {r['hcpcs_code']} | ${r['avg_submitted_charge']:,.0f}" for r in claims]
            selected_idx = st.selectbox("Choose claim", range(len(options)), format_func=lambda i: options[i])

            if st.button("Send to AI Analysis →", type="primary"):
                st.session_state.selected_claim = claims[selected_idx]
                st.session_state.selected_type = "physician"
                st.success("Claim selected. Go to the AI Analysis tab.")


# ── Tab 3: AI Analysis ────────────────────────────────────────────────────────
with tab3:
    st.subheader("AI-Powered Claim Analysis (RAG + LLaMA 3.3 70B)")

    if not st.session_state.kb_indexed and get_kb_stats()["total_chunks"] == 0:
        st.warning("Index the knowledge base first (sidebar Step 3)")
    elif st.session_state.selected_claim is None:
        st.info("Select a claim from the Claims Explorer tab first.")
    else:
        claim = st.session_state.selected_claim
        claim_type = st.session_state.selected_type

        st.subheader("Selected Claim")
        if claim_type == "inpatient":
            cols = st.columns(4)
            cols[0].metric("Hospital", claim.get("provider_name", "")[:25])
            cols[1].metric("DRG Code", claim.get("drg_code", ""))
            cols[2].metric("Avg Charge", f"${claim.get('avg_submitted_charge', 0):,.0f}")
            cols[3].metric("Charge/Payment Ratio", f"{claim.get('charge_to_payment_ratio', 0):.1f}x")
            st.caption(f"DRG Description: {claim.get('drg_description', '')}")
        else:
            cols = st.columns(4)
            cols[0].metric("Provider", claim.get("provider_name", "")[:25])
            cols[1].metric("HCPCS", claim.get("hcpcs_code", ""))
            cols[2].metric("Avg Charge", f"${claim.get('avg_submitted_charge', 0):,.0f}")
            cols[3].metric("Specialty", claim.get("provider_type", "")[:20])

        st.divider()

        col_analyze, col_qa = st.columns([1, 1])

        with col_analyze:
            st.markdown("#### Run Full Analysis")
            st.caption("RAG retrieves relevant policies → LLM analyzes the claim")

            if st.button("Analyze Claim with AI", type="primary", use_container_width=True):
                with st.spinner("Retrieving policies from vector store..."):
                    time.sleep(0.5)

                with st.spinner("Sending to Groq LLaMA 3.3 70B..."):
                    if claim_type == "inpatient":
                        result = analyze_inpatient_claim(claim)
                    else:
                        result = analyze_physician_claim(claim)

                st.session_state.last_analysis = result

                # Save to DB
                claim_id = f"{claim_type}_{claim.get('id', 0)}_{int(time.time())}"
                save_analysis(
                    claim_id=claim_id,
                    claim_type=claim_type,
                    claim_data=claim,
                    retrieved_policies=result.get("retrieved_policies", []),
                    analysis=result.get("explanation", ""),
                    risk_score=result.get("risk_score", 0),
                    risk_label=result.get("risk_label", "Unknown"),
                )

        with col_qa:
            st.markdown("#### Ask a Question")
            st.caption("Conversational RAG — ask anything about this claim")
            question = st.text_input("Your question", placeholder="Why is this DRG code high risk?")
            if st.button("Ask", use_container_width=True) and question:
                with st.spinner("Retrieving context and generating answer..."):
                    answer = ask_question_about_claim(claim, question, claim_type)
                st.info(answer)

        # Show analysis results
        if st.session_state.last_analysis:
            result = st.session_state.last_analysis
            st.divider()
            st.subheader("Analysis Results")

            risk_score = result.get("risk_score", 0)
            risk_label = result.get("risk_label", "Unknown")
            color = {"Low": "green", "Moderate": "orange", "High": "red", "Critical": "darkred"}.get(risk_label, "gray")

            col_score, col_label = st.columns([1, 3])
            col_score.metric("Risk Score", f"{risk_score}/100")
            col_label.markdown(f"**Risk Level:** <span style='color:{color}; font-size:1.4em'>{risk_label}</span>", unsafe_allow_html=True)

            # Progress bar as risk gauge
            st.progress(risk_score / 100)

            st.markdown("**Key Findings:**")
            for finding in result.get("key_findings", []):
                st.markdown(f"- {finding}")

            col_pol, col_rec = st.columns(2)
            with col_pol:
                st.markdown("**Policy Concerns:**")
                st.warning(result.get("policy_concerns", "None identified"))
            with col_rec:
                st.markdown("**Recommended Action:**")
                st.info(result.get("recommended_action", "N/A"))

            st.markdown("**AI Explanation:**")
            st.write(result.get("explanation", ""))

            # Show retrieved policies (the RAG part)
            with st.expander("Retrieved Policy Context (RAG)", expanded=False):
                st.caption("These are the policy chunks retrieved from ChromaDB that informed the analysis.")
                for i, policy in enumerate(result.get("retrieved_policies", []), 1):
                    st.markdown(f"**Policy {i}** | Source: `{policy['source']}` | Relevance: `{policy['similarity_score']}`")
                    st.text(policy["text"][:500] + "..." if len(policy["text"]) > 500 else policy["text"])
                    st.divider()


# ── Tab 4: History ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Analysis History")
    analyses = get_recent_analyses(limit=30)

    if not analyses:
        st.info("No analyses yet. Run some claim analyses in the AI Analysis tab.")
    else:
        for a in analyses:
            risk_label = a.get("risk_label", "Unknown")
            color = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Critical": "🔴"}.get(risk_label, "⚪")
            with st.expander(f"{color} {risk_label} | Score: {a['risk_score']}/100 | {a['claim_type'].title()} | {a['analyzed_at'][:19]}"):
                claim_data = json.loads(a.get("claim_data", "{}"))
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Claim:**")
                    if a["claim_type"] == "inpatient":
                        st.write(f"Hospital: {claim_data.get('provider_name', '')}")
                        st.write(f"DRG: {claim_data.get('drg_code', '')} — {claim_data.get('drg_description', '')}")
                    else:
                        st.write(f"Provider: {claim_data.get('provider_name', '')}")
                        st.write(f"HCPCS: {claim_data.get('hcpcs_code', '')} — {claim_data.get('hcpcs_description', '')}")
                with col2:
                    st.markdown("**Analysis:**")
                    st.write(a.get("analysis", ""))


# ── Tab 5: Knowledge Base ─────────────────────────────────────────────────────
with tab5:
    st.subheader("RAG Knowledge Base Explorer")
    st.caption("These are the policy documents embedded in ChromaDB that the AI uses for context retrieval.")

    kb_stats = get_kb_stats()
    col1, col2 = st.columns(2)
    col1.metric("Total Chunks", kb_stats["total_chunks"])
    col2.metric("Documents Indexed", kb_stats["documents"])

    st.divider()

    if kb_stats["sources"]:
        st.markdown("**Indexed Documents:**")
        for source in kb_stats["sources"]:
            st.markdown(f"- `{source}`")

    st.divider()
    st.subheader("Test Retrieval")
    st.caption("Try a query to see what policies ChromaDB retrieves — this shows RAG in action.")

    test_query = st.text_input("Test query", placeholder="What is the fraud risk for high charge-to-payment ratio?")
    if st.button("Retrieve Policies", use_container_width=True) and test_query:
        from app.rag_pipeline import retrieve_relevant_policies
        with st.spinner("Searching vector store..."):
            results = retrieve_relevant_policies(test_query, n_results=4)

        if results:
            for i, r in enumerate(results, 1):
                with st.expander(f"Result {i} | Source: {r['source']} | Similarity: {r['similarity_score']}"):
                    st.write(r["text"])
        else:
            st.warning("No results. Index the knowledge base first (sidebar Step 3).")

    st.divider()
    st.subheader("How RAG Works")
    st.markdown("""
    ```
    INDEXING (one-time setup):
    Policy Documents (.txt)
         ↓ chunk_document()
    Text Chunks (400 words each)
         ↓ SentenceTransformer.encode()
    Dense Vectors (384 dimensions)
         ↓ collection.add()
    ChromaDB Vector Store

    RETRIEVAL (every query):
    User Query / Claim Details
         ↓ SentenceTransformer.encode()
    Query Vector
         ↓ cosine similarity search
    Top-K Relevant Chunks
         ↓ inject into LLM prompt
    Groq LLaMA 3.3 70B
         ↓
    Grounded Analysis + Risk Score
    ```
    """)
