"""
Real-time Claims Intelligence Platform — Streamlit Dashboard
Extensions: Outpatient Tab | Batch Analysis | Conversational Memory | Auto-refresh
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
import io

from app.data_ingestion import (
    init_db, ingest_inpatient, ingest_physician, ingest_outpatient,
    get_inpatient_claims, get_physician_claims, get_outpatient_claims,
    get_stats, save_analysis, get_recent_analyses,
)
from app.rag_pipeline import load_and_index_policies, get_kb_stats
from app.llm_analysis import (
    analyze_inpatient_claim, analyze_physician_claim, analyze_outpatient_claim,
    ask_question_about_claim, batch_analyze_claims,
)

st.set_page_config(page_title="Claims Intelligence Platform", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.risk-low    { color: #28a745; font-weight: bold; }
.risk-mod    { color: #ffc107; font-weight: bold; }
.risk-high   { color: #fd7e14; font-weight: bold; }
.risk-crit   { color: #dc3545; font-weight: bold; }
.chat-user   { background:#e8f4fd; padding:8px 12px; border-radius:8px; margin:4px 0; }
.chat-bot    { background:#f0f0f0; padding:8px 12px; border-radius:8px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "data_loaded": False,
    "kb_indexed": False,
    "selected_claim": None,
    "selected_type": None,
    "last_analysis": None,
    "chat_history": [],          # Extension 5: conversational memory
    "auto_refresh": False,       # Extension 8: auto-refresh toggle
    "last_refresh": time.time(),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Claims Intelligence")
    st.caption("CMS Data + RAG + Groq LLaMA 3.3 70B")
    st.divider()

    st.subheader("Setup")
    if st.button("1. Initialize Database", use_container_width=True):
        init_db()
        st.success("Database ready")

    if st.button("2. Load CMS Claims Data", use_container_width=True):
        with st.spinner("Fetching from CMS APIs..."):
            n_inp = ingest_inpatient(limit=300)
            n_phy = ingest_physician(limit=300)
            n_out = ingest_outpatient(limit=300)
        st.success(f"Loaded {n_inp} inpatient | {n_phy} physician | {n_out} outpatient")
        st.session_state.data_loaded = True

    if st.button("3. Index Policy Knowledge Base", use_container_width=True):
        with st.spinner("Embedding policy documents..."):
            n_chunks = load_and_index_policies(force_reload=True)
        st.success(f"Indexed {n_chunks} policy chunks")
        st.session_state.kb_indexed = True

    st.divider()

    # Extension 8: Auto-refresh toggle
    st.subheader("⚡ Auto-refresh")
    auto = st.toggle("Refresh every 60s", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto
    if auto:
        elapsed = int(time.time() - st.session_state.last_refresh)
        remaining = max(0, 60 - elapsed)
        st.caption(f"Next refresh in {remaining}s")
        if elapsed >= 60:
            st.session_state.last_refresh = time.time()
            st.rerun()

    st.divider()
    kb = get_kb_stats()
    db = get_stats()
    st.caption(f"KB: {kb['total_chunks']} chunks | {kb['documents']} docs")
    st.caption(f"Inpatient: {db['inpatient_count']} | Physician: {db['physician_count']} | Outpatient: {db['outpatient_count']}")
    st.caption(f"Analyses: {db['analyses_count']}")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "🏥 Inpatient", "👨‍⚕️ Physician", "🏨 Outpatient",
    "🤖 AI Analysis", "📦 Batch Analysis", "📜 History",
])


# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    st.title("Real-time Claims Intelligence Platform")
    st.caption("Medicare claims analysis using RAG + Groq LLaMA 3.3 70B")

    db = get_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Inpatient Claims",  f"{db['inpatient_count']:,}")
    c2.metric("Physician Claims",  f"{db['physician_count']:,}")
    c3.metric("Outpatient Claims", f"{db['outpatient_count']:,}")
    c4.metric("Avg Inpatient Charge", f"${db['inpatient_avg_charge']:,.0f}")
    c5.metric("AI Analyses Run",   f"{db['analyses_count']:,}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Inpatient Charge-to-Payment Ratio")
        claims = get_inpatient_claims(limit=20)
        if claims:
            df = pd.DataFrame(claims)
            fig = px.bar(df.head(15), x="drg_code", y="charge_to_payment_ratio",
                         color="charge_to_payment_ratio", color_continuous_scale="RdYlGn_r",
                         hover_data=["provider_name", "drg_description"],
                         labels={"charge_to_payment_ratio": "Ratio", "drg_code": "DRG"})
            fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="High Risk (10x)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load CMS data first (sidebar Step 2)")

    with col_r:
        st.subheader("Outpatient Charge-to-Allowed Ratio")
        out_claims = get_outpatient_claims(limit=20)
        if out_claims:
            df_out = pd.DataFrame(out_claims)
            fig2 = px.bar(df_out.head(15), x="apc_code", y="charge_to_allowed_ratio",
                          color="charge_to_allowed_ratio", color_continuous_scale="RdYlGn_r",
                          hover_data=["provider_name", "apc_description"],
                          labels={"charge_to_allowed_ratio": "Ratio", "apc_code": "APC"})
            fig2.add_hline(y=8, line_dash="dash", line_color="red", annotation_text="High Risk (8x)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Load CMS data first (sidebar Step 2)")

    # Risk distribution from analyses
    analyses = get_recent_analyses(limit=100)
    if analyses:
        st.divider()
        st.subheader("Risk Distribution of Analyzed Claims")
        risk_counts = {}
        for a in analyses:
            lbl = a.get("risk_label", "Unknown")
            risk_counts[lbl] = risk_counts.get(lbl, 0) + 1
        fig3 = px.pie(values=list(risk_counts.values()), names=list(risk_counts.keys()),
                      color=list(risk_counts.keys()),
                      color_discrete_map={"Low":"#28a745","Moderate":"#ffc107",
                                          "High":"#fd7e14","Critical":"#dc3545"})
        st.plotly_chart(fig3, use_container_width=True)


# ── Tab 2: Inpatient ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("Medicare Inpatient Claims (by Hospital & DRG)")
    claims = get_inpatient_claims(limit=100)
    if not claims:
        st.info("Load CMS data first (sidebar Step 2)")
    else:
        df = pd.DataFrame(claims)
        df["risk_flag"] = df["charge_to_payment_ratio"].apply(
            lambda x: "🔴 Critical" if x > 15 else ("🟠 High" if x > 10 else ("🟡 Moderate" if x > 6 else "🟢 Low"))
        )
        st.dataframe(df[["provider_name","city","state","drg_code","drg_description",
                          "total_discharges","avg_submitted_charge","avg_medicare_payment",
                          "charge_to_payment_ratio","risk_flag"]], use_container_width=True, height=400)
        st.divider()
        st.subheader("Select Claim → AI Analysis")
        opts = [f"{r['provider_name']} | DRG {r['drg_code']} | {r['charge_to_payment_ratio']}x" for r in claims]
        idx = st.selectbox("Inpatient claim", range(len(opts)), format_func=lambda i: opts[i], key="inp_sel")
        if st.button("Send to AI Analysis →", key="inp_btn", type="primary"):
            st.session_state.selected_claim = claims[idx]
            st.session_state.selected_type = "inpatient"
            st.session_state.chat_history = []
            st.success("Claim selected — go to AI Analysis tab")


# ── Tab 3: Physician ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Medicare Physician Claims (by Provider & HCPCS)")
    claims = get_physician_claims(limit=100)
    if not claims:
        st.info("Load CMS data first (sidebar Step 2)")
    else:
        df = pd.DataFrame(claims)
        st.dataframe(df[["provider_name","provider_type","city","state","hcpcs_code",
                          "hcpcs_description","total_beneficiaries","total_services",
                          "avg_submitted_charge","avg_medicare_payment"]], use_container_width=True, height=400)
        st.divider()
        st.subheader("Select Claim → AI Analysis")
        opts = [f"{r['provider_name']} ({r['provider_type']}) | {r['hcpcs_code']} | ${r['avg_submitted_charge']:,.0f}" for r in claims]
        idx = st.selectbox("Physician claim", range(len(opts)), format_func=lambda i: opts[i], key="phy_sel")
        if st.button("Send to AI Analysis →", key="phy_btn", type="primary"):
            st.session_state.selected_claim = claims[idx]
            st.session_state.selected_type = "physician"
            st.session_state.chat_history = []
            st.success("Claim selected — go to AI Analysis tab")


# ── Tab 4: Outpatient (Extension 10) ─────────────────────────────────────────
with tab4:
    st.subheader("Medicare Outpatient Claims (by Hospital & APC)")
    st.caption("APC = Ambulatory Payment Classification — how outpatient services are grouped and paid")

    claims = get_outpatient_claims(limit=100)
    if not claims:
        st.info("Load CMS data first (sidebar Step 2)")
    else:
        df = pd.DataFrame(claims)
        df["risk_flag"] = df["charge_to_allowed_ratio"].apply(
            lambda x: "🔴 Critical" if x > 12 else ("🟠 High" if x > 8 else ("🟡 Moderate" if x > 5 else "🟢 Low"))
        )

        col_l, col_r = st.columns(2)
        col_l.metric("Total Outpatient Records", f"{len(df):,}")
        col_r.metric("Avg Charge-to-Allowed Ratio", f"{df['charge_to_allowed_ratio'].mean():.1f}x")

        st.dataframe(df[["provider_name","city","state","apc_code","apc_description",
                          "beneficiary_count","total_services","avg_submitted_charge",
                          "avg_medicare_allowed","avg_medicare_payment",
                          "outlier_services","charge_to_allowed_ratio","risk_flag"]],
                     use_container_width=True, height=420)

        st.divider()
        col_chart, col_sel = st.columns([2, 1])
        with col_chart:
            top_apc = df.groupby("apc_description")["avg_submitted_charge"].mean().reset_index()
            top_apc = top_apc.sort_values("avg_submitted_charge", ascending=False).head(10)
            fig = px.bar(top_apc, x="avg_submitted_charge", y="apc_description",
                         orientation="h", color="avg_submitted_charge",
                         color_continuous_scale="Reds",
                         title="Top 10 APC Codes by Avg Submitted Charge",
                         labels={"avg_submitted_charge": "Avg Charge ($)", "apc_description": "APC"})
            st.plotly_chart(fig, use_container_width=True)

        with col_sel:
            st.subheader("Select → AI Analysis")
            opts = [f"{r['provider_name']} | APC {r['apc_code']} | {r['charge_to_allowed_ratio']}x" for r in claims]
            idx = st.selectbox("Outpatient claim", range(len(opts)), format_func=lambda i: opts[i], key="out_sel")
            if st.button("Send to AI Analysis →", key="out_btn", type="primary"):
                st.session_state.selected_claim = claims[idx]
                st.session_state.selected_type = "outpatient"
                st.session_state.chat_history = []
                st.success("Claim selected — go to AI Analysis tab")


# ── Tab 5: AI Analysis (with conversational memory) ───────────────────────────
with tab5:
    st.subheader("AI-Powered Claim Analysis")
    st.caption("RAG + Groq LLaMA 3.3 70B — with conversational memory")

    if st.session_state.selected_claim is None:
        st.info("Select a claim from the Inpatient, Physician, or Outpatient tabs first.")
    else:
        claim = st.session_state.selected_claim
        ctype = st.session_state.selected_type

        # Claim summary header
        st.subheader(f"Selected: {ctype.title()} Claim")
        if ctype == "inpatient":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hospital", claim.get("provider_name","")[:20])
            c2.metric("DRG", claim.get("drg_code",""))
            c3.metric("Avg Charge", f"${claim.get('avg_submitted_charge',0):,.0f}")
            c4.metric("Charge/Payment", f"{claim.get('charge_to_payment_ratio',0):.1f}x")
        elif ctype == "outpatient":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hospital", claim.get("provider_name","")[:20])
            c2.metric("APC", claim.get("apc_code",""))
            c3.metric("Avg Charge", f"${claim.get('avg_submitted_charge',0):,.0f}")
            c4.metric("Charge/Allowed", f"{claim.get('charge_to_allowed_ratio',0):.1f}x")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Provider", claim.get("provider_name","")[:20])
            c2.metric("HCPCS", claim.get("hcpcs_code",""))
            c3.metric("Avg Charge", f"${claim.get('avg_submitted_charge',0):,.0f}")
            c4.metric("Specialty", claim.get("provider_type","")[:18])

        st.divider()
        col_left, col_right = st.columns([1, 1])

        # ── Left: Run Analysis ────────────────────────────────────────────────
        with col_left:
            st.markdown("#### Run Full Analysis")
            if st.button("Analyze with AI", type="primary", use_container_width=True):
                with st.spinner("Retrieving policies + analyzing with LLaMA 3.3 70B..."):
                    if ctype == "inpatient":
                        result = analyze_inpatient_claim(claim)
                    elif ctype == "outpatient":
                        result = analyze_outpatient_claim(claim)
                    else:
                        result = analyze_physician_claim(claim)
                st.session_state.last_analysis = result

                claim_id = f"{ctype}_{claim.get('id',0)}_{int(time.time())}"
                save_analysis(claim_id, ctype, claim,
                              result.get("retrieved_policies", []),
                              result.get("explanation", ""),
                              result.get("risk_score", 0),
                              result.get("risk_label", "Unknown"))

            if st.session_state.last_analysis:
                r = st.session_state.last_analysis
                score = r.get("risk_score", 0)
                label = r.get("risk_label", "Unknown")
                color = {"Low":"green","Moderate":"orange","High":"red","Critical":"darkred"}.get(label,"gray")

                st.metric("Risk Score", f"{score}/100")
                st.markdown(f"**Risk:** <span style='color:{color};font-size:1.3em'>{label}</span>", unsafe_allow_html=True)
                st.progress(score / 100)

                st.markdown("**Key Findings:**")
                for f in r.get("key_findings", []):
                    st.markdown(f"- {f}")

                st.warning(f"**Policy Concern:** {r.get('policy_concerns','')}")
                st.info(f"**Action:** {r.get('recommended_action','')}")
                st.write(r.get("explanation",""))

                with st.expander("Retrieved Policy Context (RAG)"):
                    for i, p in enumerate(r.get("retrieved_policies",[]), 1):
                        st.markdown(f"**{i}.** `{p['source']}` | Relevance: `{p['similarity_score']}`")
                        st.text(p["text"][:400] + "..." if len(p["text"]) > 400 else p["text"])
                        st.divider()

        # ── Right: Conversational Q&A (Extension 5) ───────────────────────────
        with col_right:
            st.markdown("#### Chat About This Claim")
            st.caption("Memory is kept for the entire session — ask follow-up questions")

            # Show chat history
            chat_container = st.container(height=300)
            with chat_container:
                if not st.session_state.chat_history:
                    st.caption("No messages yet. Ask a question below.")
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

            question = st.text_input("Ask a question", placeholder="Why is this claim high risk?", key="qa_input")
            col_ask, col_clear = st.columns([3, 1])

            with col_ask:
                if st.button("Send", use_container_width=True, type="primary") and question:
                    with st.spinner("Thinking..."):
                        answer = ask_question_about_claim(
                            claim, question, ctype,
                            chat_history=st.session_state.chat_history,
                        )
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

            with col_clear:
                if st.button("Clear", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()


# ── Tab 6: Batch Analysis (Extension 2) ──────────────────────────────────────
with tab6:
    st.subheader("Batch Claim Analysis")
    st.caption("Analyze multiple claims at once and export results to CSV")

    col_cfg, col_run = st.columns([2, 1])
    with col_cfg:
        batch_type = st.selectbox("Claim type to batch analyze",
                                  ["inpatient", "physician", "outpatient"])
        batch_limit = st.slider("Number of claims to analyze", 5, 50, 10)
        st.warning(f"This will make {batch_limit} API calls to Groq. Each call ~2-3 seconds.")

    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_batch = st.button("Run Batch Analysis", type="primary", use_container_width=True)

    if run_batch:
        if batch_type == "inpatient":
            claims_to_analyze = get_inpatient_claims(limit=batch_limit)
        elif batch_type == "outpatient":
            claims_to_analyze = get_outpatient_claims(limit=batch_limit)
        else:
            claims_to_analyze = get_physician_claims(limit=batch_limit)

        if not claims_to_analyze:
            st.error("No claims loaded. Run Setup first.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"Analyzing claim {current}/{total}...")

            with st.spinner("Running batch analysis..."):
                results = batch_analyze_claims(claims_to_analyze, batch_type, update_progress)

            progress_bar.progress(1.0)
            status_text.text(f"Done! Analyzed {len(results)} claims.")

            # Save all to DB
            for r in results:
                save_analysis(
                    claim_id=f"batch_{batch_type}_{r.get('id',0)}_{int(time.time())}",
                    claim_type=batch_type,
                    claim_data={k: v for k, v in r.items()
                                if k not in ["risk_score","risk_label","key_findings",
                                             "policy_concerns","recommended_action","explanation"]},
                    retrieved_policies=[],
                    analysis=r.get("explanation",""),
                    risk_score=r.get("risk_score",0),
                    risk_label=r.get("risk_label","Unknown"),
                )

            # Results table
            df_results = pd.DataFrame(results)
            risk_cols = ["risk_score","risk_label","key_findings","recommended_action"]

            if batch_type == "inpatient":
                show_cols = ["provider_name","drg_code","avg_submitted_charge",
                             "charge_to_payment_ratio"] + risk_cols
            elif batch_type == "outpatient":
                show_cols = ["provider_name","apc_code","avg_submitted_charge",
                             "charge_to_allowed_ratio"] + risk_cols
            else:
                show_cols = ["provider_name","hcpcs_code","avg_submitted_charge"] + risk_cols

            show_cols = [c for c in show_cols if c in df_results.columns]

            st.divider()
            st.subheader(f"Results — {len(results)} claims analyzed")

            # Risk summary
            risk_dist = df_results["risk_label"].value_counts().reset_index()
            risk_dist.columns = ["Risk Level","Count"]
            col_tbl, col_chart = st.columns([1,1])
            with col_tbl:
                st.dataframe(df_results[show_cols], use_container_width=True, height=350)
            with col_chart:
                fig = px.pie(risk_dist, values="Count", names="Risk Level",
                             color="Risk Level",
                             color_discrete_map={"Low":"#28a745","Moderate":"#ffc107",
                                                 "High":"#fd7e14","Critical":"#dc3545"},
                             title="Batch Risk Distribution")
                st.plotly_chart(fig, use_container_width=True)

            # CSV Export
            csv_buffer = io.StringIO()
            df_results[show_cols].to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"batch_analysis_{batch_type}_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── Tab 7: History ────────────────────────────────────────────────────────────
with tab7:
    st.subheader("Analysis History")
    analyses = get_recent_analyses(limit=50)
    if not analyses:
        st.info("No analyses yet. Run some in AI Analysis or Batch Analysis tabs.")
    else:
        icon = {"Low":"🟢","Moderate":"🟡","High":"🟠","Critical":"🔴"}
        for a in analyses:
            lbl = a.get("risk_label","Unknown")
            with st.expander(f"{icon.get(lbl,'⚪')} {lbl} | {a['risk_score']}/100 | {a['claim_type'].title()} | {a['analyzed_at'][:19]}"):
                try:
                    cd = json.loads(a.get("claim_data","{}"))
                except Exception:
                    cd = {}
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Claim:**")
                    if a["claim_type"] == "inpatient":
                        st.write(f"Hospital: {cd.get('provider_name','')}")
                        st.write(f"DRG: {cd.get('drg_code','')} — {cd.get('drg_description','')}")
                    elif a["claim_type"] == "outpatient":
                        st.write(f"Hospital: {cd.get('provider_name','')}")
                        st.write(f"APC: {cd.get('apc_code','')} — {cd.get('apc_description','')}")
                    else:
                        st.write(f"Provider: {cd.get('provider_name','')}")
                        st.write(f"HCPCS: {cd.get('hcpcs_code','')}")
                with col2:
                    st.markdown("**Analysis:**")
                    st.write(a.get("analysis",""))
