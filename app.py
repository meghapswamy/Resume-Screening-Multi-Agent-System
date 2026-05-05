import streamlit as st
import tempfile
import os
import json
from datetime import datetime
from crewai import Crew, Process
from tasks.jd_task import create_jd_task
from tasks.resume_task import create_resume_task
from tasks.scorer_task import create_scorer_task
from tasks.report_task import create_report_task
from tools.pdf_reader import read_pdf
from tools.ragas_evalutor import evaluate_agent_output

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screener v2.0",
    page_icon="🤖",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────
st.title("🤖 Resume Screening Agent")
st.caption("Powered by CrewAI · Groq · Llama 3 70B · RAGAS Evaluation")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🔍 Single Screen",
    "📦 Batch Screening"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SINGLE SCREEN
# ══════════════════════════════════════════════════════════════
with tab1:

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📋 Job Description")
        jd_text = st.text_area(
            label="JD",
            height=300,
            placeholder="""We are looking for a Senior Data Scientist with 3-5 years of experience.
Required: Python, Machine Learning, LangChain, RAG pipelines, Azure OpenAI.
Preferred: LangGraph, CrewAI, MLflow, Docker.
Experience with vector databases (PGVector, Pinecone) is a must.""",
            label_visibility="collapsed",
            key="single_jd"
        )

    with col2:
        st.subheader("📄 Resume")
        upload_tab, paste_tab = st.tabs(["Upload PDF", "Paste Text"])
        resume_text = ""

        with upload_tab:
            uploaded_file = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key="single_pdf"
            )
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                resume_text = read_pdf(tmp_path)
                os.unlink(tmp_path)
                st.success(f"✅ Extracted {len(resume_text)} characters")
                with st.expander("Preview extracted text"):
                    st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

        with paste_tab:
            pasted = st.text_area(
                "Paste resume",
                height=250,
                label_visibility="collapsed",
                key="single_paste"
            )
            if pasted:
                resume_text = pasted

    # ── Run Button ─────────────────────────────────────────────
    st.divider()
    run_col, _, status_col = st.columns([1, 2, 1])

    with run_col:
        run_btn = st.button(
            "🚀 Screen Candidate",
            type="primary",
            use_container_width=True,
            disabled=not (jd_text and resume_text),
            key="single_run"
        )
    with status_col:
        if not jd_text:
            st.warning("Add a job description")
        elif not resume_text:
            st.warning("Add a resume")
        else:
            st.success("Ready to screen!")

    # ── Run Pipeline ───────────────────────────────────────────
    if run_btn:
        st.divider()
        progress_bar = st.progress(0, text="Starting agents...")

        try:
            # Build tasks
            progress_bar.progress(0.20, text="🔍 Agent 1: Analysing JD...")
            jd_task     = create_jd_task(jd_text)

            progress_bar.progress(0.40, text="📄 Agent 2: Parsing resume...")
            resume_task = create_resume_task(resume_text)

            progress_bar.progress(0.60, text="⚖️ Agent 3: Scoring candidate...")
            scorer_task = create_scorer_task(jd_task, resume_task)

            progress_bar.progress(0.75, text="📝 Agent 4: Writing report...")
            report_task = create_report_task(jd_task, resume_task, scorer_task)

            crew = Crew(
                agents=[
                    jd_task.agent,
                    resume_task.agent,
                    scorer_task.agent,
                    report_task.agent
                ],
                tasks=[jd_task, resume_task, scorer_task, report_task],
                process=Process.sequential,
                verbose=False
            )

            with st.spinner("Agents working... (30–60 seconds)"):
                result = crew.kickoff()

            progress_bar.progress(0.90, text="🧪 Running RAGAS evaluation...")

            # ── RAGAS Evaluation ───────────────────────────────
            # Read intermediate outputs from completed tasks
            jd_output     = str(jd_task.output.raw)     if jd_task.output     else jd_text
            resume_output = str(resume_task.output.raw) if resume_task.output else resume_text
            scorer_output = str(scorer_task.output.raw) if scorer_task.output else str(result)

            with st.spinner("Evaluating output quality with RAGAS..."):
                ragas_scores = evaluate_agent_output(
                    jd_analysis=jd_output,
                    resume_profile=resume_output,
                    scorer_output=scorer_output
                )

            progress_bar.progress(1.0, text="✅ Complete!")

            # ── Display Report ─────────────────────────────────
            st.subheader("📊 Hiring Report")
            st.markdown(str(result))

            # ── Display RAGAS Quality Badge ────────────────────
            st.divider()
            st.subheader("🧪 Output Quality (RAGAS)")

            q1, q2, q3, q4 = st.columns(4)
            q1.metric(
                "Faithfulness",
                f"{ragas_scores['faithfulness']:.0%}",
                help="Did the agent use actual input data? Low = hallucination risk"
            )
            q2.metric(
                "Answer Relevancy",
                f"{ragas_scores['answer_relevancy']:.0%}",
                help="Did the output actually answer the scoring task?"
            )
            q3.metric(
                "Overall Quality",
                f"{ragas_scores['overall_quality']:.0%}",
            )
            q4.metric(
                "Hallucination Risk",
                ragas_scores['hallucination_risk']
            )

            # Warning if quality is low
            if ragas_scores['faithfulness'] < 0.5:
                st.error(
                    "⚠️ High hallucination risk detected. "
                    "The agent may have used fabricated data. "
                    "Re-run or review the report manually."
                )
            elif ragas_scores['faithfulness'] < 0.7:
                st.warning(
                    "🟡 Medium quality score. Review the report "
                    "before making hiring decisions."
                )
            else:
                st.success("✅ Output quality looks good. Low hallucination risk.")

            # ── Save to results/ ───────────────────────────────
            os.makedirs("results", exist_ok=True)
            run_record = {
                "timestamp":   datetime.now().isoformat(),
                "type":        "single",
                "report":      str(result),
                "ragas_scores": ragas_scores
            }
            fname = f"results/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(fname, "w") as f:
                json.dump(run_record, f, indent=2)

            # ── Download Button ────────────────────────────────
            st.divider()
            report_with_quality = (
                str(result) +
                f"\n\n--- RAGAS Quality Scores ---\n"
                f"Faithfulness:     {ragas_scores['faithfulness']:.0%}\n"
                f"Answer Relevancy: {ragas_scores['answer_relevancy']:.0%}\n"
                f"Overall Quality:  {ragas_scores['overall_quality']:.0%}\n"
                f"Hallucination Risk: {ragas_scores['hallucination_risk']}\n"
            )
            st.download_button(
                label="⬇️ Download Report + Quality Scores",
                data=report_with_quality,
                file_name="hiring_report.txt",
                mime="text/plain"
            )

        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# ══════════════════════════════════════════════════════════════
# TAB 2 — BATCH SCREENING (placeholder for Step 3)
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# TAB 2 — BATCH SCREENING
# ══════════════════════════════════════════════════════════════
with tab2:
    import pandas as pd
    from batch_runner import run_batch
    from tools.pdf_reader import read_pdf

    st.subheader("📦 Batch Resume Screening")
    st.caption("Upload one JD and multiple resumes — get a ranked leaderboard")

    # ── JD Input ───────────────────────────────────────────────
    st.markdown("**Step 1: Job Description**")
    batch_jd = st.text_area(
        "Batch JD",
        height=150,
        label_visibility="collapsed",
        key="batch_jd",
        placeholder="Paste the job description here..."
    )

    # ── Resume Uploads ─────────────────────────────────────────
    st.markdown("**Step 2: Upload Resumes (PDF, max 10)**")
    uploaded_resumes = st.file_uploader(
        "Upload resumes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_pdfs"
    )

    if uploaded_resumes:
        st.success(f"✅ {len(uploaded_resumes)} resume(s) uploaded")
        with st.expander("View uploaded files"):
            for f in uploaded_resumes:
                st.write(f"📄 {f.name}")

    # ── Options ────────────────────────────────────────────────
    st.markdown("**Step 3: Options**")
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        run_ragas_batch = st.checkbox(
            "Run RAGAS evaluation on each resume",
            value=True,
            help="Adds ~30 seconds per resume but detects hallucinations"
        )
    with opt_col2:
        delay = st.slider(
            "Delay between resumes (seconds)",
            min_value=2,
            max_value=10,
            value=3,
            help="Prevents Groq API rate limiting"
        )

    # ── Run Batch ──────────────────────────────────────────────
    st.divider()
    batch_ready = batch_jd and uploaded_resumes and len(uploaded_resumes) > 0

    batch_btn = st.button(
        f"🚀 Screen {len(uploaded_resumes) if uploaded_resumes else 0} Candidate(s)",
        type="primary",
        disabled=not batch_ready,
        key="batch_run"
    )

    if batch_btn:
        # Parse all PDFs
        resumes = []
        for f in uploaded_resumes[:10]:  # cap at 10
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            text = read_pdf(tmp_path)
            os.unlink(tmp_path)
            resumes.append({"label": f.name, "text": text})

        # Progress tracking
        progress_bar  = st.progress(0, text="Starting batch...")
        status_text   = st.empty()
        total         = len(resumes)

        def progress_callback(i, total, label):
            pct = i / total
            progress_bar.progress(
                pct,
                text=f"Processing {i+1}/{total}: {label}"
            )
            status_text.caption(f"⏳ Running agents on: {label}")

        # Run batch
        with st.spinner(f"Screening {total} candidates... (~{total * 60}s)"):
            batch_result = run_batch(
                jd_text=batch_jd,
                resumes=resumes,
                run_ragas=run_ragas_batch,
                delay_seconds=delay,
                progress_callback=progress_callback
            )

        progress_bar.progress(1.0, text="✅ Batch complete!")
        status_text.empty()

        # ── Leaderboard Table ──────────────────────────────────
        st.divider()
        st.subheader("🏆 Candidate Leaderboard")

        results = batch_result["results"]

        # Build DataFrame
        table_data = []
        for r in results:
            row = {
                "Rank":           f"#{r['rank']}",
                "Candidate":      r["name"],
                "File":           r["label"],
                "Score":          f"{r['score']}/100",
                "Recommendation": r["recommendation"],
                "Status":         r["status"]
            }
            if run_ragas_batch and r.get("ragas_scores"):
                row["Faithfulness"]  = f"{r['ragas_scores'].get('faithfulness', 0):.0%}"
                row["Quality"]       = r['ragas_scores'].get('hallucination_risk', 'N/A')
            table_data.append(row)

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Individual Reports ─────────────────────────────────
        st.divider()
        st.subheader("📄 Individual Reports")

        for r in results:
            emoji = "🥇" if r["rank"] == 1 else "🥈" if r["rank"] == 2 else "🥉" if r["rank"] == 3 else "📄"
            with st.expander(
                f"{emoji} #{r['rank']} — {r['name']} | "
                f"Score: {r['score']}/100 | {r['recommendation']}"
            ):
                st.markdown(r["report"])
                if run_ragas_batch and r.get("ragas_scores"):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Faithfulness",     f"{r['ragas_scores'].get('faithfulness', 0):.0%}")
                    m2.metric("Answer Relevancy", f"{r['ragas_scores'].get('answer_relevancy', 0):.0%}")
                    m3.metric("Hallucination Risk", r['ragas_scores'].get('hallucination_risk', 'N/A'))

        # ── Download Full Batch Report ─────────────────────────
        st.divider()
        batch_export = json.dumps(batch_result, indent=2)
        st.download_button(
            label="⬇️ Download Full Batch Report (JSON)",
            data=batch_export,
            file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

