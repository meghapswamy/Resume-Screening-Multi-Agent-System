import streamlit as st
import tempfile
import os
from crewai import Crew, Process
from tasks.jd_task import create_jd_task
from tasks.resume_task import create_resume_task
from tasks.scorer_task import create_scorer_task
from tasks.report_task import create_report_task
from tools.pdf_reader import read_pdf

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screener",
    page_icon="🤖",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────
st.title("🤖 Resume Screening Agent")
st.caption("Powered by CrewAI · Groq · Llama 3 70B")
st.divider()

# ── Layout: Two columns ────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Job Description")
    jd_text = st.text_area(
        label="Paste the job description here",
        height=300,
        placeholder="""We are looking for a Senior Data Scientist with 3-5 years of experience.
Required: Python, Machine Learning, LangChain, RAG pipelines, Azure OpenAI.
Preferred: LangGraph, CrewAI, MLflow, Docker.
Experience with vector databases (PGVector, Pinecone) is a must.""",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("📄 Resume")
    upload_tab, paste_tab = st.tabs(["Upload PDF", "Paste Text"])

    resume_text = ""

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload resume PDF",
            type=["pdf"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            # Save to temp file and extract text
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            resume_text = read_pdf(tmp_path)
            os.unlink(tmp_path)  # clean up temp file
            st.success(f"✅ Extracted {len(resume_text)} characters from PDF")
            with st.expander("Preview extracted text"):
                st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

    with paste_tab:
        pasted_resume = st.text_area(
            "Paste resume text",
            height=250,
            placeholder="Paste resume content here...",
            label_visibility="collapsed"
        )
        if pasted_resume:
            resume_text = pasted_resume

# ── Run Button ─────────────────────────────────────────────────
st.divider()
run_col, _, info_col = st.columns([1, 2, 1])

with run_col:
    run_btn = st.button(
        "🚀 Screen Candidate",
        type="primary",
        use_container_width=True,
        disabled=not (jd_text and resume_text)
    )

with info_col:
    if not jd_text:
        st.warning("Add a job description")
    elif not resume_text:
        st.warning("Add a resume")
    else:
        st.success("Ready to screen!")

# ── Agent Progress + Results ───────────────────────────────────
if run_btn:
    st.divider()

    # Progress tracking
    progress_bar = st.progress(0, text="Starting agents...")
    status = st.empty()

    agent_steps = [
        (0.25, "🔍 Agent 1: Analysing job description..."),
        (0.50, "📄 Agent 2: Parsing resume..."),
        (0.75, "⚖️  Agent 3: Scoring candidate..."),
        (1.00, "📝 Agent 4: Writing hiring report..."),
    ]

    # We'll update progress using a callback
    step_index = [0]

    def update_progress():
        if step_index[0] < len(agent_steps):
            val, msg = agent_steps[step_index[0]]
            progress_bar.progress(val, text=msg)
            status.caption(msg)
            step_index[0] += 1

    # Kick off first progress update
    update_progress()

    try:
        # ── Build and run Crew ─────────────────────────────────
        jd_task     = create_jd_task(jd_text)
        update_progress()

        resume_task = create_resume_task(resume_text)
        update_progress()

        scorer_task = create_scorer_task(jd_task, resume_task)
        update_progress()

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
            verbose=False  # ← clean UI, no console noise
        )

        with st.spinner("Agents working... this takes 30–60 seconds"):
            result = crew.kickoff()

        progress_bar.progress(1.0, text="✅ Complete!")
        status.empty()

        # ── Display Results ────────────────────────────────────
        st.divider()
        st.subheader("📊 Hiring Report")

        # Show raw report in a nice box
        st.markdown(str(result))

        # Download button
        st.divider()
        st.download_button(
            label="⬇️ Download Report",
            data=str(result),
            file_name="hiring_report.txt",
            mime="text/plain",
            use_container_width=False
        )

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Something went wrong: {str(e)}")
        st.exception(e)