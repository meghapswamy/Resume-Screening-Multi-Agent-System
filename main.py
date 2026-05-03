from crewai import Crew, Process
from tasks.jd_task import create_jd_task
from tasks.resume_task import create_resume_task
from tasks.scorer_task import create_scorer_task
from tasks.report_task import create_report_task
from tools.pdf_reader import read_pdf
import sys
import os

# ── Sample JD ──────────────────────────────────────────────────
sample_jd = """
We are looking for a Senior Data Scientist with 3-5 years of experience.
Required: Python, Machine Learning, LangChain, RAG pipelines, Azure OpenAI.
Preferred: LangGraph, CrewAI, MLflow, Docker.
You will build and deploy GenAI solutions, design RAG architectures,
evaluate LLM outputs using RAGAS, and collaborate with product teams.
Experience with vector databases (PGVector, Pinecone) is a must.
"""

# ── Load Resume ─────────────────────────────────────────────────
# Usage: python main.py resume.pdf
# If no PDF provided, falls back to sample text

if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    print(f"📄 Reading resume from: {pdf_path}")
    resume_text = read_pdf(pdf_path)
    print(f"✅ Extracted {len(resume_text)} characters from PDF\n")
else:
    print("ℹ️  No PDF provided — using sample resume\n")
    resume_text = """
    Megha P — AI/ML Engineer
    Experience: 2.7 years at Genpact

    Skills: Python, LangChain, RAG, Azure OpenAI, PGVector, FastAPI,
            LlamaIndex, Prompt Engineering, NLP, Pandas, Scikit-learn,
            PostgreSQL, Redis, Docker

    Projects:
    - GoDaddy AI Coach: Built GenAI training simulator using Azure OpenAI,
      LangChain, RAG, PGVector, Redis. Reduced AHT by 30%.
    - NPS Prediction Model: ML pipeline with 93% accuracy using Python,
      Scikit-learn, Azure ML.
    - AHT Automation Bot: Multi-turn conversational AI using GPT-4,
      LangChain, FastAPI.

    Education: B.E. Computer Science, RV College of Engineering, 2023. CGPA 8.75
    Certifications: AWS Solutions Architect Associate, ML Specialization (Coursera)
    """

# ── Build Task Chain ───────────────────────────────────────────
jd_task     = create_jd_task(sample_jd)
resume_task = create_resume_task(resume_text)
scorer_task = create_scorer_task(jd_task, resume_task)
report_task = create_report_task(jd_task, resume_task, scorer_task)

# ── Run Crew ───────────────────────────────────────────────────
crew = Crew(
    agents=[
        jd_task.agent,
        resume_task.agent,
        scorer_task.agent,
        report_task.agent
    ],
    tasks=[jd_task, resume_task, scorer_task, report_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()

print("\n" + "="*50)
print("         FINAL HIRING REPORT")
print("="*50 + "\n")
print(result)