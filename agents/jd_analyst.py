from crewai import Agent, LLM
from dotenv import load_dotenv
import os

load_dotenv()

# ── Correct way to pass Groq to CrewAI ────────────────────────
llm = LLM(
    model=os.getenv("GROQ_MODEL_NAME"),   # "groq/llama3-70b-8192"
    api_key=os.getenv("GROQ_API_KEY")     # your Groq key
)

jd_analyst = Agent(
    role="Job Description Analyst",

    goal=(
        "Extract and structure all key requirements from a job description — "
        "required skills, experience level, responsibilities, and nice-to-haves. "
        "Produce a clean, structured breakdown that other agents can score against."
    ),

    backstory=(
        "You are a senior technical recruiter with 10 years of experience hiring "
        "AI/ML and software engineers. You have reviewed thousands of job descriptions "
        "and know exactly which requirements are hard blockers vs. nice-to-haves. "
        "You think in structured categories: technical skills, years of experience, "
        "domain knowledge, and soft skills."
    ),

    llm=llm,               # ← LLM object, not a string anymore
    verbose=True,
    allow_delegation=False
)