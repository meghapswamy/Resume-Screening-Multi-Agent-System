from crewai import Agent, LLM
import os
from dotenv import load_dotenv
load_dotenv()

llm= LLM(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_MODEL_NAME")
)


resume_parser = Agent(
    role="Resume Parser",

    goal=(
        "Extract structured information from a raw resume text — "
        "skills, years of experience, past roles, projects, and education. "
        "Produce a clean structured profile that the Scorer agent can evaluate."
    ),

    backstory=(
        "You are an expert resume analyst who has parsed thousands of technical "
        "resumes for AI/ML and software engineering roles. You are precise, "
        "structured, and never infer what isn't explicitly stated. "
        "You separate what the candidate HAS done from what they claim to know."
    ),

    llm=llm,
    verbose=True,
    allow_delegation=False
)