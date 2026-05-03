from crewai import Task
from agents.jd_analyst import jd_analyst

def create_jd_task(job_description: str) -> Task:
    return Task(
        description=(
            f"Analyse the following job description carefully:\n\n"
            f"{job_description}\n\n"
            f"Extract and return a structured breakdown with these exact sections:\n"
            f"1. REQUIRED SKILLS (technical, must-have)\n"
            f"2. PREFERRED SKILLS (nice-to-have)\n"
            f"3. EXPERIENCE REQUIRED (years + domain)\n"
            f"4. KEY RESPONSIBILITIES (top 5)\n"
            f"5. HARD BLOCKERS (anything that would auto-reject a candidate)\n"
        ),

        expected_output=(
            "A clean structured breakdown with all 5 sections clearly labeled. "
            "Use bullet points under each section. Be specific and technical."
        ),

        agent=jd_analyst
    )