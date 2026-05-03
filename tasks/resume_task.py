from crewai import Task
from agents.resume_parser import resume_parser

def create_resume_task(resume_text:str) ->Task:
    return Task(
        description=(
            f"Parse the following resume carefully:\n\n"
            f"{resume_text}\n\n"
            f"Extract and return a structured profile with these exact sections:\n"
            f"1. TECHNICAL SKILLS (list every tool, language, framework mentioned)\n"
            f"2. YEARS OF EXPERIENCE (total + breakdown by domain if possible)\n"
            f"3. PAST ROLES (company, title, duration)\n"
            f"4. KEY PROJECTS (name, tech stack, outcome/impact if mentioned)\n"
            f"5. EDUCATION & CERTIFICATIONS\n"
            f"6. GAPS OR WEAKNESSES (skills missing or unclear from the resume)\n"
        ),

        expected_output=(
            "A clean structured profile with all 6 sections clearly labeled. "
            "Only extract what is explicitly stated — do not assume or infer. "
            "Use bullet points under each section."
        ),

        agent=resume_parser
    )