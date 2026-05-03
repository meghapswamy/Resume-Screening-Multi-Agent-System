from crewai import Task
from agents.scorer import scorer

def create_scorer_task(jd_task,resume_task) -> Task:
    return Task(
        description=(
            "Using the JD analysis and resume profile from previous agents, "
            "score the candidate across these 5 categories:\n\n"
            "1. TECHNICAL SKILLS MATCH (0-25 pts)\n"
            "   Score how many required + preferred skills the candidate has.\n"
            "   Penalise if skills are listed without project evidence.\n\n"
            "2. EXPERIENCE RELEVANCE (0-25 pts)\n"
            "   Does their experience level and domain match the JD?\n"
            "   Production experience scores higher than academic/side projects.\n\n"
            "3. PROJECT IMPACT (0-20 pts)\n"
            "   Do their projects demonstrate measurable outcomes?\n"
            "   Look for numbers: accuracy %, time saved, scale, users.\n\n"
            "4. HARD BLOCKER CHECK (0-20 pts)\n"
            "   Does the candidate meet every hard blocker from the JD?\n"
            "   Any missing hard blocker = automatic deduction.\n\n"
            "5. GROWTH POTENTIAL (0-10 pts)\n"
            "   Certifications, side projects, learning trajectory.\n\n"
            "Finally provide:\n"
            "- TOTAL SCORE (out of 100)\n"
            "- HIRE RECOMMENDATION: Strong Yes / Yes / Maybe / No\n"
            "- TOP 3 STRENGTHS\n"
            "- TOP 3 GAPS\n"
            "- ONE LINE SUMMARY for the hiring manager\n"
        ),

        expected_output=(
            "A detailed scorecard with all 5 category scores, "
            "justified with specific references to the JD and resume. "
            "Followed by total score, hire recommendation, strengths, gaps, "
            "and a one-line hiring manager summary."
        ),

        agent=scorer,
        context=[jd_task,resume_task]  
    )