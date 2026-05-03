from crewai import Task
from agents.report_writer import report_writer

def create_report_task(jd_task, resume_task, scorer_task) -> Task:
    return Task(
        description=(
            "Using the JD analysis, resume profile, and scorecard from "
            "previous agents, write a final hiring report.\n\n"
            "The report MUST follow this exact structure:\n\n"
            "================================================\n"
            "        CANDIDATE HIRING REPORT\n"
            "================================================\n\n"
            "CANDIDATE SNAPSHOT\n"
            "- Name:\n"
            "- Current Role & Company:\n"
            "- Total Experience:\n"
            "- Applying For:\n\n"
            "RECOMMENDATION: [STRONG YES / YES / MAYBE / NO]\n\n"
            "OVERALL SCORE: [X/100]\n\n"
            "WHY HIRE\n"
            "(3 bullet points — specific, evidence-backed, no fluff)\n\n"
            "WHY HESITATE\n"
            "(2-3 bullet points — honest gaps, not deal-breakers unless noted)\n\n"
            "SCORE BREAKDOWN\n"
            "- Technical Skills Match:    [X/25]\n"
            "- Experience Relevance:      [X/25]\n"
            "- Project Impact:            [X/20]\n"
            "- Hard Blocker Check:        [X/20]\n"
            "- Growth Potential:          [X/10]\n\n"
            "INTERVIEW RECOMMENDED: Yes/No\n"
            "SUGGESTED INTERVIEW FOCUS AREAS:\n"
            "(2-3 areas to probe based on gaps identified)\n\n"
            "================================================\n"
            "HIRING MANAGER ONE-LINER:\n"
            "[One sentence that captures the full picture]\n"
            "================================================\n"
        ),

        expected_output=(
            "A clean, formatted hiring report following the exact structure "
            "provided. Professional tone, scannable layout, no technical jargon. "
            "Uses only data from the provided scorecard — no hallucination."
        ),

        agent=report_writer,
        context=[jd_task, resume_task, scorer_task]  # all 3 previous outputs
    )