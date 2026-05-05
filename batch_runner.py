import os
import json
import time
from datetime import datetime
from crewai import Crew, Process
from tasks.jd_task import create_jd_task
from tasks.resume_task import create_resume_task
from tasks.scorer_task import create_scorer_task
from tasks.report_task import create_report_task
from tools.ragas_evalutor import evaluate_agent_output

def extract_score_from_report(report_text: str) -> int:
    """
    Extracts the numeric total score from the report text.
    Looks for patterns like 'TOTAL SCORE: 82/100' or '82/100'
    Falls back to 0 if not found.
    """
    import re
    # Try to find score pattern like 82/100
    patterns = [
        r'TOTAL SCORE[:\s]+(\d+)',
        r'OVERALL SCORE[:\s]+(\d+)',
        r'(\d+)/100',
    ]
    for pattern in patterns:
        match = re.search(pattern, report_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score
    return 0

def extract_recommendation(report_text: str) -> str:
    """
    Extracts hire recommendation from report text.
    """
    import re
    match = re.search(
        r'RECOMMENDATION[:\s]+(Strong Yes|Yes|Maybe|No)',
        report_text,
        re.IGNORECASE
    )
    return match.group(1) if match else "Unknown"

def extract_candidate_name(report_text: str) -> str:
    """
    Extracts candidate name from report text.
    """
    import re
    match = re.search(r'Name[:\s]+([^\n]+)', report_text, re.IGNORECASE)
    return match.group(1).strip() if match else "Unknown Candidate"

def run_single_candidate(
    jd_text: str,
    resume_text: str,
    candidate_label: str,
    run_ragas: bool = True
) -> dict:
    """
    Runs the full 4-agent pipeline on one resume.
    Returns a result dict with report, scores, and RAGAS metrics.
    """
    try:
        # Build task chain
        jd_task     = create_jd_task(jd_text)
        resume_task = create_resume_task(resume_text)
        scorer_task = create_scorer_task(jd_task, resume_task)
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

        result = crew.kickoff()
        report_text = str(result)

        # Extract key fields from report
        score          = extract_score_from_report(report_text)
        recommendation = extract_recommendation(report_text)
        name           = extract_candidate_name(report_text)

        # RAGAS evaluation
        ragas_scores = {}
        if run_ragas:
            jd_output     = str(jd_task.output.raw)     if jd_task.output     else jd_text
            resume_output = str(resume_task.output.raw) if resume_task.output else resume_text
            scorer_output = str(scorer_task.output.raw) if scorer_task.output else report_text

            ragas_scores = evaluate_agent_output(
                jd_analysis=jd_output,
                resume_profile=resume_output,
                scorer_output=scorer_output
            )

        return {
            "status":          "success",
            "label":           candidate_label,
            "name":            name,
            "score":           score,
            "recommendation":  recommendation,
            "report":          report_text,
            "ragas_scores":    ragas_scores,
            "timestamp":       datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "status":          "error",
            "label":           candidate_label,
            "name":            candidate_label,
            "score":           0,
            "recommendation":  "Error",
            "report":          f"Error: {str(e)}",
            "ragas_scores":    {},
            "timestamp":       datetime.now().isoformat()
        }

def run_batch(
    jd_text: str,
    resumes: list[dict],   # [{"label": "resume1.pdf", "text": "..."}]
    run_ragas: bool = True,
    delay_seconds: int = 3,
    progress_callback=None
) -> dict:
    """
    Runs pipeline on multiple resumes sequentially.

    Args:
        jd_text:           The job description text
        resumes:           List of dicts with label + text
        run_ragas:         Whether to run RAGAS evaluation
        delay_seconds:     Delay between runs (rate limit protection)
        progress_callback: Optional function(i, total, label) for UI updates

    Returns:
        dict with results list + summary
    """
    results = []
    total   = len(resumes)

    for i, resume in enumerate(resumes):
        label = resume["label"]

        if progress_callback:
            progress_callback(i, total, label)

        result = run_single_candidate(
            jd_text=jd_text,
            resume_text=resume["text"],
            candidate_label=label,
            run_ragas=run_ragas
        )
        results.append(result)

        # Rate limit protection — wait between Groq API calls
        if i < total - 1:
            time.sleep(delay_seconds)

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Add rank
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Save batch run to results/
    os.makedirs("results", exist_ok=True)
    batch_record = {
        "timestamp": datetime.now().isoformat(),
        "type":      "batch",
        "total":     total,
        "results":   results
    }
    fname = f"results/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(batch_record, f, indent=2)

    return batch_record