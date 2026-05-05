from tools.ragas_evalutor import evaluate_agent_output

# Minimal test data
test_jd = "Required: Python, LangChain, RAG. Experience: 3-5 years."

test_resume = "Megha P. Skills: Python, LangChain, RAG, PGVector. 2.7 years at Genpact."

test_scorer_output = """
TECHNICAL SKILLS MATCH: 22/25 - Candidate has Python, LangChain, RAG with project evidence.
EXPERIENCE RELEVANCE: 20/25 - 2.7 years, slightly below 3-year minimum.
PROJECT IMPACT: 18/20 - AHT reduced by 30%, NPS 93% accuracy.
HARD BLOCKER CHECK: 18/20 - All key requirements met.
GROWTH POTENTIAL: 8/10 - AWS cert, Coursera ML cert.
TOTAL: 86/100
RECOMMENDATION: Strong Yes
"""

print("Running RAGAS evaluation...")
scores = evaluate_agent_output(test_jd, test_resume, test_scorer_output)
print("\n=== RAGAS Scores ===")
for key, val in scores.items():
    print(f"{key}: {val}")