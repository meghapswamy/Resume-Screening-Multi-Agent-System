from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
from dotenv import load_dotenv
import os

load_dotenv()

# ── Configure RAGAS to use Groq instead of OpenAI ─────────────
def get_ragas_llm():
    """
    By default RAGAS uses OpenAI. We override it with Groq.
    RAGAS needs a LangChain-wrapped LLM — so we use ChatGroq here,
    NOT CrewAI's LLM class (which is LiteLLM-based).
    This is why we kept langchain-groq installed.
    """
    return LangchainLLMWrapper(
        ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192"  # no "groq/" prefix for LangChain
        )
    )

def get_ragas_embeddings():
    """
    RAGAS needs embeddings for answer_relevancy scoring.
    We use a free HuggingFace model instead of OpenAI embeddings.
    all-MiniLM-L6-v2 is small, fast, and good enough for evaluation.
    """
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

def evaluate_agent_output(
    jd_analysis: str,
    resume_profile: str,
    scorer_output: str
) -> dict:
    """
    Runs RAGAS evaluation on the scorer agent's output.

    Args:
        jd_analysis:    Output from Agent 1 (JD Analyst)
        resume_profile: Output from Agent 2 (Resume Parser)
        scorer_output:  Output from Agent 3 (Scorer)

    Returns:
        dict with faithfulness and answer_relevancy scores (0-1)
    """

    # RAGAS expects a dataset with these exact keys
    eval_data = {
        "question": [
            "Score the candidate's resume against the job description "
            "and provide a detailed scorecard with hire recommendation."
        ],
        "answer": [scorer_output],
        "contexts": [
            # contexts = what the agent had access to
            [jd_analysis, resume_profile]
        ]
    }

    dataset = Dataset.from_dict(eval_data)

    llm        = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    # Run evaluation
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False  # return partial results if one metric fails
    )

    scores = results.to_pandas()

    return {
        "faithfulness":      round(float(scores["faithfulness"].iloc[0]), 3),
        "answer_relevancy":  round(float(scores["answer_relevancy"].iloc[0]), 3),
        "overall_quality":   round(
            (float(scores["faithfulness"].iloc[0]) +
             float(scores["answer_relevancy"].iloc[0])) / 2, 3
        ),
        "hallucination_risk": (
            "🔴 HIGH"   if float(scores["faithfulness"].iloc[0]) < 0.5 else
            "🟡 MEDIUM" if float(scores["faithfulness"].iloc[0]) < 0.7 else
            "🟢 LOW"
        )
    }