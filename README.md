# 🤖 Resume Screening Multi-Agent System v2.0

An intelligent resume screening pipeline built with **CrewAI**, **LangChain**, and **Groq (Llama 3 70B)**. Four specialised AI agents collaborate to analyse a job description, parse a resume, score the candidate, and produce a hiring report — automatically.

> Built as a hands-on learning project to understand multi-agent orchestration, agentic AI design patterns, LLM hallucination prevention, and LLMOps evaluation in production pipelines.

---

## 🆕 What's New in v2.0

| Feature | v1.0 | v2.0 |
|---|---|---|
| Resume screening | Single resume | Batch up to 10 resumes |
| Candidate ranking | Manual | Auto-ranked leaderboard |
| Output quality | No check | RAGAS evaluation on every run |
| Hallucination detection | None | Faithfulness + risk badge 🟢🟡🔴 |
| Run history | None | Persisted JSON audit trail |
| UI | 1 tab | 3 tabs (Single / Batch / History) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Input                        │
│         Job Description + Resume (PDF/Text)         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │   Agent 1: JD Analyst  │
          │  Extracts requirements │
          │  from job description  │
          └────────────┬───────────┘
                       │ structured JD breakdown
                       ▼
          ┌────────────────────────┐
          │ Agent 2: Resume Parser │
          │ Extracts candidate     │
          │ profile + gaps         │
          └────────────┬───────────┘
                       │ structured candidate profile
                       ▼
          ┌────────────────────────┐
          │   Agent 3: Scorer      │
          │ Scores candidate across│
          │ 5 categories (0–100)   │
          └────────────┬───────────┘
                       │ scorecard + recommendation
                       ▼
          ┌────────────────────────┐
          │ Agent 4: Report Writer │
          │ Produces final hiring  │
          │ report for manager     │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐        ┌─────────────────┐
          │     Final Output       │───────▶│  RAGAS Quality  │
          │  Hiring Report (txt)   │        │  Evaluation     │
          └────────────────────────┘        └─────────────────┘
```

Each agent receives **explicit context** from all upstream agents — preventing LLM hallucination across agent boundaries.

---

## 🧠 Agents

| Agent | Role | Output |
|---|---|---|
| **JD Analyst** | Extracts required skills, experience, responsibilities, and hard blockers from JD | Structured 5-section JD breakdown |
| **Resume Parser** | Extracts candidate profile including skills, projects, experience, and explicit gaps | Structured 6-section candidate profile |
| **Scorer** | Scores candidate against JD across 5 weighted categories | Scorecard (0–100) + hire recommendation |
| **Report Writer** | Formats scorecard into a clean, non-technical hiring report | Final hiring report |

### Scoring Rubric

| Category | Points | What it measures |
|---|---|---|
| Technical Skills Match | 0–25 | Required + preferred skills with project evidence |
| Experience Relevance | 0–25 | Years, domain, production vs academic experience |
| Project Impact | 0–20 | Measurable outcomes — %, time saved, scale |
| Hard Blocker Check | 0–20 | Auto-rejects if must-have requirements are missing |
| Growth Potential | 0–10 | Certifications, learning trajectory |

---

## 🧪 RAGAS Evaluation

Every run is automatically evaluated for output quality:

| Metric | What it checks | Catches |
|---|---|---|
| **Faithfulness** | Did agent use actual input data? | Hallucination |
| **Answer Relevancy** | Did output answer the task? | Off-topic outputs |
| **Hallucination Risk** | 🟢 LOW / 🟡 MEDIUM / 🔴 HIGH | Context grounding failures |

> In v1.0 the scorer hallucinated a fictional candidate "Rachel Lee" instead of using the actual resume. RAGAS catches this with a low faithfulness score and 🔴 HIGH risk flag.

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| **CrewAI** | Multi-agent orchestration framework |
| **Groq + Llama 3 70B** | Free, fast LLM inference (no OpenAI key needed) |
| **LiteLLM** | LLM provider abstraction layer |
| **LangChain** | Underlying LLM tooling used by CrewAI + RAGAS |
| **RAGAS** | LLM output evaluation framework |
| **FastEmbed** | Lightweight local embeddings (no torchvision needed) |
| **PyMuPDF** | PDF text extraction |
| **Streamlit** | Browser-based 3-tab UI |
| **Pandas** | Batch results leaderboard table |
| **Python 3.10+** | Core language |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/meghapswamy/Resume-Screening-Multi-Agent-System
cd Resume-Screening-Multi-Agent-System
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=groq/llama3-70b-8192
```

Get your **free** Groq API key at 👉 https://console.groq.com

### 5. Run

**Streamlit UI (recommended)**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

**Terminal**
```bash
# With a PDF resume
python main.py path/to/resume.pdf

# With sample data
python main.py
```

---

## 📁 Project Structure

```
Resume-Screening-Multi-Agent-System/
│
├── agents/
│   ├── jd_analyst.py        # Agent 1 — JD extraction
│   ├── resume_parser.py     # Agent 2 — resume parsing
│   ├── scorer.py            # Agent 3 — candidate scoring
│   └── report_writer.py     # Agent 4 — report generation
│
├── tasks/
│   ├── jd_task.py
│   ├── resume_task.py
│   ├── scorer_task.py       # explicit context injection
│   └── report_task.py       # explicit context injection
│
├── tools/
│   ├── pdf_reader.py        # PDF extraction (PyMuPDF)
│   └── ragas_evaluator.py   # RAGAS quality evaluation
│
├── results/                 # Auto-saved JSON run history
│   └── .gitkeep
│
├── app.py                   # Streamlit UI (3 tabs)
├── batch_runner.py          # Batch screening logic
├── main.py                  # Terminal entry point
├── .env                     # API keys — never committed
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 💡 Key Engineering Decisions

### 1. Explicit context injection over implicit passing
CrewAI passes outputs implicitly in sequential mode, but this caused the scorer to hallucinate a fictional candidate ("Rachel Lee") instead of using actual upstream outputs. Fix: explicitly passing `context=[jd_task, resume_task]` to downstream agents.

### 2. RAGAS before batch scaling
Evaluation infrastructure was built before scaling to batch. Running 10 unvalidated LLM outputs is 10× the hallucination risk. Evaluation first, scale second.

### 3. Sequential batch over parallel execution
Groq free tier has rate limits (~30 requests/minute). Sequential execution with configurable delay keeps us safely under the limit. In production with a paid tier: `asyncio` with a `Semaphore` to cap concurrency.

### 4. FastEmbed over sentence-transformers
FastEmbed is 10× lighter, has no `torchvision` dependency, and provides the same embedding quality for evaluation tasks. `sentence-transformers` caused import errors due to bundled computer vision model dependencies.

### 5. `task.output.raw` for intermediate capture
After `crew.kickoff()`, every task stores its output in `.output.raw`. This allows RAGAS to access intermediate agent outputs without any changes to agent or task code.

### 6. Separation of analysis and presentation
Agents 1–3 are analytical (extract, parse, score). Agent 4 is a communication agent (present). This keeps each agent's scope clean and outputs more reliable.

---

## 🔮 Roadmap

- [x] 4-agent multi-agent pipeline (CrewAI)
- [x] Streamlit UI
- [x] RAGAS evaluation — automated hallucination detection
- [x] Batch screening — upload up to 10 resumes, auto-rank candidates
- [x] Run history — persisted JSON audit trail
- [ ] MLflow experiment tracking — log runs, compare prompt versions
- [ ] LangGraph rewrite — conditional flows (second-opinion agent if score < 50)
- [ ] Bias detection agent — audit scorecard for non-skill correlations
- [ ] Export report as PDF
- [ ] Deploy on Streamlit Cloud

---

## 🎓 Concepts Demonstrated

| Concept | Where |
|---|---|
| Multi-agent orchestration | 4-agent CrewAI sequential pipeline |
| Explicit context injection | `context=[...]` in scorer + report tasks |
| LLM hallucination prevention | Context grounding fix in scorer task |
| LLMOps evaluation | RAGAS faithfulness + answer relevancy |
| Batch pipeline design | Sequential execution with rate-limit protection |
| LiteLLM provider routing | `groq/llama3-70b-8192` model string |
| Lightweight embeddings | FastEmbed over sentence-transformers |
| Run persistence | JSON audit trail per run |
| Production vs dev mode | `verbose=False` in UI, `True` in terminal |

---

## 🙏 Credits

- [CrewAI](https://docs.crewai.com) — multi-agent framework
- [Groq](https://console.groq.com) — free LLM inference
- [RAGAS](https://docs.ragas.io) — LLM evaluation framework
- [FastEmbed](https://github.com/qdrant/fastembed) — lightweight embeddings
- [Streamlit](https://streamlit.io) — UI framework
- [LiteLLM](https://docs.litellm.ai) — provider abstraction