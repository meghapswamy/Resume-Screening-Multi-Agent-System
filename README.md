# 🤖 Resume Screening Multi-Agent System

An intelligent resume screening pipeline built with **CrewAI**, **LangChain**, and **Groq (Llama 3 70B)**. Four specialised AI agents collaborate to analyse a job description, parse a resume, score the candidate, and produce a hiring report — automatically.

> Built as part of a hands-on learning project to understand multi-agent orchestration, agentic AI design patterns, and LLM hallucination prevention in production pipelines.

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
          ┌────────────────────────┐
          │     Final Output       │
          │  Hiring Report (txt)   │
          └────────────────────────┘
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

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| **CrewAI** | Multi-agent orchestration framework |
| **Groq + Llama 3 70B** | Free, fast LLM inference (no OpenAI key needed) |
| **LiteLLM** | LLM provider abstraction layer |
| **LangChain** | Underlying LLM tooling used by CrewAI |
| **PyMuPDF** | PDF text extraction |
| **Streamlit** | Browser-based UI |
| **Python 3.10+** | Core language |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/resume-screening-agent
cd resume-screening-agent
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

**Option A — Streamlit UI (recommended)**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

**Option B — Terminal**
```bash
# With a PDF resume
python main.py path/to/resume.pdf

# With sample data
python main.py
```

---

## 📁 Project Structure

```
resume-screening-agent/
│
├── agents/
│   ├── jd_analyst.py        # Agent 1 — JD extraction
│   ├── resume_parser.py     # Agent 2 — resume parsing
│   ├── scorer.py            # Agent 3 — candidate scoring
│   └── report_writer.py     # Agent 4 — report generation
│
├── tasks/
│   ├── jd_task.py           # Task for Agent 1
│   ├── resume_task.py       # Task for Agent 2
│   ├── scorer_task.py       # Task for Agent 3 (explicit context injection)
│   └── report_task.py       # Task for Agent 4 (explicit context injection)
│
├── tools/
│   └── pdf_reader.py        # PDF text extraction utility (PyMuPDF)
│
├── app.py                   # Streamlit UI
├── main.py                  # Terminal entry point
├── .env                     # API keys — never commit this
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 💡 Key Engineering Decisions

### 1. Explicit context injection over implicit passing
CrewAI passes outputs implicitly in sequential mode, but this caused the scorer to hallucinate a fictional candidate ("Rachel Lee") instead of using actual upstream outputs. Fixing this required explicitly passing `context=[jd_task, resume_task]` to downstream agents — a real production failure mode and its fix.

### 2. LiteLLM for provider abstraction
Instead of hardcoding OpenAI, we use LiteLLM's routing string (`groq/llama3-70b-8192`). Swapping providers requires changing one environment variable — no code changes.

### 3. Separation of analysis and presentation
Agents 1–3 are analytical (extract, parse, score). Agent 4 is a communication agent (present). This separation keeps each agent's scope clean and outputs more reliable.

### 4. Gap extraction in resume parsing
The Resume Parser explicitly surfaces missing skills in a "GAPS" section rather than leaving absence detection to the Scorer. This prevents a known LLM failure mode where absence of information is not correctly inferred.

---

## 🔮 Roadmap

- [ ] RAGAS evaluation — automated hallucination detection on agent outputs
- [ ] Batch screening — upload 5–10 resumes, auto-rank all candidates  
- [ ] Multiple JD support — match one resume against several roles
- [ ] MLflow experiment tracking — log runs, compare prompt versions
- [ ] LangGraph rewrite — conditional flows (e.g. second-opinion agent if score < 50)
- [ ] Bias detection agent — audit scorecard for non-skill correlations
- [ ] Export report as PDF

---

## 🎓 Concepts Demonstrated

| Concept | Where |
|---|---|
| Multi-agent orchestration | 4-agent CrewAI sequential pipeline |
| Explicit context injection | `context=[...]` in scorer + report tasks |
| LLM hallucination prevention | Context grounding fix in scorer task |
| Sequential process design | `Process.sequential` — assembly line pattern |
| LiteLLM provider routing | `groq/llama3-70b-8192` model string |
| Role-based agent design | Each agent: single responsibility, clear scope |
| PDF input handling | PyMuPDF extraction with temp file cleanup |
| Production vs dev mode | `verbose=False` in UI, `True` in terminal |
| Dependency management | Pinned top-level, transitive resolved by framework |

---

## 🙏 Credits

- [CrewAI](https://docs.crewai.com) — multi-agent framework
- [Groq](https://console.groq.com) — free LLM inference
- [LiteLLM](https://docs.litellm.ai) — provider abstraction
- [Streamlit](https://streamlit.io) — UI framework