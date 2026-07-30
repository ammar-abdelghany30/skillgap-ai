# SkillGap-AI

An AI-powered career coach that analyzes a candidate's CV against a target job description, identifies missing skills, validates them against the real job market using RAG, and generates a personalized learning roadmap.

Built with **LangChain**, **Mistral AI**, **FAISS**, **HuggingFace Embeddings**, **Pydantic**, and **Streamlit**.

---

# Features

- 📄 Extract structured information from CVs
- 💼 Extract skills and requirements from job descriptions
- 📊 Compare candidate skills against job requirements
- 🔍 Validate missing skills using real-world job postings (RAG)
- 📚 Generate personalized learning roadmaps from roadmap.sh content
- 💡 Suggest alternative career paths based on the candidate's current skills
- 🖥️ Interactive Streamlit interface

---

# Project Architecture

```text
User uploads

CV.pdf
JD.pdf
      │
      ▼
Extract PDF Text
      │
      ▼
run_end_to_end()
      │
      ▼
──────────────────────────────────────
Chain 1
Extract candidate information
──────────────────────────────────────
      │
      ▼
CVExtractionResult
      │
──────────────────────────────────────
Chain 2
Extract job requirements
──────────────────────────────────────
      │
      ▼
JDExtractionResult
      │
──────────────────────────────────────
Chain 3
Compare candidate vs job
──────────────────────────────────────
      │
      ▼
GapAnalysisResult
      │
      ├──────────────► Chain 4
      │                  Retrieve roadmap documents
      │                  from FAISS
      │                  ↓
      │                  Generate learning roadmap
      │
      └──────────────► Chain 5
                         Retrieve similar jobs
                         from FAISS
                         ↓
                         Suggest alternative roles
      │
      ▼
Merge all outputs
      │
      ▼
Return structured JSON
      │
      ▼
Display in Streamlit
```

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Mistral Small |
| Framework | LangChain LCEL |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Output Parsing | Pydantic |
| UI | Streamlit |
| PDF Parsing | PyPDF |
| Environment | Python |

---

# RAG Knowledge Sources

The project builds **two independent FAISS indexes**.

### Job Postings Index

- 600 real job postings
- Filtered from a Kaggle dataset
- Covers:
  - Backend Developer
  - Frontend Developer
  - Data Analyst
  - Data Scientist

Used for:

- Market validation
- Similar job retrieval

---

### Learning Roadmap Index

- 1,171 markdown documents
- Extracted from roadmap.sh

Used for:

- Learning roadmap generation
- Skill recommendations

---

# Chunking Strategy

Instead of splitting documents into fixed-size chunks, the project keeps each document as its natural semantic unit.

### Job Postings

- One CSV row = one chunk

Each posting already contains:

- title
- skills
- requirements
- responsibilities

Keeping them together preserves context.

---

### Roadmap Documents

- One markdown file = one chunk

Each roadmap.sh topic already represents one learning concept.

No additional splitting was needed.

---

# Market Evidence for Missing Skills ⭐

One of the project's unique features is **Market Evidence**.

Instead of trusting the LLM alone, every missing skill is validated against **real job postings**.

Example:

```
Missing Skill:
Docker
```

The system searches similar job postings for the same target role.

Result:

```
Docker appears in

83 / 100 Backend Developer jobs
```

This allows the application to distinguish between:

- skills required by almost every employer
- skills mentioned by only one company

As a result, the learning roadmap is grounded in actual market demand rather than only the language model's reasoning.

---

# Pipeline

### Chain 1 — CV Extraction

Input:

- CV text

Output:

- structured candidate profile

---

### Chain 2 — Job Description Extraction

Input:

- Job Description

Output:

- structured requirements

---

### Chain 3 — Skill Gap Analysis

Compares:

- candidate skills
- experience
- education

against

- job requirements

Produces:

- current skills
- missing skills
- match percentage

---

### Chain 4 — Roadmap Generation (RAG)

Each missing skill retrieves the most relevant roadmap documents before generating learning steps.

---

### Chain 5 — Suggested Jobs (RAG)

Searches similar job postings based on the candidate's current skills and recommends adjacent career paths.

---

# Output

The application returns a structured Pydantic model containing:

- Match Percentage
- Current Skills
- Missing Skills
- Market Evidence
- Personalized Learning Roadmap
- Suggested Jobs
- Overall Career Feedback

---

# Setup

```bash
git clone <repo>

pip install -r requirements.txt

python src/ingestion.py

python src/test_retrieval.py

streamlit run app.py
```

---

# Project Structure

```
SkillGap-AI
│
├── app.py
├── src
│   ├── chains.py
│   ├── schemas.py
│   ├── ingestion.py
│   ├── app.py
│   └── test_retrieval.py
│
├── data
│   ├── raw
│   └── roadmaps
│
├── vectorstore
│   ├── job_postings_index
│   └── roadmap_index
│
└── requirements.txt
```
