# SkillGap-AI

RAG-powered career coach that analyzes CVs, identifies skill gaps against real job requirements, and generates personalized learning roadmaps.

## Architecture

**Input:** User-pasted CV text + target job description text
**Pipeline:** LangChain LCEL chains (extraction → gap analysis → roadmap generation), grounded by two separate FAISS retrieval indexes
**Output:** Structured `CareerGapAnalysis` (current skills, missing skills, roadmap, suggested jobs) via Pydantic output parsing

## Data sources

| Source | Content | Rows/Chunks |
|---|---|---|
| Kaggle job descriptions dataset (filtered) | Real-world job postings across 4 target roles (Backend Developer, Frontend Developer, Data Analyst, Data Scientist) | 600 |
| roadmap.sh content (via sparse git checkout) | Topic-level learning content across 10 career roadmaps | 1,171 |

## Chunking strategy — and why

RAG retrieval quality depends more on chunk boundaries than on embedding model choice. The two data sources here already have natural semantic units, so **document-level chunking** was used instead of fixed-size character splitting:

- **Job postings:** one CSV row = one chunk. A posting's title, skills, and requirements are only meaningful together — splitting a single posting into smaller pieces (e.g. by character count) would separate "5 years experience" from *what* it applies to, destroying the context needed for accurate skill-gap comparison.
- **Roadmap topics:** one markdown file = one chunk. The roadmap.sh source repo already structures content as one file per topic node (e.g. `docker@<id>.md`), so each file is already a coherent, self-contained unit — re-splitting it would only fragment an already-correct boundary.

**Why not `RecursiveCharacterTextSplitter` (the common default)?** Fixed-size chunking is the right tool when a source has no natural structure (e.g. a long-form article). Both data sources here already had natural document boundaries, so respecting those boundaries directly produced cleaner, more coherent chunks than arbitrary size-based splitting would have.

## Embedding model

`sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face, run locally through `langchain-huggingface`)

Chosen over a paid API-based embedding model (e.g. OpenAI embeddings) because:
- Free and runs locally — no API cost or rate limits during heavy iteration while building
- Fast enough for this dataset size (600 + 1,171 chunks embeds in seconds)
- 384-dimension output is sufficient quality for this scale of retrieval; the paid LLM API budget is reserved for the reasoning chains (CV/JD extraction, gap analysis) where output quality matters most

## Vector store

FAISS, via `langchain_community.vectorstores.FAISS`, with **two separate indexes** rather than one combined index:

- `vectorstore/job_postings_index`
- `vectorstore/roadmap_index`

Kept separate because the two data types serve distinct purposes in the pipeline — job postings provide market-context grounding for the gap analysis, while roadmaps drive the learning-plan generation. A single mixed index risks a query for "learning resources" returning a job posting, or vice versa; separate indexes make retrieval purpose-specific by construction.

## Retrieval validation

Chunking and embedding choices were validated with manual similarity-search sanity checks (see `test_retrieval.py`) before wiring retrieval into any LLM chain — confirming, for example, that querying "Docker containers" correctly surfaces Docker-specific roadmap content (including a semantically-related "Containers" topic chunk, not just exact keyword matches) rather than unrelated topics.

## Pipeline stages (chains.py)

1. **CV extraction** (no retrieval) — structured skill/experience extraction from user-provided CV text
2. **JD extraction** (no retrieval) — structured requirement extraction from user-provided job description text
3. **Gap comparison** (retrieval-grounded) — compares extracted skills vs. requirements; missing skills are cross-checked against `job_postings_index` to distinguish company-specific quirks from genuine market-wide requirements
4. **Roadmap generation** (retrieval-driven) — each missing skill is queried against `roadmap_index`; retrieved topic content grounds the generated learning plan
5. **Suggested jobs** (retrieval-driven) — candidate's current skill profile is queried against `job_postings_index` to surface adjacent roles as alternatives

## Setup

```bash
pip install -r requirements.txt
python src/ingestion.py     # builds both FAISS indexes
python src/test_retrieval.py    # sanity-checks retrieval quality
```