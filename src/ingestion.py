"""
ingestion.py

Builds two separate FAISS indexes:
  1. job_postings_index  -- one vector per job description row
  2. roadmap_index          -- one vector per roadmap topic markdown file

Kept as two indexes (not one mixed index) so retrieval never accidentally
returns a roadmap chunk when searching for market context, or vice versa.
See chains.py for how each index gets queried.

"""

import os
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

JOB_CSV_PATH = "data/raw/job_descriptions_filtered.csv"
ROADMAPS_DIR = "data/roadmaps"

JOB_INDEX_PATH = "vectorstore/job_postings_index"
ROADMAP_INDEX_PATH = "vectorstore/roadmap_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Step 1: Build Documents (chunk) for job postings
# ---------------------------------------------------------------------------

def load_job_postings(csv_path: str) -> list[Document]:
    """
    One row = one chunk (one Document). No text splitting -- each posting
    is already a coherent, self-contained unit, so splitting it further
    would separate skills from context and hurt retrieval quality.
    """
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        chunk_text = (
            f"Job Title: {row['Job Title']}\n"
            f"Role: {row['Role']}\n"
            f"Experience Required: {row['Experience']}\n"
            f"Qualifications: {row['Qualifications']}\n"
            f"Skills: {row['skills']}\n"
            f"Description: {row['Job Description']}\n"
            f"Responsibilities: {row['Responsibilities']}"
        )

        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": "job_posting",
                "job_id": str(row["Job Id"]),
                "job_title": row["Job Title"],
                "role": row["Role"],
            },
        )
        documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# Step 2: Build Documents (chunk) for roadmap topic files
# ---------------------------------------------------------------------------

def load_roadmap_topics(roadmaps_dir: str) -> list[Document]:
    """
    One topic markdown file = one chunk (one Document). Each file under
    <roadmap>/content/ already represents a single coherent topic node,
    so (as with job postings) no further splitting is needed here.

    Empty or near-empty stub files are skipped -- some roadmap nodes in
    the source repo have no written content yet, and indexing an empty
    chunk would just add retrieval noise.
    """
    documents = []
    roadmaps_path = Path(roadmaps_dir)

    for roadmap_folder in sorted(roadmaps_path.iterdir()):
        if not roadmap_folder.is_dir():
            continue

        content_dir = roadmap_folder / "content"
        if not content_dir.exists():
            continue

        for md_file in sorted(content_dir.glob("*.md")):
            text = md_file.read_text(encoding="utf-8").strip()

            # Skip stub files with little to no real content
            if len(text) < 30:
                continue

            # filename looks like "docker@a1b2.md" -- strip the @id suffix
            topic_name = md_file.stem.split("@")[0].replace("-", " ")

            doc = Document(
                page_content=text,
                metadata={
                    "source": "roadmap",
                    "roadmap": roadmap_folder.name,
                    "topic": topic_name,
                },
            )
            documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# Step 3: Embed + Index
# ---------------------------------------------------------------------------

def build_and_save_index(documents: list[Document], save_path: str, embeddings) -> None:
    if not documents:
        raise ValueError(f"No documents to index for {save_path} -- check source data.")

    index = FAISS.from_documents(documents, embeddings)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    index.save_local(save_path)
    print(f"Saved {len(documents)} vectors to {save_path}")


def main():
    print("Loading embedding model (first run downloads it, ~90MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("\n--- Job postings ---")
    job_docs = load_job_postings(JOB_CSV_PATH)
    print(f"Loaded {len(job_docs)} job posting chunks")
    build_and_save_index(job_docs, JOB_INDEX_PATH, embeddings)

    print("\n--- Roadmap topics ---")
    roadmap_docs = load_roadmap_topics(ROADMAPS_DIR)
    print(f"Loaded {len(roadmap_docs)} roadmap topic chunks")
    build_and_save_index(roadmap_docs, ROADMAP_INDEX_PATH, embeddings)

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()