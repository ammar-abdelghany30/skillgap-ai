"""
test_retrieval.py

Standalone sanity check for both FAISS indexes. Run this any time you
rebuild the indexes (after re-running ingestion.py) to quickly confirm
retrieval is still returning relevant results before wiring chains.py
against them.

This is NOT part of the app pipeline -- it's a manual verification tool.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

JOB_INDEX_PATH = "vectorstore/job_postings_index"
ROADMAP_INDEX_PATH = "vectorstore/roadmap_index"


def run_query(index, query: str, k: int = 3):
    print(f"\nQuery: \"{query}\"")
    print("-" * 60)
    results = index.similarity_search(query, k=k)
    for i, r in enumerate(results, 1):
        print(f"[{i}] metadata: {r.metadata}")
        print(f"    {r.page_content[:120]}...")
    print()


def main():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Loading indexes...")
    job_index = FAISS.load_local(
        JOB_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    roadmap_index = FAISS.load_local(
        ROADMAP_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )

    print("\n" + "=" * 60)
    print("JOB POSTINGS INDEX")
    print("=" * 60)
    run_query(job_index, "backend engineer requirements")
    run_query(job_index, "data scientist with machine learning experience")

    print("\n" + "=" * 60)
    print("ROADMAP INDEX")
    print("=" * 60)
    run_query(roadmap_index, "how to learn Docker")
    run_query(roadmap_index, "SQL for data analysis")

    print("\nSanity check complete. Review the results above --")
    print("each query's top results should clearly match its topic.")


if __name__ == "__main__":
    main()