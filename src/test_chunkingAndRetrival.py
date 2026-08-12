"""
inspect_and_test_chunks.py

Two things in one script:
1. Dump real sample chunk text from specific roadmaps (for manual inspection)
2. Run realistic skill queries against roadmap_index to sanity-check that
   retrieval returns logically relevant content -- this is the same kind
   of check as test_retrieval.py, just targeted at the roadmaps you're
   about to lean on for DevOps/AI Engineer/ML Engineer grounding.

Run this before wiring new target roles into app.py, so you know the
underlying roadmap content is actually good before trusting it.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ROADMAP_INDEX_PATH = "vectorstore/roadmap_index"


def show_sample_chunks(index, roadmap_name: str, n: int = 4):
    docs = [
        d for d in index.docstore._dict.values()
        if d.metadata.get("roadmap") == roadmap_name
    ]
    print(f"\n{'=' * 60}")
    print(f"ROADMAP: {roadmap_name}  ({len(docs)} total chunks, showing {min(n, len(docs))})")
    print("=" * 60)
    for d in docs[:n]:
        print(f"\n--- topic: {d.metadata.get('topic')} ---")
        print(d.page_content[:400])


def run_query(index, query: str, k: int = 3):
    print(f"\nQuery: \"{query}\"")
    print("-" * 60)
    results = index.similarity_search(query, k=k)
    for i, r in enumerate(results, 1):
        print(f"[{i}] roadmap={r.metadata.get('roadmap')}  topic={r.metadata.get('topic')}")
        print(f"    {r.page_content[:150]}...")


def main():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Loading roadmap index...")
    roadmap_index = FAISS.load_local(
        ROADMAP_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )

    # --- Part 1: raw sample chunks from the three roadmaps in question ---
    for roadmap_name in ["ai-data-scientist", "ai-engineer", "backend","machine-learning"]:
        show_sample_chunks(roadmap_index, roadmap_name, n=3)

    # --- Part 2: logical retrieval test with realistic missing-skill queries ---
    print("\n\n" + "=" * 60)
    print("RETRIEVAL SANITY CHECK -- realistic skill queries")
    print("=" * 60)

    test_queries = [
        "MLOps model deployment",       # should hit ai-engineer / machine-learning
        "vector databases and embeddings",  # should hit ai-engineer
        "CI/CD pipelines",              # should hit devops (and maybe backend)
        "Kubernetes container orchestration",  # should hit devops
        "REST API design",              # should hit backend
        "prompt engineering",           # should hit ai-engineer specifically
    ]
    for q in test_queries:
        run_query(roadmap_index, q)

    print("\n\nReview above: for each query, do the top results' `roadmap` and "
          "`topic` fields actually match what you'd expect a human to associate "
          "with that query? That's the real pass/fail here, not just 'no errors'.")


if __name__ == "__main__":
    main()