"""

Extracts text from your actual CV (PDF) and runs it through the full
5-chain pipeline against a job description you provide, printing the
final combined result.

"""

import sys
from pathlib import Path

sys.path.insert(0, "src")

from dotenv import load_dotenv
from pypdf import PdfReader

from chains import run_end_to_end, get_llm

load_dotenv()

CV_PDF_PATH = "data/AmmarAbdelghanyCV.pdf"


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def main():
    import os

    if not os.getenv("MISTRAL_API_KEY"):
        print("Set MISTRAL_API_KEY in your .env file first.")
        return

    if not Path(CV_PDF_PATH).exists():
        print(f"CV not found at {CV_PDF_PATH}")
        print("Place your CV PDF there, or edit CV_PDF_PATH at the top of this script.")
        return

    print(f"Extracting text from {CV_PDF_PATH} ...")
    cv_text = extract_text_from_pdf(CV_PDF_PATH)
    print(f"Extracted {len(cv_text)} characters.\n")
    print("--- First 300 characters (sanity check) ---")
    print(cv_text[:300])
    print("---\n")

    # Edit this to whatever role you want to test against
    jd_text = """
    We are hiring a Junior Machine Learning Engineer. Requirements:
    strong Python skills, experience with LLM/RAG pipelines, familiarity
    with vector databases (FAISS or similar), and prompt engineering
    experience. Experience with LangChain and structured output parsing
    is a strong plus. SQL knowledge required. 0-2 years experience welcome.
    """

    print("Running full 5-chain pipeline (this calls the LLM several times "
          "and queries both FAISS indexes -- may take 15-30 seconds)...\n")

    llm = get_llm()
    result = run_end_to_end(cv_text, jd_text, llm)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()