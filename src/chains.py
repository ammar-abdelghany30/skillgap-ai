from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel

from langchain_classic.output_parsers import OutputFixingParser
from langchain_mistralai import ChatMistralAI

from schemas import CVExtractionResult, JDExtractionResult


def get_llm(temperature: float = 0):

    return ChatMistralAI(model="mistral-small-latest", temperature=temperature)


# ---------------------------------------------------------------------------
# Chain 1: CV extraction
# ---------------------------------------------------------------------------

def build_cv_extraction_chain(llm=None):
    llm = llm or get_llm()

    base_parser = PydanticOutputParser(pydantic_object=CVExtractionResult)

    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

    prompt = ChatPromptTemplate.from_template(
        "You are an expert CV/resume parser.\n"
        "Extract structured information from the CV below.\n"
        "Only extract what is explicitly stated or clearly implied -- "
        "do not invent details that aren't in the text.\n\n"
        "CV:\n{cv_text}\n\n"
        "{format_instructions}"
    ).partial(format_instructions=base_parser.get_format_instructions())

    return prompt | llm | fixing_parser


# ---------------------------------------------------------------------------
# Chain 2: JD extraction
# ---------------------------------------------------------------------------

def build_jd_extraction_chain(llm=None):
    llm = llm or get_llm()

    base_parser = PydanticOutputParser(pydantic_object=JDExtractionResult)
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

    prompt = ChatPromptTemplate.from_template(
        "You are an expert job description parser.\n"
        "Extract structured requirements from the job description below.\n"
        "Mark a requirement as mandatory only if it's stated as required/must-have; "
        "otherwise mark it as preferred.\n\n"
        "Job Description:\n{jd_text}\n\n"
        "{format_instructions}"
    ).partial(format_instructions=base_parser.get_format_instructions())

    return prompt | llm | fixing_parser


# ---------------------------------------------------------------------------
# Chain 3: Gap analysis
# ---------------------------------------------------------------------------

class GapAnalysisResult(BaseModel):
    """
    Output of Chain 3. Deliberately narrower than the final
    CareerGapAnalysis in schemas.py -- this chain only handles the
    comparison step. Roadmap generation and job suggestions (which need
    RAG retrieval) are separate chains layered on top later.
    """
    current_skills: List[str] = Field(
        description="Skills the candidate has that also appear in the job requirements"
    )
    missing_skills: List[str] = Field(
        description="Required or preferred skills from the JD that the candidate does not have"
    )
    match_percentage: float = Field(
        description="Rough percentage of required skills the candidate covers (0-100)"
    )
    overall_feedback: str = Field(
        description="1-2 sentence overall assessment of how well the candidate fits the role"
    )


def build_gap_analysis_chain(llm=None):
    llm = llm or get_llm()

    base_parser = PydanticOutputParser(pydantic_object=GapAnalysisResult)
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

    prompt = ChatPromptTemplate.from_template(
        "Compare the candidate's extracted skills against the job requirements "
        "below, and produce a gap analysis.\n\n"
        "Candidate skills (JSON):\n{candidate_skills}\n\n"
        "Job requirements (JSON):\n{job_requirements}\n\n"
        "Match skills by meaning, not just exact string match (e.g. 'JS' and "
        "'JavaScript' are the same skill). Mandatory requirements matter more "
        "than preferred ones for the match percentage.\n\n"
        "{format_instructions}"
    ).partial(format_instructions=base_parser.get_format_instructions())

    return prompt | llm | fixing_parser


# ---------------------------------------------------------------------------
# Full pipeline: Chain 1 -> Chain 2 -> Chain 3 wired as one LCEL Runnable
# ---------------------------------------------------------------------------

def build_full_pipeline(llm=None):
    """
    Wires all three chains into a single composed Runnable
    Input:  {"cv_text": "...", "jd_text": "..."}
    Output: GapAnalysisResult
    """
    llm = llm or get_llm()

    cv_chain = build_cv_extraction_chain(llm)
    jd_chain = build_jd_extraction_chain(llm)
    gap_chain = build_gap_analysis_chain(llm)

    extraction_stage = RunnableParallel(
        cv_result=(lambda x: {"cv_text": x["cv_text"]}) | cv_chain,
        jd_result=(lambda x: {"jd_text": x["jd_text"]}) | jd_chain,
    )


    def reshape_for_gap_chain(extraction_results: dict) -> dict:
        return {
            "candidate_skills": extraction_results["cv_result"].model_dump_json(),
            "job_requirements": extraction_results["jd_result"].model_dump_json(),
        }

    full_pipeline = extraction_stage | reshape_for_gap_chain | gap_chain
    return full_pipeline


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("MISTRAL_API_KEY"):
        print("Set MISTRAL_API_KEY (in your .env file) to run this smoke test.")
        raise SystemExit(0)

    sample_cv = """
    Ammar Abdelghany
    Senior year Computer and Systems Engineering student, Ain Shams University.
    Interned as an AI/LLM Engineering Trainee at Tips Hindawi and as a
    Software Engineering Trainee at Fuzetek. Skilled in Python, SQL, Docker,
    and has built a distributed marketplace project using MariaDB Spider
    Engine sharding.
    """

    sample_jd = """
    We are hiring a Junior Backend Engineer. Requirements: strong Python
    skills, experience with SQL databases, familiarity with Docker and
    containerized deployments. Experience with Kubernetes and CI/CD
    pipelines is a plus. 0-2 years of experience welcome.
    """

    print("Running full pipeline (Chain 1 + Chain 2 in parallel -> Chain 3)...\n")
    pipeline = build_full_pipeline()
    result = pipeline.invoke({"cv_text": sample_cv, "jd_text": sample_jd})
    print(result.model_dump_json(indent=2))