"""
chains.py

LCEL chains for the SkillGap-AI pipeline.

Chain 1: CV extraction       -- CV text -> CVExtractionResult
Chain 2: JD extraction       -- JD text -> JDExtractionResult
Chain 3: Gap analysis        -- compares Chain 1 + Chain 2 output
Chain 4: Roadmap generation  -- missing skills -> RoadmapResult            (RAG: roadmap_index)
Chain 5: Suggested jobs      -- current skills -> SuggestedJobsResult      (RAG: job_postings_index)


"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.output_parsers import OutputFixingParser
from langchain_mistralai import ChatMistralAI

from schemas import CVExtractionResult, JDExtractionResult, GapAnalysisResult


def get_llm(temperature: float = 0):
    # temperature=0 -- we want consistent, deterministic extraction,
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
    Wires all three chains into a single composed Runnable, rather than
    just calling three functions in sequence from app.py. This is the
    actual "chain architecture" deliverable: one Runnable that takes raw
    CV + JD text and returns a GapAnalysisResult, with the intermediate
    extraction steps run in parallel (they don't depend on each other)
    before being fed into the comparison step.

    Input:  {"cv_text": "...", "jd_text": "..."}
    Output: GapAnalysisResult
    """
    llm = llm or get_llm()

    cv_chain = build_cv_extraction_chain(llm)
    jd_chain = build_jd_extraction_chain(llm)
    gap_chain = build_gap_analysis_chain(llm)

    # RunnableParallel runs Chain 1 and Chain 2 concurrently -- they're
    # independent (CV extraction doesn't need the JD, and vice versa),
    # so there's no reason to run them sequentially and waste time.
    extraction_stage = RunnableParallel(
        cv_result=(lambda x: {"cv_text": x["cv_text"]}) | cv_chain,
        jd_result=(lambda x: {"jd_text": x["jd_text"]}) | jd_chain,
    )

    # Reshape the two extraction results into the input shape gap_chain
    # expects, then feed into Chain 3. This whole thing -- extraction
    # stage piped into a reshape step piped into gap_chain -- is ONE
    # composed Runnable end to end.
    def reshape_for_gap_chain(extraction_results: dict) -> dict:
        return {
            "candidate_skills": extraction_results["cv_result"].model_dump_json(),
            "job_requirements": extraction_results["jd_result"].model_dump_json(),
        }

    full_pipeline = extraction_stage | reshape_for_gap_chain | gap_chain
    return full_pipeline


# ---------------------------------------------------------------------------
# Vector store loading (needed for Chains 4 and 5)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
JOB_INDEX_PATH = "vectorstore/job_postings_index"
ROADMAP_INDEX_PATH = "vectorstore/roadmap_index"


def load_vectorstores():
    """
    Loads the two FAISS indexes built by ingestion.py. Call this once
    (not per-request) and reuse the returned indexes across chain calls.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    job_index = FAISS.load_local(
        JOB_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    roadmap_index = FAISS.load_local(
        ROADMAP_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return job_index, roadmap_index


# ---------------------------------------------------------------------------
# Chain 4: Roadmap generation (RAG -- queries roadmap_index)
# ---------------------------------------------------------------------------

class RoadmapStep(BaseModel):
    skill: str = Field(description="The missing skill this step addresses")
    action: str = Field(description="Concrete next action to learn this skill")
    grounded_in_source: Optional[str] = Field(
        default=None,
        description="Which roadmap/topic this recommendation was grounded in "
                    "(e.g. 'backend/docker')"
    )


class RoadmapResult(BaseModel):
    steps: List[RoadmapStep] = Field(description="Ordered learning roadmap")


def build_roadmap_chain(roadmap_index, llm=None, k: int = 2):
    """
    RAG chain: for each missing skill, retrieves relevant roadmap content
    from roadmap_index BEFORE calling the LLM, so recommendations are
    grounded in real roadmap.sh content rather than invented from the
    model's training data alone.
    """
    llm = llm or get_llm()

    base_parser = PydanticOutputParser(pydantic_object=RoadmapResult)
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

    def retrieve_context(input_dict: dict) -> dict:
        missing_skills = input_dict["missing_skills"]
        context_blocks = []
        for skill in missing_skills:
            docs = roadmap_index.similarity_search(skill, k=k)
            snippets = "\n".join(
                f"  - [{d.metadata.get('roadmap')}/{d.metadata.get('topic')}] "
                f"{d.page_content[:300]}"
                for d in docs
            )
            context_blocks.append(f"Missing skill: {skill}\nRetrieved content:\n{snippets}")
        return {
            "missing_skills_text": ", ".join(missing_skills),
            "retrieved_context": "\n\n".join(context_blocks),
        }

    prompt = ChatPromptTemplate.from_template(
        "The candidate is missing these skills: {missing_skills_text}\n\n"
        "Relevant learning content retrieved for each missing skill:\n"
        "{retrieved_context}\n\n"
        "For each missing skill, write ONE concrete next learning action "
        "grounded in the retrieved content above (not generic advice). "
        "Set grounded_in_source to the [roadmap/topic] tag the content came from.\n\n"
        "{format_instructions}"
    ).partial(format_instructions=base_parser.get_format_instructions())

    return RunnableLambda(retrieve_context) | prompt | llm | fixing_parser


# ---------------------------------------------------------------------------
# Chain 5: Suggested jobs (RAG -- queries job_postings_index)
# ---------------------------------------------------------------------------

class SuggestedJobsResult(BaseModel):
    suggested_jobs: List[str] = Field(
        description="2-4 adjacent job titles the candidate is a strong match for"
    )
    reasoning: str = Field(
        description="Brief explanation of why these roles fit the candidate's current skills"
    )


def build_suggested_jobs_chain(job_index, llm=None, k: int = 6):
    """
    RAG chain: embeds the candidate's current skills, retrieves the
    nearest real job postings from job_postings_index, then asks the LLM
    to synthesize suggested adjacent roles grounded in those real
    postings rather than guessing role titles from memory.
    """
    llm = llm or get_llm()

    base_parser = PydanticOutputParser(pydantic_object=SuggestedJobsResult)
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

    def retrieve_similar_roles(input_dict: dict) -> dict:
        skills_text = ", ".join(input_dict["current_skills"])
        docs = job_index.similarity_search(skills_text, k=k)

        # Dedup by role -- the underlying dataset has many near-duplicate
        # postings per role (see Day 1 retrieval test), so without this
        # the LLM would just see the same role repeated several times.
        seen_roles = []
        context_lines = []
        for d in docs:
            role = d.metadata.get("role")
            if role and role not in seen_roles:
                seen_roles.append(role)
                context_lines.append(f"- {role}: {d.page_content[:150]}")

        return {
            "skills_text": skills_text,
            "retrieved_roles": "\n".join(context_lines),
        }

    prompt = ChatPromptTemplate.from_template(
        "Candidate's current skills: {skills_text}\n\n"
        "Real job postings retrieved as similar matches:\n{retrieved_roles}\n\n"
        "Based only on the roles shown above, suggest 2-4 adjacent job titles "
        "this candidate is already a strong match for.\n\n"
        "{format_instructions}"
    ).partial(format_instructions=base_parser.get_format_instructions())

    return RunnableLambda(retrieve_similar_roles) | prompt | llm | fixing_parser


# ---------------------------------------------------------------------------
# End-to-end runner: ties all 5 chains together for a single CV + JD input
# ---------------------------------------------------------------------------

def run_end_to_end(cv_text: str, jd_text: str, llm=None) -> dict:
    """
    Runs the complete pipeline: Chains 1-3 (extraction + gap analysis,
    no RAG) followed by Chains 4-5 (roadmap + suggested jobs, RAG-grounded).
    """
    llm = llm or get_llm()
    job_index, roadmap_index = load_vectorstores()

    gap_pipeline = build_full_pipeline(llm)
    gap_result = gap_pipeline.invoke({"cv_text": cv_text, "jd_text": jd_text})

    roadmap_chain = build_roadmap_chain(roadmap_index, llm)
    roadmap_result = roadmap_chain.invoke({"missing_skills": gap_result.missing_skills})

    jobs_chain = build_suggested_jobs_chain(job_index, llm)
    jobs_result = jobs_chain.invoke({"current_skills": gap_result.current_skills})

    return {
        "current_skills": gap_result.current_skills,
        "missing_skills": gap_result.missing_skills,
        "match_percentage": gap_result.match_percentage,
        "overall_feedback": gap_result.overall_feedback,
        "recommended_roadmap": [step.model_dump() for step in roadmap_result.steps],
        "suggested_jobs": jobs_result.suggested_jobs,
        "suggested_jobs_reasoning": jobs_result.reasoning,
    }



    # Quick manual smoke test

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