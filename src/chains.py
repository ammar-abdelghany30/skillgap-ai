"""
chains.py

LCEL chains for the SkillGap-AI pipeline.

Chain 1: CV extraction        -- CV text -> CVExtractionResult
Chain 2: JD extraction        -- JD text -> JDExtractionResult

Both chains follow the same pattern: prompt | llm | parser, wrapped with
OutputFixingParser so a malformed LLM response gets one automatic retry
(re-prompted with the parsing error) instead of crashing the pipeline.
This is the core "output parser" lesson -- forcing the LLM into a schema
and handling the case where it doesn't comply on the first try.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain_classic.output_parsers import OutputFixingParser
from langchain_mistralai import ChatMistralAI

from schemas import CVExtractionResult, JDExtractionResult


def get_llm(temperature: float = 0):
    # temperature=0 -- we want consistent, deterministic extraction,
    # not creative variation, since this feeds structured parsing.
    #
    # mistral-small-latest -- runs on Mistral's free "Experiment" tier
    # (rate-limited, no cost). Good enough for structured extraction
    # tasks like this; reserve mistral-large-latest for anything needing
    # deeper reasoning, if your rate limit allows it.
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

    cv_chain = build_cv_extraction_chain()
    result = cv_chain.invoke({"cv_text": sample_cv})
    print(result.model_dump_json(indent=2))