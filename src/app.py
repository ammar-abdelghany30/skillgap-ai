"""
app.py

Streamlit UI for SkillGap-AI. This is the actual application entry point --
run with `streamlit run app.py`, not `python app.py`.

Keeps all pipeline logic in chains.py; this file is presentation only:
collect input (CV upload + JD text + target role), call run_end_to_end(),
render the structured result.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from chains import run_end_to_end, get_llm, load_vectorstores

load_dotenv()

TARGET_ROLES = ["Backend Developer", "Frontend Developer", "Data Analyst", "Data Scientist" , "Other"]


@st.cache_resource
def get_cached_llm():
    return get_llm()


@st.cache_resource
def get_cached_vectorstores():
    return load_vectorstores()


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SkillGap-AI", layout="wide")
st.title("SkillGap-AI")
st.markdown("""
<p style="font-size:20px; color:#B0B0B0;">
RAG-powered career coach: upload your CV, paste a target job description,
get a grounded skill-gap analysis and learning roadmap.
</p>
""", unsafe_allow_html=True)

if not os.getenv("MISTRAL_API_KEY"):
    st.error("MISTRAL_API_KEY not found. Add it to your .env file before running.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Your CV")
    cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])

with col2:
    st.subheader("2. Target Job")
    target_role = st.selectbox("Target role (used for market grounding)", TARGET_ROLES)
    jd_text = st.text_area("Paste the job description", height=220)

run_clicked = st.button("Analyze", type="primary", disabled=not (cv_file and jd_text))

if run_clicked:
    with st.spinner("Extracting CV text..."):
        cv_text = extract_text_from_pdf(cv_file)

    if len(cv_text) < 30:
        st.error("Couldn't extract meaningful text from this PDF. "
                  "It may be a scanned/image-only PDF -- try a text-based CV export instead.")
        st.stop()

    with st.spinner("Running the analysis pipeline (5 chains + retrieval)... "
                     "this can take 15-30 seconds"):
        llm = get_cached_llm()
        job_index, roadmap_index = get_cached_vectorstores()

        # run_end_to_end reloads vectorstores internally by default; here we
        # want to reuse the cached ones, so we call the pieces directly
        # instead of the convenience wrapper.
        from chains import (
            build_full_pipeline,
            ground_missing_skills,
            build_roadmap_chain,
            build_suggested_jobs_chain,
        )

        gap_pipeline = build_full_pipeline(llm)

        pipeline_output = gap_pipeline.invoke(
            {
                "cv_text": cv_text,
                "jd_text": jd_text,
            }
        )

        cv_result = pipeline_output["cv_result"]
        jd_result = pipeline_output["jd_result"]
        gap_result = pipeline_output["gap_result"]

        grounded = ground_missing_skills(
            missing_skills=gap_result.missing_skills,
            target_job_title=gap_result.target_job_title or target_role,
            job_index=job_index,
        )

        roadmap_result = build_roadmap_chain(roadmap_index, llm).invoke(
            {"missing_skills": gap_result.missing_skills}
        )

        all_candidate_skills = [
            skill.name
            for skill in cv_result.skills
        ]

        jobs_result = build_suggested_jobs_chain(job_index, llm).invoke(
            {
                "current_skills": all_candidate_skills
            }
        )

    st.success("Analysis complete.")

    # --- Match summary ---
    st.subheader("Match Summary")
    m1, m2 = st.columns([1, 3])
    with m1:
        st.metric("Match", f"{gap_result.match_percentage:.0f}%")
    with m2:
        st.write(gap_result.overall_feedback)

    # --- Skills table ---
    st.subheader("Skills")
    skill_col1, skill_col2 = st.columns(2)
    with skill_col1:
        st.markdown("**✅ Current Skills**")
        for s in gap_result.current_skills:
            st.write(f"- {s}")
    with skill_col2:
        st.markdown("**❌ Missing Skills**")
        for s in gap_result.missing_skills:
            st.write(f"- {s}")

    # --- Grounded evidence (Day 3 feature -- the visible "proof") ---
    st.subheader("Market Evidence for Missing Skills")
    st.caption("How often each missing skill actually appears in similar real job postings "
               "(retrieved from a 600-posting job market index), not just an LLM guess.")
    for g in grounded:
        pct = (g.supporting_postings_count / g.total_postings_checked * 100) \
            if g.total_postings_checked else 0
        st.write(
            f"**{g.skill}** — appears in {g.supporting_postings_count}/"
            f"{g.total_postings_checked} similar postings ({pct:.0f}%)"
        )
        if g.example_job_ids:
            st.caption(f"Example job IDs: {', '.join(g.example_job_ids)}")

    # --- Roadmap ---
    st.subheader("Recommended Roadmap")
    for step in roadmap_result.steps:
        with st.expander(f"📘 {step.skill}"):
            st.write(step.action)
            if step.grounded_in_source:
                st.caption(f"Source: {step.grounded_in_source}")

    # --- Suggested jobs ---
    st.subheader("Suggested Alternative Roles")
    st.write(", ".join(jobs_result.suggested_jobs))
    st.caption(jobs_result.reasoning)