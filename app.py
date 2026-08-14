"""
app.py
"""

import sys
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).parent / "src"))
from chains import run_end_to_end, get_llm, load_vectorstores

load_dotenv()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_analysis_context" not in st.session_state:
    st.session_state.last_analysis_context = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

ROLES_WITH_GROUNDING = ["Backend Developer", "Frontend Developer", "Data Analyst", "Data Scientist"]
TARGET_ROLES = ROLES_WITH_GROUNDING + ["DevOps Engineer", "AI Engineer", "Machine Learning Engineer", "Other"]


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

# ----------------UI---------------------

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
    st.caption("__or__")
    cv_text_input = st.text_area("Paste CV text instead", height=150)

with col2:
    st.subheader("2. Target Job")
    target_role = st.selectbox("Target role (used for market grounding)", TARGET_ROLES)
    jd_text = st.text_area("Paste the job description", height=220)

run_clicked = st.button("Analyze", type="primary", disabled=not ((cv_file or cv_text_input) and jd_text))

if run_clicked:
    st.session_state.chat_history = []
    if cv_text_input.strip():
        cv_text = cv_text_input.strip()
    else:
        with st.spinner("Extracting CV text..."):
            cv_text = extract_text_from_pdf(cv_file)

    if len(cv_text) < 30:
        st.error("Could not extract meaningful text from this PDF. "
                  "It may be a scanned/image-only PDF -- try a text-based CV export instead.")
        st.stop()

    with st.spinner("Running the analysis pipeline (5 chains + retrieval)... "
                     "this can take 15-30 seconds"):
        llm = get_cached_llm()
        job_index, roadmap_index = get_cached_vectorstores()

        from chains import (
            build_full_pipeline,
            ground_missing_skills,
            build_roadmap_chain,
            build_suggested_jobs_chain,
            compute_match_percentage,
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

        gap_result.match_percentage = compute_match_percentage(gap_result, jd_result)

        grounded = None
        if target_role in ROLES_WITH_GROUNDING:
            grounded = ground_missing_skills(
                missing_skills=gap_result.missing_skills,
                target_job_title=gap_result.target_job_title or target_role,
                job_index=job_index,
            )

        roadmap_result = None
        if gap_result.missing_skills:
            roadmap_result = build_roadmap_chain(roadmap_index, llm).invoke(
                {"missing_skills": gap_result.missing_skills}
            )

        all_candidate_skills = [skill.name for skill in cv_result.skills]

        jobs_result = build_suggested_jobs_chain(job_index, llm).invoke(
            {"current_skills": all_candidate_skills}
        )

    # --- Save everything needed for display into session_state so it ---
    # --- survives reruns triggered by the chatbot below ---
    st.session_state.analysis_results = {
        "gap_result": gap_result,
        "roadmap_result": roadmap_result,
        "jobs_result": jobs_result,
        "grounded": grounded,
        "target_role": target_role,
    }

    st.session_state.last_analysis_context = (
        f"Target role: {gap_result.target_job_title}\n"
        f"Current skills: {', '.join(gap_result.current_skills)}\n"
        f"Missing skills: {', '.join(gap_result.missing_skills)}\n"
        f"Match: {gap_result.match_percentage}%"
    )

# ---------------------------------------------------------------------
# Display section -- runs on EVERY rerun (not just when Analyze was
# clicked), driven by whether saved results exist in session_state.
# This is what keeps results visible while chatting below.
# ---------------------------------------------------------------------
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    gap_result = results["gap_result"]
    roadmap_result = results["roadmap_result"]
    jobs_result = results["jobs_result"]
    grounded = results["grounded"]
    target_role = results["target_role"]

    st.success("Analysis complete.")

    perfect_match = gap_result.match_percentage >= 100 or not gap_result.missing_skills

    st.subheader("Match Summary")
    m1, m2 = st.columns([1, 3])
    with m1:
        st.metric("Match", f"{gap_result.match_percentage:.0f}%")
    with m2:
        if perfect_match:
            st.write("🎯 This job fits you perfectly — you meet all the listed requirements.")
        else:
            st.write(gap_result.overall_feedback)

    st.subheader("Skills")
    if perfect_match:
        st.markdown("**Current Skills**")
        for s in gap_result.current_skills:
            st.write(f"- {s}")
    else:
        skill_col1, skill_col2 = st.columns(2)
        with skill_col1:
            st.markdown("**Current Skills**")
            for s in gap_result.current_skills:
                st.write(f"- {s}")
        with skill_col2:
            st.markdown("**Missing Skills**")
            for s in gap_result.missing_skills:
                st.write(f"- {s}")

    if not perfect_match:
        if target_role in ROLES_WITH_GROUNDING:
            st.subheader("Market Evidence for Missing Skills")
            st.caption("How often each missing skill appears in similar real job postings.")

            for g in grounded:
                pct = (
                        g.supporting_postings_count / g.total_postings_checked * 100
                ) if g.total_postings_checked else 0

                st.write(
                    f"**{g.skill}** -- appears in "
                    f"{g.supporting_postings_count}/"
                    f"{g.total_postings_checked} postings ({pct:.0f}%)"
                )

                if g.example_job_ids:
                    job_ids_text = ", ".join(g.example_job_ids)
                    st.caption(f"Example job IDs: {job_ids_text}")
        else:
            st.subheader("Market evidence isn't available yet for this role.")

        st.subheader("Recommended Roadmap")
        for step in roadmap_result.steps:
            with st.expander(f"{step.skill}"):
                st.write(step.action)
                if step.grounded_in_source:
                    st.caption(f"Source: {step.grounded_in_source}")

    st.subheader("Suggested Alternative Roles")
    st.write(", ".join(jobs_result.suggested_jobs))
    st.caption(jobs_result.reasoning)


st.divider()
chat_header_col1, chat_header_col2 = st.columns([4, 1])
with chat_header_col1:
    st.subheader("Career Coach Chat")
with chat_header_col2:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.analysis_results:
    placeholder = "e.g. 'Suggest a project idea covering my missing skills'"
else:
    placeholder = "e.g. 'Tips to make my CV more ATS-friendly?'"

user_question = st.chat_input(placeholder)

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    from chains import build_cv_advisor_chain
    advisor_chain = build_cv_advisor_chain(get_cached_llm())

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[:-1]
    )
    context_text = st.session_state.last_analysis_context or "No analysis run yet."

    with st.spinner("Thinking..."):
        response = advisor_chain.invoke({
            "context": context_text,
            "history": history_text,
            "question": user_question,
        })

    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
    st.rerun()