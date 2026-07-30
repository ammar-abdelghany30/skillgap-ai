"""
schemas.py

Pydantic models defining the structured outputs for every stage of the
AI Career Coach pipeline.

Pipeline:
1. CV Extraction
2. Job Description Extraction
3. Career Gap Analysis
4. Learning Roadmap Generation
5. Career Recommendations

These schemas are used with LangChain's PydanticOutputParser so the LLM
returns predictable JSON instead of free-form text.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Shared Models
# ============================================================================

class SkillEntry(BaseModel):
    """Represents one technical or soft skill."""

    name: str = Field(
        description="Skill name (e.g. Python, Docker, Communication)"
    )

    category: str = Field(
        description=(
            "Skill category such as Programming Language, "
            "Framework, Tool, Database, Cloud, Soft Skill, etc."
        )
    )

    proficiency_hint: Optional[str] = Field(
        default=None,
        description=(
            "Evidence of proficiency if inferable from the CV "
            "(e.g. 'Built 3 projects', 'Advanced', 'Internship experience')."
        ),
    )


class ExperienceEntry(BaseModel):
    """One work experience or internship."""

    company: str = Field(description="Company name")

    role: str = Field(description="Job title")

    duration: Optional[str] = Field(
        default=None,
        description="Employment duration if available"
    )

    summary: Optional[str] = Field(
        default=None,
        description="Short summary of responsibilities"
    )


class ProjectEntry(BaseModel):
    """One project listed in the CV."""

    name: str = Field(description="Project name")

    technologies: List[str] = Field(
        description="Technologies used in the project"
    )

    description: Optional[str] = Field(
        default=None,
        description="Brief description of the project"
    )


class EducationEntry(BaseModel):
    """Educational qualification."""

    degree: str = Field(
        description="Degree name"
    )

    institution: str = Field(
        description="University or institution"
    )

    gpa: Optional[float] = Field(
        default=None,
        description="GPA if available"
    )

    graduation_year: Optional[int] = Field(
        default=None,
        description="Graduation year if mentioned"
    )


class CertificationEntry(BaseModel):
    """Professional certification."""

    name: str = Field(
        description="Certification name"
    )

    issuer: Optional[str] = Field(
        default=None,
        description="Organization issuing the certificate"
    )

class ResponsibilityEntry(BaseModel):
    """
    Represents one responsibility extracted from the job description.
    """

    description: str = Field(
        description="A single job responsibility or duty."
    )


# ============================================================================
# Chain 1 : CV Extraction
# ============================================================================

class CVExtractionResult(BaseModel):
    """
    Structured information extracted from a candidate CV.
    """

    candidate_name: Optional[str] = Field(
        default=None,
        description="Candidate's full name"
    )

    candidate_current_role: Optional[str] = Field(
        default=None,
        description="Most recent or current job title"
    )

    years_of_experience: Optional[float] = Field(
        default=None,
        description="Approximate total years of professional experience"
    )

    education: List[EducationEntry] = Field(
        default_factory=list,
        description="Educational background"
    )

    experiences: List[ExperienceEntry] = Field(
        default_factory=list,
        description="Professional experiences and internships"
    )

    skills: List[SkillEntry] = Field(
        default_factory=list,
        description="Technical and soft skills"
    )

    projects: List[ProjectEntry] = Field(
        default_factory=list,
        description="Projects completed by the candidate"
    )

    certifications: List[CertificationEntry] = Field(
        default_factory=list,
        description="Professional certifications"
    )

    languages: List[str] = Field(
        default_factory=list,
        description="Languages spoken by the candidate"
    )


# ============================================================================
# Chain 2 : Job Description Extraction
# ============================================================================

class RequirementEntry(BaseModel):
    """Represents one job requirement."""

    name: str = Field(
        description="Required skill or qualification"
    )

    is_mandatory: bool = Field(
        description=(
            "True if must-have, False if preferred/nice-to-have"
        )
    )


class JDExtractionResult(BaseModel):
    """
    Structured information extracted from a job description.
    """

    job_title: str = Field(
        description="Target job title"
    )

    seniority_level: Optional[str] = Field(
        default=None,
        description="Junior, Mid, Senior, etc."
    )

    years_required: Optional[float] = Field(
        default=None,
        description="Years of experience requested"
    )


    target_candidate: Optional[str] = Field(
        default=None,
        description=(
            "Who this job is intended for. "
            "Examples: Student, Fresh Graduate, Entry-Level, "
            "Experienced Professional, Any."
        )
    )

    requirements: List[RequirementEntry] = Field(
        default_factory=list,
        description="Skills and qualifications extracted from the JD"
    )

    responsibilities: List[ResponsibilityEntry] = Field(
        default_factory=list,
        description="Responsibilities extracted from the job description."
    )


# ============================================================================
# Chain 3 / 4 / 5 : Career Gap Analysis
# ============================================================================
class GapAnalysisResult(BaseModel):

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

class RoadmapStep(BaseModel):
    """
    One learning step in the recommended roadmap.
    """

    skill: str = Field(
        description="Skill to learn"
    )

    action: str = Field(
        description="Concrete action for learning the skill"
    )

    priority: Optional[int] = Field(
        default=None,
        description="Learning priority (1 = highest)"
    )

    estimated_duration: Optional[str] = Field(
        default=None,
        description="Estimated learning duration"
    )

    grounded_in_source: Optional[str] = Field(
        default=None,
        description=(
            "Retrieved roadmap/document chunk supporting this recommendation"
        )
    )


class CareerGapAnalysis(BaseModel):
    """
    Final output shown to the user.
    """

    current_skills: List[str] = Field(
        default_factory=list,
        description="Skills matching the target job"
    )

    missing_skills: List[str] = Field(
        default_factory=list,
        description="Required skills the candidate lacks"
    )

    strengths: List[str] = Field(
        default_factory=list,
        description="Candidate strengths"
    )

    weaknesses: List[str] = Field(
        default_factory=list,
        description="Areas needing improvement"
    )

    match_percentage: float = Field(
        description="Estimated match percentage (0-100)"
    )

    recommended_roadmap: List[RoadmapStep] = Field(
        default_factory=list,
        description="Ordered learning roadmap"
    )

    suggested_jobs: List[str] = Field(
        default_factory=list,
        description="Alternative job roles suitable for the candidate"
    )

    overall_feedback: str = Field(
        description="Overall assessment and career advice"
    )