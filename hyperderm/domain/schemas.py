from typing import Any
from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    descriptors: list[str] = Field(default_factory=list)
    body_part: str = ""
    symptoms: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    image_path: str | None = None


class Candidate(BaseModel):
    main_class: str
    sub_class: str
    disease: str
    score: float


class DiagnoseResponse(BaseModel):
    hierarchy_path: list[str]
    candidate_list: list[Candidate]
    supporting_evidence: list[dict[str, Any]]
    counter_evidence: list[dict[str, Any]] = Field(default_factory=list)
    why_not_top_alternatives: list[dict[str, Any]] = Field(default_factory=list)
    final_recommendation: str
    uncertainty: float
    bias_checks: dict[str, Any]
    privacy_safe_fields_used: list[str]
    agentic: dict[str, Any] = Field(default_factory=dict)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)


class ChatbotRequest(BaseModel):
    message: str
    session_id: str | None = None
    image_path: str | None = None


class ChatbotResponse(BaseModel):
    answer: str
    session_id: str
    top_candidate: Candidate | None = None
    candidate_list: list[Candidate] = Field(default_factory=list)
    supporting_evidence: list[dict[str, Any]] = Field(default_factory=list)
    graph_context: list[dict[str, Any]] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)
    memory_summary: str = ""
    explainability: dict[str, Any] = Field(default_factory=dict)
    used_fallback: bool = False
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    model: str


class DiseaseHierarchyRecord(BaseModel):
    main_class: str
    sub_class: str
    disease: str
    morphologies: list[str]
    body_regions: list[str]


class EvidenceRecord(BaseModel):
    disease: str
    source: str
    evidence_id: str
    title: str
    journal: str
    pubdate: str
    doi: str
