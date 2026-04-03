from __future__ import annotations

from typing import Any

from hyperderm.workflows.diagnosis_graph import build_diagnosis_workflow


class FakeDiagnosisService:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self._candidates = candidates

    def retrieve_candidates(
        self,
        descriptors: list[str],
        body_part: str,
        symptoms: list[str],
        effects: list[str],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self._candidates[:limit]

    @staticmethod
    def compile_output(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            return {
                "hierarchy_path": ["Other", "Unspecified", "Insufficient Evidence"],
                "candidate_list": [],
                "supporting_evidence": [],
                "counter_evidence": [],
                "why_not_top_alternatives": [],
                "final_recommendation": "Insufficient Evidence",
                "uncertainty": 1.0,
                "bias_checks": {},
                "privacy_safe_fields_used": [],
            }

        top = candidates[0]
        return {
            "hierarchy_path": [top["main_class"], top["sub_class"], top["disease"]],
            "candidate_list": candidates,
            "supporting_evidence": top.get("evidence", []),
            "counter_evidence": [],
            "why_not_top_alternatives": [],
            "final_recommendation": top["disease"],
            "uncertainty": 0.2,
            "bias_checks": {},
            "privacy_safe_fields_used": ["descriptors"],
        }


class FakeEvidenceService:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls = 0

    def collect_for_disease(self, disease_name: str) -> list[dict[str, Any]]:
        self.calls += 1
        return self.rows


class FakeLocalRagService:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls = 0

    def retrieve(self, query_terms: list[str], disease_hint: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        self.calls += 1
        return self.rows[:limit]


def test_agentic_rag_skips_external_research_when_confident() -> None:
    candidates = [
        {
            "main_class": "Inflammatory",
            "sub_class": "Dermatitis",
            "disease": "Atopic Dermatitis",
            "score": 8.5,
            "evidence": [{"title": "Graph source", "source": "neo4j:e1", "doi": ""}],
        }
    ]
    diagnosis_service = FakeDiagnosisService(candidates)
    evidence_service = FakeEvidenceService(
        [{"title": "External source", "source": "Crossref", "source_id": "x1", "doi": "10.1/x"}]
    )
    local_rag_service = FakeLocalRagService(
        [{"title": "Local source", "source": "backup:l1", "doi": ""}]
    )

    workflow = build_diagnosis_workflow(
        diagnosis_service,
        evidence_service=evidence_service,
        local_rag_service=local_rag_service,
    )
    state = workflow.invoke(
        {
            "descriptors": [],
            "body_part": "arm",
            "symptoms": ["itch"],
            "effects": [],
            "image_path": None,
        }
    )

    assert evidence_service.calls == 0
    assert local_rag_service.calls == 1

    final = state["final"]
    assert final["agentic"]["mode"] == "langgraph-rag-agentic-v2"
    assert final["agentic"]["research_triggered"] is False
    assert final["agentic"]["local_rag_used"] is True
    assert any(item.get("title") == "Local source" for item in final["supporting_evidence"])


def test_agentic_rag_triggers_external_research_when_low_confidence() -> None:
    candidates = [
        {
            "main_class": "Infectious",
            "sub_class": "Fungal",
            "disease": "Tinea Corporis",
            "score": 1.2,
            "evidence": [],
        }
    ]
    diagnosis_service = FakeDiagnosisService(candidates)
    evidence_service = FakeEvidenceService(
        [
            {
                "title": "External tinea review",
                "source": "Crossref",
                "source_id": "x2",
                "doi": "10.2/x",
            }
        ]
    )
    local_rag_service = FakeLocalRagService([])

    workflow = build_diagnosis_workflow(
        diagnosis_service,
        evidence_service=evidence_service,
        local_rag_service=local_rag_service,
    )
    state = workflow.invoke(
        {
            "descriptors": [],
            "body_part": "trunk",
            "symptoms": ["itch"],
            "effects": ["spread"],
            "image_path": None,
        }
    )

    assert evidence_service.calls == 1
    final = state["final"]
    assert final["agentic"]["research_triggered"] is True
    assert final["agentic"]["research_used"] is True
    assert any(item.get("title") == "External tinea review" for item in final["supporting_evidence"])
    assert any(step.get("step") == "assess" and step.get("needs_research") is True for step in final["agent_trace"])
