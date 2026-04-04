from __future__ import annotations

from typing import Any

from hyperderm.workflows.chatbot_graph import build_chatbot_workflow


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


class FailingDiagnosisService:
    def retrieve_candidates(
        self,
        descriptors: list[str],
        body_part: str,
        symptoms: list[str],
        effects: list[str],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("neo4j unavailable")


class FakeMedGemmaService:
    def extract_query_features(self, message: str) -> dict[str, Any]:
        return {
            "descriptors": ["plaque", "scaling"],
            "body_part": "elbow",
            "symptoms": ["itch"],
            "effects": [],
            "source": "fake",
        }

    def generate_chat_answer(
        self,
        user_message: str,
        top_candidate: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        visual_features: dict[str, Any] | None = None,
        graph_context: list[dict[str, Any]] | None = None,
        memory_summary: str = "",
        suggested_questions: list[str] | None = None,
    ) -> str:
        if top_candidate:
            return f"Likely condition is {top_candidate['disease']}"
        return "No strong match"


class FakeLocalBackupService:
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


class FakeLocalRagService:
    def retrieve(self, query_terms: list[str], disease_hint: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        return [{"title": "Local evidence", "source": "backup:e1", "doi": ""}]


class EmptyLocalRagService:
    def retrieve(self, query_terms: list[str], disease_hint: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        return []


class FakeGraphRagService:
    def retrieve_context(self, disease_name: str, limit: int = 5) -> list[dict[str, Any]]:
        return [{"disease": disease_name, "evidence": [{"title": "Graph context evidence", "source": "neo4j:context", "doi": ""}]}]


class CountingGraphRagService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve_context(self, disease_name: str, limit: int = 5) -> list[dict[str, Any]]:
        self.calls.append(disease_name)
        return [{"disease": disease_name, "evidence": []}]


class EmptyGraphRagService:
    def retrieve_context(self, disease_name: str, limit: int = 5) -> list[dict[str, Any]]:
        return []


class FakeMemoryService:
    def summary(self, session_id: str, limit: int = 8) -> str:
        return ""

    def append(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        return None


class TurnAwareMemoryService(FakeMemoryService):
    def __init__(self, prior_user_turns: int) -> None:
        self._prior_user_turns = prior_user_turns

    def load_recent(self, session_id: str, limit: int = 8) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index in range(self._prior_user_turns):
            rows.append(
                {
                    "session_id": session_id,
                    "role": "user",
                    "content": f"turn-{index + 1}",
                    "metadata": {
                        "descriptors": ["plaque"],
                        "body_part": "elbow",
                        "symptoms": ["itch"],
                        "effects": [],
                    },
                }
            )
        return rows[-limit:]


def test_multi_agent_uses_neo4j_path() -> None:
    diagnosis_service = FakeDiagnosisService(
        [
            {
                "main_class": "Inflammatory",
                "sub_class": "Psoriasis",
                "disease": "Plaque Psoriasis",
                "score": 7.0,
                "evidence": [{"title": "Graph evidence", "source": "neo4j:e1", "doi": ""}],
            }
        ]
    )
    workflow = build_chatbot_workflow(
        diagnosis_service=diagnosis_service,
        medgemma_service=FakeMedGemmaService(),
        local_backup_service=FakeLocalBackupService([]),
        local_rag_service=EmptyLocalRagService(),
        graph_rag_service=EmptyGraphRagService(),
        memory_service=FakeMemoryService(),
    )

    state = workflow.invoke({"user_message": "itchy scaling plaque on elbow"})
    final = state["final"]

    assert final["used_fallback"] is False
    assert final["top_candidate"]["disease"] == "Plaque Psoriasis"
    assert "not a confirmed medical diagnosis" in final["answer"].lower()
    assert "neo4j_diagnosis_agent" in final["agent_roles"]


def test_multi_agent_fallback_path_when_neo4j_empty() -> None:
    diagnosis_service = FakeDiagnosisService([])
    workflow = build_chatbot_workflow(
        diagnosis_service=diagnosis_service,
        medgemma_service=FakeMedGemmaService(),
        local_backup_service=FakeLocalBackupService(
            [
                {
                    "main_class": "Infectious",
                    "sub_class": "Fungal",
                    "disease": "Tinea Corporis",
                    "score": 3.0,
                    "evidence": [],
                }
            ]
        ),
        local_rag_service=FakeLocalRagService(),
        graph_rag_service=FakeGraphRagService(),
        memory_service=FakeMemoryService(),
    )

    state = workflow.invoke({"user_message": "ring-like itchy lesions on trunk"})
    final = state["final"]

    assert final["used_fallback"] is True
    assert final["top_candidate"]["disease"] == "Tinea Corporis"
    assert any(item.get("step") == "local_backup_diagnosis_tool" for item in final["tool_trace"])


def test_multi_agent_abstains_on_zero_score_candidate() -> None:
    diagnosis_service = FakeDiagnosisService(
        [
            {
                "main_class": "Infectious",
                "sub_class": "Fungal",
                "disease": "Onychomycosis",
                "score": 0.0,
                "evidence": [],
            }
        ]
    )
    workflow = build_chatbot_workflow(
        diagnosis_service=diagnosis_service,
        medgemma_service=FakeMedGemmaService(),
        local_backup_service=FakeLocalBackupService([]),
        local_rag_service=EmptyLocalRagService(),
        graph_rag_service=EmptyGraphRagService(),
        memory_service=FakeMemoryService(),
    )

    state = workflow.invoke({"user_message": "red itchy cheeks"})
    final = state["final"]

    assert final["top_candidate"] is None
    assert final["candidate_list"] == []
    assert "enough reliable evidence" in final["answer"].lower()


def test_multi_agent_abstains_on_low_score_without_evidence() -> None:
    diagnosis_service = FakeDiagnosisService(
        [
            {
                "main_class": "Skin Appendages Disorders",
                "sub_class": "Sebacious Glands and Acneiform Disorders",
                "disease": "Acne",
                "score": 2.0,
                "evidence": [],
            }
        ]
    )
    workflow = build_chatbot_workflow(
        diagnosis_service=diagnosis_service,
        medgemma_service=FakeMedGemmaService(),
        local_backup_service=FakeLocalBackupService([]),
        local_rag_service=EmptyLocalRagService(),
        graph_rag_service=EmptyGraphRagService(),
        memory_service=FakeMemoryService(),
    )

    state = workflow.invoke({"user_message": "red itchy on cheeks"})
    final = state["final"]

    assert final["top_candidate"] is None
    assert final["candidate_list"] == []
    assert "consult a dermatologist" in final["answer"].lower()


def test_multi_agent_forces_diagnosis_by_third_round() -> None:
    diagnosis_service = FakeDiagnosisService(
        [
            {
                "main_class": "Skin Appendages Disorders",
                "sub_class": "Sebacious Glands and Acneiform Disorders",
                "disease": "Acne",
                "score": 2.0,
                "evidence": [],
            }
        ]
    )
    workflow = build_chatbot_workflow(
        diagnosis_service=diagnosis_service,
        medgemma_service=FakeMedGemmaService(),
        local_backup_service=FakeLocalBackupService([]),
        local_rag_service=EmptyLocalRagService(),
        graph_rag_service=EmptyGraphRagService(),
        memory_service=TurnAwareMemoryService(prior_user_turns=2),
    )

    state = workflow.invoke({"user_message": "still itchy on elbow", "session_id": "s1"})
    final = state["final"]

    assert final["top_candidate"] is not None
    assert final["top_candidate"]["disease"] == "Acne"
    assert final["candidate_list"]
    assert final["suggested_questions"] == []


def test_multi_agent_falls_back_when_neo4j_errors() -> None:
    workflow = build_chatbot_workflow(
        diagnosis_service=FailingDiagnosisService(),
        medgemma_service=FakeMedGemmaService(),
        local_backup_service=FakeLocalBackupService(
            [
                {
                    "main_class": "Infectious",
                    "sub_class": "Fungal",
                    "disease": "Tinea Corporis",
                    "score": 3.0,
                    "evidence": [],
                }
            ]
        ),
        local_rag_service=FakeLocalRagService(),
        graph_rag_service=EmptyGraphRagService(),
        memory_service=FakeMemoryService(),
    )

    state = workflow.invoke({"user_message": "itchy ring lesion on trunk"})
    final = state["final"]

    assert final["used_fallback"] is True
    assert final["top_candidate"] is not None
    assert final["top_candidate"]["disease"] == "Tinea Corporis"
    assert any("neo4j_unavailable" in str(item.get("error", "")) for item in final["tool_trace"])


def test_strict_mode_allows_medgemma_direct_inference_without_graph_candidates() -> None:
    diagnosis_service = FakeDiagnosisService([])
    workflow = build_chatbot_workflow(
        diagnosis_service=diagnosis_service,
        medgemma_service=FakeMedGemmaService(),
        local_backup_service=FakeLocalBackupService([]),
        local_rag_service=EmptyLocalRagService(),
        graph_rag_service=EmptyGraphRagService(),
        memory_service=TurnAwareMemoryService(prior_user_turns=1),
        strict_neo4j_only=True,
    )

    state = workflow.invoke({"user_message": "it is patchy and itchy on my face", "session_id": "strict-1"})
    final = state["final"]

    assert final["used_fallback"] is False
    assert final["top_candidate"] is None
    assert final["candidate_list"] == []
    assert "no strong match" in final["answer"].lower()
    reason = ((final.get("explainability", {}).get("reasoning", {}).get("confidence", {}).get("reason", "")))
    assert reason == "medgemma_direct_inference_no_graph_match"


def test_multi_agent_graph_context_for_all_selected_candidates() -> None:
    diagnosis_service = FakeDiagnosisService(
        [
            {
                "main_class": "Infectious",
                "sub_class": "Fungal",
                "disease": "Tinea Cruris",
                "score": 4.0,
                "evidence": [],
            },
            {
                "main_class": "Inflammatory",
                "sub_class": "Dermatitis",
                "disease": "Contact Dermatitis",
                "score": 3.0,
                "evidence": [],
            },
        ]
    )
    graph_rag_service = CountingGraphRagService()
    workflow = build_chatbot_workflow(
        diagnosis_service=diagnosis_service,
        medgemma_service=FakeMedGemmaService(),
        local_backup_service=FakeLocalBackupService([]),
        local_rag_service=EmptyLocalRagService(),
        graph_rag_service=graph_rag_service,
        memory_service=FakeMemoryService(),
    )

    state = workflow.invoke({"user_message": "itchy patches on groin and trunk"})
    final = state["final"]

    assert final["used_fallback"] is False
    assert set(graph_rag_service.calls) == {"Tinea Cruris", "Contact Dermatitis"}
    assert len(final["graph_context"]) == 2


def test_round_1_intake_engagement_with_minimal_input() -> None:
    """On round 1 with no candidates and minimal input (very short message), engage with probing questions."""

    diagnosis_service = FakeDiagnosisService([])  # No candidates from Neo4j
    workflow = build_chatbot_workflow(
        diagnosis_service=diagnosis_service,
        medgemma_service=FakeMedGemmaService(),
        local_backup_service=FakeLocalBackupService([]),
        local_rag_service=EmptyLocalRagService(),
        graph_rag_service=EmptyGraphRagService(),
        memory_service=FakeMemoryService(),
    )

    state = workflow.invoke({"user_message": "Hi"})
    final = state["final"]

    # Should NOT abstain on round 1 with minimal input and no candidates
    reason = ((final.get("explainability", {}).get("reasoning", {}).get("confidence", {}).get("reason", "")))
    assert reason == "round_1_intake_engagement"

    # Top candidate should be None but answer should be generated (simple engagement)
    assert final["top_candidate"] is None
    # The answer should be conversational (simple engagement, not diagnostic)
    assert "Please answer the questions below" in final["answer"] or "clinical questions" in final["answer"]
