from __future__ import annotations

from typing import Any, TypedDict
from langgraph.graph import END, StateGraph

from hyperderm.services.diagnosis_service import DiagnosisService
from hyperderm.services.feature_extractor import extract_visual_features


class DiagnosisState(TypedDict, total=False):
    descriptors: list[str]
    body_part: str
    symptoms: list[str]
    effects: list[str]
    image_path: str | None
    visual_features: dict[str, Any]
    candidates: list[dict[str, Any]]
    final: dict[str, Any]


def build_diagnosis_workflow(diagnosis_service: DiagnosisService):
    def node_extract_features(state: DiagnosisState) -> DiagnosisState:
        return {
            "visual_features": extract_visual_features(
                image_path=state.get("image_path"),
                descriptors=state.get("descriptors", []),
            )
        }

    def node_retrieve_candidates(state: DiagnosisState) -> DiagnosisState:
        candidates = diagnosis_service.retrieve_candidates(
            descriptors=state.get("descriptors", []),
            body_part=state.get("body_part", ""),
            symptoms=state.get("symptoms", []),
            effects=state.get("effects", []),
            limit=5,
        )
        return {"candidates": candidates}

    def node_compile_output(state: DiagnosisState) -> DiagnosisState:
        return {"final": diagnosis_service.compile_output(state.get("candidates", []))}

    graph = StateGraph(DiagnosisState)
    graph.add_node("extract_features", node_extract_features)
    graph.add_node("retrieve_candidates", node_retrieve_candidates)
    graph.add_node("compile_output", node_compile_output)

    graph.set_entry_point("extract_features")
    graph.add_edge("extract_features", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "compile_output")
    graph.add_edge("compile_output", END)

    return graph.compile()
