from __future__ import annotations

from typing import Any, TypedDict
from langgraph.graph import END, StateGraph

from hyperderm.services.diagnosis_service import DiagnosisService
from hyperderm.services.evidence_service import EvidenceService
from hyperderm.services.feature_extractor import extract_visual_features
from hyperderm.services.local_rag_service import LocalRagService


class DiagnosisState(TypedDict, total=False):
    descriptors: list[str]
    body_part: str
    symptoms: list[str]
    effects: list[str]
    image_path: str | None
    visual_features: dict[str, Any]
    candidates: list[dict[str, Any]]
    local_evidence: list[dict[str, Any]]
    external_evidence: list[dict[str, Any]]
    needs_research: bool
    research_triggered: bool
    agent_trace: list[dict[str, Any]]
    final: dict[str, Any]


def build_diagnosis_workflow(
    diagnosis_service: DiagnosisService,
    evidence_service: EvidenceService | None = None,
    local_rag_service: LocalRagService | None = None,
):
    def _append_trace(state: DiagnosisState, event: dict[str, Any]) -> list[dict[str, Any]]:
        trace = list(state.get("agent_trace", []))
        trace.append(event)
        return trace

    def node_plan(state: DiagnosisState) -> DiagnosisState:
        descriptors = [item.strip().lower() for item in state.get("descriptors", []) if item.strip()]
        symptoms = [item.strip().lower() for item in state.get("symptoms", []) if item.strip()]
        effects = [item.strip().lower() for item in state.get("effects", []) if item.strip()]
        body_part = state.get("body_part", "").strip().lower()

        strategy = "descriptor_first"
        if not descriptors and symptoms:
            strategy = "symptom_first"
        if not descriptors and not symptoms and state.get("image_path"):
            strategy = "image_first"

        return {
            "descriptors": descriptors,
            "symptoms": symptoms,
            "effects": effects,
            "body_part": body_part,
            "agent_trace": _append_trace(
                state,
                {
                    "step": "plan",
                    "strategy": strategy,
                    "has_image": bool(state.get("image_path")),
                    "descriptor_count": len(descriptors),
                    "symptom_count": len(symptoms),
                },
            ),
        }

    def node_extract_features(state: DiagnosisState) -> DiagnosisState:
        descriptors = state.get("descriptors", [])
        if not descriptors and not state.get("image_path"):
            return {
                "visual_features": {},
                "agent_trace": _append_trace(state, {"step": "extract_features", "status": "skipped"}),
            }

        return {
            "visual_features": extract_visual_features(
                image_path=state.get("image_path"),
                descriptors=descriptors,
            ),
            "agent_trace": _append_trace(state, {"step": "extract_features", "status": "completed"}),
        }

    def node_retrieve_candidates(state: DiagnosisState) -> DiagnosisState:
        candidates = diagnosis_service.retrieve_candidates(
            descriptors=state.get("descriptors", []),
            body_part=state.get("body_part", ""),
            symptoms=state.get("symptoms", []),
            effects=state.get("effects", []),
            limit=5,
        )
        return {
            "candidates": candidates,
            "agent_trace": _append_trace(
                state,
                {
                    "step": "retrieve_graph_candidates",
                    "candidate_count": len(candidates),
                    "top_score": float(candidates[0]["score"]) if candidates else 0.0,
                },
            ),
        }

    def node_retrieve_local_rag(state: DiagnosisState) -> DiagnosisState:
        if local_rag_service is None:
            return {
                "local_evidence": [],
                "agent_trace": _append_trace(
                    state,
                    {"step": "retrieve_local_rag", "status": "skipped", "reason": "service_unavailable"},
                ),
            }

        candidates = state.get("candidates", [])
        top_disease = str(candidates[0].get("disease", "")).strip() if candidates else ""
        query_terms = [
            *state.get("descriptors", []),
            *state.get("symptoms", []),
            *state.get("effects", []),
            state.get("body_part", ""),
        ]

        rows = local_rag_service.retrieve(query_terms=query_terms, disease_hint=top_disease or None, limit=5)
        return {
            "local_evidence": rows,
            "agent_trace": _append_trace(
                state,
                {
                    "step": "retrieve_local_rag",
                    "status": "completed",
                    "rows": len(rows),
                    "disease_hint": top_disease,
                },
            ),
        }

    def node_assess_need_for_research(state: DiagnosisState) -> DiagnosisState:
        candidates = state.get("candidates", [])
        top_score = float(candidates[0]["score"]) if candidates else 0.0
        top_evidence_count = len(candidates[0].get("evidence", [])) if candidates else 0
        local_evidence_count = len(state.get("local_evidence", []))

        needs_research = (not candidates) or top_score < 3.0 or (top_evidence_count + local_evidence_count) == 0
        # Track trigger independently from external fetch success.
        research_triggered = needs_research

        return {
            "needs_research": needs_research,
            "research_triggered": research_triggered,
            "agent_trace": _append_trace(
                state,
                {
                    "step": "assess",
                    "needs_research": needs_research,
                    "local_evidence_count": local_evidence_count,
                    "reason": "low_confidence_or_sparse_evidence" if needs_research else "enough_signal",
                },
            ),
        }

    def _route_after_assess(state: DiagnosisState) -> str:
        return "research" if state.get("needs_research") else "finalize"

    def node_collect_research(state: DiagnosisState) -> DiagnosisState:
        if evidence_service is None:
            return {
                "external_evidence": [],
                "agent_trace": _append_trace(
                    state,
                    {"step": "collect_research", "status": "skipped", "reason": "service_unavailable"},
                ),
            }

        candidates = state.get("candidates", [])
        if not candidates:
            return {
                "external_evidence": [],
                "agent_trace": _append_trace(
                    state,
                    {"step": "collect_research", "status": "skipped", "reason": "no_candidate"},
                ),
            }

        disease_name = str(candidates[0].get("disease", "")).strip()
        if not disease_name:
            return {
                "external_evidence": [],
                "agent_trace": _append_trace(
                    state,
                    {"step": "collect_research", "status": "skipped", "reason": "empty_disease"},
                ),
            }

        try:
            rows = evidence_service.collect_for_disease(disease_name)
            return {
                "external_evidence": rows[:5],
                "agent_trace": _append_trace(
                    state,
                    {
                        "step": "collect_research",
                        "status": "completed",
                        "disease": disease_name,
                        "rows": len(rows),
                    },
                ),
            }
        except Exception as error:  # noqa: BLE001
            return {
                "external_evidence": [],
                "agent_trace": _append_trace(
                    state,
                    {
                        "step": "collect_research",
                        "status": "failed",
                        "disease": disease_name,
                        "error": str(error)[:180],
                    },
                ),
            }

    def node_compile_output(state: DiagnosisState) -> DiagnosisState:
        final = diagnosis_service.compile_output(state.get("candidates", []))

        local_evidence = state.get("local_evidence", [])
        external_evidence = state.get("external_evidence", [])
        if local_evidence or external_evidence:
            normalized = [
                {
                    "title": item.get("title", ""),
                    "source": item.get("evidence_id") or item.get("source_id") or item.get("source", ""),
                    "doi": item.get("doi", ""),
                }
                for item in [*local_evidence, *external_evidence]
            ]
            merged = list(final.get("supporting_evidence", []))
            merged.extend(normalized)

            deduped: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for row in merged:
                key = (str(row.get("title", "")).strip().lower(), str(row.get("source", "")).strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(row)
            final["supporting_evidence"] = deduped[:6]

        final["agentic"] = {
            "mode": "langgraph-rag-agentic-v2",
            "research_triggered": bool(state.get("research_triggered", False)),
            "local_rag_used": bool(local_evidence),
            "research_used": bool(external_evidence),
        }
        final["agent_trace"] = state.get("agent_trace", [])[-10:]
        return {"final": final}

    graph = StateGraph(DiagnosisState)
    graph.add_node("plan", node_plan)
    graph.add_node("extract_features", node_extract_features)
    graph.add_node("retrieve_candidates", node_retrieve_candidates)
    graph.add_node("retrieve_local_rag", node_retrieve_local_rag)
    graph.add_node("assess", node_assess_need_for_research)
    graph.add_node("collect_research", node_collect_research)
    graph.add_node("compile_output", node_compile_output)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "extract_features")
    graph.add_edge("extract_features", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "retrieve_local_rag")
    graph.add_edge("retrieve_local_rag", "assess")
    graph.add_conditional_edges(
        "assess",
        _route_after_assess,
        {
            "research": "collect_research",
            "finalize": "compile_output",
        },
    )
    graph.add_edge("collect_research", "compile_output")
    graph.add_edge("compile_output", END)

    return graph.compile()
