from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from hyperderm.services.diagnosis_service import DiagnosisService
from hyperderm.services.conversation_memory import ConversationMemoryService
from hyperderm.services.graph_rag_service import GraphRagService
from hyperderm.services.local_backup_diagnosis_service import LocalBackupDiagnosisService
from hyperderm.services.local_rag_service import LocalRagService
from hyperderm.services.medgemma_chat_service import MedGemmaChatService
from hyperderm.services.feature_extractor import extract_visual_features


class ChatbotState(TypedDict, total=False):
    session_id: str
    user_message: str
    image_path: str | None
    plan: dict[str, Any]
    memory_slots: dict[str, Any]
    recent_messages: list[dict[str, Any]]
    greeting_intent: bool
    current_turn_signal_count: int
    conversation_round: int
    force_diagnosis: bool
    descriptors: list[str]
    body_part: str
    symptoms: list[str]
    effects: list[str]
    extraction_source: str
    visual_features: dict[str, Any]
    graph_context: list[dict[str, Any]]
    graph_rag_attempted: bool
    neo4j_candidates: list[dict[str, Any]]
    fallback_candidates: list[dict[str, Any]]
    selected_candidates: list[dict[str, Any]]
    used_fallback: bool
    supporting_evidence: list[dict[str, Any]]
    should_abstain: bool
    abstain_reason: str
    suggested_questions: list[str]
    memory_summary: str
    draft_answer: str
    safe_answer: str
    tool_trace: list[dict[str, Any]]
    final: dict[str, Any]


def build_chatbot_workflow(
    diagnosis_service: DiagnosisService,
    medgemma_service: MedGemmaChatService,
    local_backup_service: LocalBackupDiagnosisService,
    local_rag_service: LocalRagService,
    graph_rag_service: GraphRagService,
    memory_service: ConversationMemoryService,
    strict_neo4j_only: bool = False,
):
    def _safe_memory_load_recent(session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        loader = getattr(memory_service, "load_recent", None)
        if callable(loader):
            try:
                rows = loader(session_id, limit=limit)
                return rows if isinstance(rows, list) else []
            except Exception:
                return []
        return []

    def _safe_memory_summary(session_id: str) -> str:
        summarizer = getattr(memory_service, "summary", None)
        if callable(summarizer):
            try:
                return str(summarizer(session_id, limit=8) or "")
            except TypeError:
                try:
                    return str(summarizer(session_id) or "")
                except Exception:
                    return ""
            except Exception:
                return ""
        return ""

    def _safe_memory_append(session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        appender = getattr(memory_service, "append", None)
        if callable(appender):
            try:
                appender(session_id, role, content, metadata)
            except Exception:
                return

    def _append_trace(state: ChatbotState, event: dict[str, Any]) -> list[dict[str, Any]]:
        trace = list(state.get("tool_trace", []))
        trace.append(event)
        return trace

    def _is_greeting_or_small_talk(message: str) -> bool:
        normalized = re.sub(r"[^a-z\s]", "", (message or "").strip().lower())
        if not normalized:
            return True
        minimal = {
            "hi",
            "hello",
            "hey",
            "help",
            "yo",
            "hola",
            "good morning",
            "good evening",
            "good afternoon",
        }
        return normalized in minimal

    def node_supervisor_plan(state: ChatbotState) -> ChatbotState:
        session_id = state.get("session_id") or "default"
        user_message = str(state.get("user_message", ""))
        recent = _safe_memory_load_recent(session_id, limit=12)
        memory_summary = _safe_memory_summary(session_id)
        prior_user_turns = sum(1 for row in recent if str(row.get("role", "")).lower() == "user")
        conversation_round = prior_user_turns + 1
        force_diagnosis = conversation_round >= 3
        greeting_intent = _is_greeting_or_small_talk(user_message)

        slots: dict[str, Any] = {
            "descriptors": [],
            "body_part": "",
            "symptoms": [],
            "effects": [],
            "image_path": "",
        }
        for row in recent:
            if str(row.get("role", "")).lower() != "user":
                continue
            metadata = row.get("metadata") or {}
            if metadata.get("descriptors"):
                slots["descriptors"] = list(metadata.get("descriptors", []))
            if metadata.get("body_part"):
                slots["body_part"] = str(metadata.get("body_part", ""))
            if metadata.get("symptoms"):
                slots["symptoms"] = list(metadata.get("symptoms", []))
            if metadata.get("effects"):
                slots["effects"] = list(metadata.get("effects", []))
            if metadata.get("image_path"):
                slots["image_path"] = str(metadata.get("image_path", ""))

        return {
            "plan": {
                "agents": [
                    "query_interpreter_agent",
                    "neo4j_diagnosis_agent",
                    "backup_diagnosis_agent",
                    "rag_evidence_agent",
                    "medical_safety_agent",
                    "response_agent",
                ]
            },
            "memory_summary": memory_summary,
            "recent_messages": recent,
            "greeting_intent": greeting_intent,
            "memory_slots": slots,
            "conversation_round": conversation_round,
            "force_diagnosis": force_diagnosis,
            "tool_trace": _append_trace(
                state,
                {
                    "step": "supervisor_plan",
                    "agent": "supervisor_agent",
                    "conversation_round": conversation_round,
                    "force_diagnosis": force_diagnosis,
                },
            ),
        }

    def node_extract_query(state: ChatbotState) -> ChatbotState:
        extracted = medgemma_service.extract_query_features(state.get("user_message", ""))
        memory_slots = state.get("memory_slots", {})
        image_path = state.get("image_path") or memory_slots.get("image_path")
        visual_features = {}
        if image_path:
            visual_features = extract_visual_features(image_path=image_path, descriptors=extracted.get("descriptors", []))
        merged_descriptors = sorted(
            {
                *memory_slots.get("descriptors", []),
                *extracted.get("descriptors", []),
                *visual_features.get("descriptor_tokens", []),
            }
        )
        merged_symptoms = sorted({*memory_slots.get("symptoms", []), *extracted.get("symptoms", [])})
        merged_effects = sorted({*memory_slots.get("effects", []), *extracted.get("effects", [])})
        body_part = extracted.get("body_part", "") or memory_slots.get("body_part", "")
        current_turn_signal_count = (
            len(extracted.get("descriptors", []))
            + len(extracted.get("symptoms", []))
            + len(extracted.get("effects", []))
            + (1 if extracted.get("body_part", "") else 0)
            + len(visual_features.get("descriptor_tokens", []))
        )
        return {
            "descriptors": merged_descriptors,
            "body_part": body_part,
            "symptoms": merged_symptoms,
            "effects": merged_effects,
            "current_turn_signal_count": current_turn_signal_count,
            "extraction_source": extracted.get("source", "unknown"),
            "visual_features": visual_features,
            "tool_trace": _append_trace(
                state,
                {
                    "step": "extract_query_features",
                    "agent": "query_interpreter_agent",
                    "source": extracted.get("source", "unknown"),
                    "image_used": bool(image_path),
                },
            ),
        }

    def node_diagnose_neo4j(state: ChatbotState) -> ChatbotState:
        try:
            candidates = diagnosis_service.retrieve_candidates(
                descriptors=state.get("descriptors", []),
                body_part=state.get("body_part", ""),
                symptoms=state.get("symptoms", []),
                effects=state.get("effects", []),
                limit=5,
            )
        except Exception as error:
            # Neo4j can be unavailable in local/dev environments. Treat as no candidates
            # so the graph routes into local backup fallback instead of returning 500.
            return {
                "neo4j_candidates": [],
                "tool_trace": _append_trace(
                    state,
                    {
                        "step": "neo4j_diagnosis_tool",
                        "agent": "neo4j_diagnosis_agent",
                        "candidate_count": 0,
                        "error": f"neo4j_unavailable:{type(error).__name__}",
                    },
                ),
            }
        return {
            "neo4j_candidates": candidates,
            "tool_trace": _append_trace(
                state,
                {
                    "step": "neo4j_diagnosis_tool",
                    "agent": "neo4j_diagnosis_agent",
                    "candidate_count": len(candidates),
                },
            ),
        }

    def _route_after_neo4j(state: ChatbotState) -> str:
        candidates = state.get("neo4j_candidates", [])
        if strict_neo4j_only and not candidates:
            return "continue"
        return "use_fallback" if not candidates else "continue"

    def node_diagnose_fallback(state: ChatbotState) -> ChatbotState:
        candidates = local_backup_service.retrieve_candidates(
            descriptors=state.get("descriptors", []),
            body_part=state.get("body_part", ""),
            symptoms=state.get("symptoms", []),
            effects=state.get("effects", []),
            limit=5,
        )
        return {
            "fallback_candidates": candidates,
            "selected_candidates": candidates,
            "used_fallback": True,
            "tool_trace": _append_trace(
                state,
                {
                    "step": "local_backup_diagnosis_tool",
                    "agent": "backup_diagnosis_agent",
                    "candidate_count": len(candidates),
                },
            ),
        }

    def node_select_primary(state: ChatbotState) -> ChatbotState:
        candidates = state.get("neo4j_candidates", [])
        graph_context = []
        graph_rag_attempted = False
        graph_rag_error = ""
        if candidates:
            graph_rag_attempted = True
            try:
                seen_diseases: set[str] = set()
                for candidate in candidates[:5]:
                    disease_name = str(candidate.get("disease", "")).strip()
                    if not disease_name or disease_name in seen_diseases:
                        continue
                    seen_diseases.add(disease_name)
                    rows = graph_rag_service.retrieve_context(disease_name, limit=3)
                    graph_context.extend(rows)
            except Exception as error:
                graph_context = []
                graph_rag_error = f"graph_rag_unavailable:{type(error).__name__}"
        return {
            "selected_candidates": candidates,
            "graph_context": graph_context,
            "graph_rag_attempted": graph_rag_attempted,
            "used_fallback": False,
            "tool_trace": _append_trace(
                state,
                {
                    "step": "select_primary_diagnosis",
                    "agent": "neo4j_diagnosis_agent",
                    "candidate_count": len(candidates),
                    "graph_rag_attempted": graph_rag_attempted,
                    "graph_rag_context_count": len(graph_context),
                    "graph_rag_diseases": [str(row.get("disease", "")) for row in graph_context[:5]],
                    "graph_rag_error": graph_rag_error,
                },
            ),
        }

    def node_retrieve_evidence(state: ChatbotState) -> ChatbotState:
        selected = state.get("selected_candidates", [])
        top_disease = str(selected[0].get("disease", "")).strip() if selected else ""
        if strict_neo4j_only:
            if selected:
                merged = list(selected[0].get("evidence", []))
                for context_row in state.get("graph_context", []):
                    merged.extend(context_row.get("evidence", []))
            else:
                merged = []
        else:
            query_terms = [
                *state.get("descriptors", []),
                *state.get("symptoms", []),
                *state.get("effects", []),
                state.get("body_part", ""),
            ]
            evidence = local_rag_service.retrieve(query_terms=query_terms, disease_hint=top_disease or None, limit=5)

            if selected:
                merged = list(selected[0].get("evidence", []))
                merged.extend(evidence)
                for context_row in state.get("graph_context", []):
                    merged.extend(context_row.get("evidence", []))
            else:
                merged = evidence

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in merged:
            key = (str(row.get("title", "")).strip().lower(), str(row.get("source", "")).strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "title": row.get("title", ""),
                    "source": row.get("source", ""),
                    "doi": row.get("doi", ""),
                }
            )

        return {
            "supporting_evidence": deduped[:6],
            "tool_trace": _append_trace(
                state,
                {
                    "step": "neo4j_evidence_tool" if strict_neo4j_only else "local_backup_evidence_tool",
                    "agent": "rag_evidence_agent",
                    "evidence_count": len(deduped[:6]),
                    "strict_neo4j_only": strict_neo4j_only,
                },
            ),
        }

    def node_generate_questions(state: ChatbotState) -> ChatbotState:
        if state.get("force_diagnosis", False):
            return {
                "suggested_questions": [],
                "tool_trace": _append_trace(
                    state,
                    {
                        "step": "generate_patient_questions",
                        "agent": "supervisor_agent",
                        "question_count": 0,
                        "suppressed_by_force_diagnosis": True,
                    },
                ),
            }

        selected = state.get("selected_candidates", [])
        top = selected[0] if selected else {}
        body_part = (state.get("body_part") or "").lower()
        candidate_name = str(top.get("disease", "")).lower()
        questions: list[str] = []
        
        # Extract previously asked questions from conversation history
        recent_messages = state.get("recent_messages", [])
        previously_asked = set()
        for msg in recent_messages:
            if str(msg.get("role", "")).lower() == "assistant":
                content = str(msg.get("content", "")).lower()
                # Mark topics that have been discussed
                if "body" in content or "location" in content or "where" in content:
                    previously_asked.add("location")
                if "descriptor" in content or "look" in content or "appearance" in content or "shape" in content:
                    previously_asked.add("appearance")
                if "symptom" in content or "itch" in content or "pain" in content or "burning" in content:
                    previously_asked.add("symptom")
                if "spread" in content or "mark" in content or "ooz" in content or "crack" in content:
                    previously_asked.add("progression")

        # Only add questions for topics not yet discussed
        if "location" not in previously_asked and not body_part:
            questions.append("Which part of your body is most affected right now?")
        if "appearance" not in previously_asked and not state.get("descriptors"):
            questions.append("How does it look: flat, scaly, raised, ring-shaped, or patchy?")
        if "symptom" not in previously_asked and not state.get("symptoms"):
            questions.append("Do you feel itching, pain, burning, or none of these?")
        if "progression" not in previously_asked and not state.get("effects"):
            questions.append("Has it spread, cracked, oozed, or left marks over time?")

        if "acne" in candidate_name:
            questions.extend([
                "Are there blackheads, whiteheads, or tender bumps?"])
        elif any(term in candidate_name for term in ["tinea", "fungal", "ringworm"]):
            questions.extend([
                "Is it ring-shaped with clearer skin in the center?",
                "Has it spread to other areas or to close contacts at home?",
            ])
        elif any(term in candidate_name for term in ["psoriasis", "eczema", "dermatitis"]):
            questions.extend([
                "Does it come and go, or stay in the same places?",
                "Any dry skin, scaling, or triggers like soaps, weather, or stress?",
            ])
        elif any(term in candidate_name for term in ["vitiligo", "depigmented"]):
            questions.extend([
                "Are the patches completely white or just lighter than the surrounding skin?",
                "Do the patches have sharp borders or hair color change?",
            ])

        # de-duplicate and keep patient-answerable concise questions
        filtered: list[str] = []
        seen = set()
        for question in questions:
            clean = question.strip()
            if clean and clean not in seen:
                seen.add(clean)
                filtered.append(clean)

        return {
            "suggested_questions": filtered[:3],  # Reduced from 5 to 3 for focus
            "tool_trace": _append_trace(
                state,
                {
                    "step": "generate_patient_questions",
                    "agent": "supervisor_agent",
                    "question_count": len(filtered[:3]),
                },
            ),
        }

    def node_assess_confidence(state: ChatbotState) -> ChatbotState:
        selected = state.get("selected_candidates", [])
        top = selected[0] if selected else {}
        top_score = float(top.get("score", 0.0)) if top else 0.0
        evidence_count = len(state.get("supporting_evidence", []))
        conversation_round = int(state.get("conversation_round", 1) or 1)
        current_turn_signal_count = int(state.get("current_turn_signal_count", 0) or 0)

        has_signal = bool(state.get("descriptors") or state.get("symptoms") or state.get("body_part"))
        force_diagnosis = bool(state.get("force_diagnosis", False))
        user_message = str(state.get("user_message", "")).strip().lower()
        greeting_intent = bool(state.get("greeting_intent", False))

        if greeting_intent and current_turn_signal_count == 0:
            # Pure greeting/small-talk should never trigger diagnosis.
            should_abstain = False
            reason = "greeting_intake_engagement"
        elif current_turn_signal_count < 2 and evidence_count == 0:
            # Hard floor: without minimum fresh signal and evidence, ask for more info.
            should_abstain = True
            reason = "insufficient_signal_or_evidence"
        elif force_diagnosis and selected and has_signal and top_score > 0.0 and evidence_count > 0 and current_turn_signal_count >= 2:
            should_abstain = False
            reason = "forced_diagnosis_after_round_limit"
        elif strict_neo4j_only and (not selected) and has_signal and conversation_round >= 2 and current_turn_signal_count >= 2:
            # In strict graph mode, allow MedGemma to provide model-only differential reasoning
            # after enough conversational signal, even when graph has no candidate match.
            should_abstain = False
            reason = "medgemma_direct_inference_no_graph_match"
        elif conversation_round == 1 and not selected and len(user_message) <= 10:
            # On round 1 with NO candidates and very minimal user input (e.g., "Hi", "Hello"),
            # engage with probing questions instead of immediately abstaining.
            should_abstain = False
            reason = "round_1_intake_engagement"
        else:
            should_abstain = (
                (not selected)
                or (top_score <= 0.0)
                or (not has_signal)
                or (evidence_count == 0)
                or (current_turn_signal_count < 2)
                or (top_score < 3.0 and evidence_count == 0)
            )
            reason = "low_confidence_or_no_signal" if should_abstain else "sufficient_signal"

        return {
            "should_abstain": should_abstain,
            "abstain_reason": reason,
            "tool_trace": _append_trace(
                state,
                {
                    "step": "assess_confidence",
                    "agent": "supervisor_agent",
                    "top_score": top_score,
                    "evidence_count": evidence_count,
                    "should_abstain": should_abstain,
                    "force_diagnosis": force_diagnosis,
                    "conversation_round": conversation_round,
                    "strict_neo4j_only": strict_neo4j_only,
                },
            ),
        }

    def _route_after_assess(state: ChatbotState) -> str:
        return "abstain" if state.get("should_abstain") else "generate"

    def node_abstain_answer(state: ChatbotState) -> ChatbotState:
        reason = str(state.get("abstain_reason", "low_confidence_or_no_signal"))
        if reason == "insufficient_signal_or_evidence":
            message = (
                "I need a little more information before suggesting a likely condition. "
                "Please share where it is located, how it looks (patchy/scaly/ring-shaped), and what you feel (itch, pain, burning). "
                "If possible, upload a clear photo so I can help more accurately."
            )
        else:
            message = (
                "Thanks for sharing that. I do not yet have enough reliable evidence to provide a specific diagnosis. "
                "Please tell me a bit more about the shape, exact location, duration, spread, and symptoms (itch, pain, burning, discharge). "
                "If this is worsening quickly, painful, or spreading, please consult a dermatologist for an in-person exam soon."
            )
        return {
            "draft_answer": message,
            "tool_trace": _append_trace(
                state,
                {
                    "step": "abstain_response",
                    "agent": "medical_safety_agent",
                    "reason": state.get("abstain_reason", "low_confidence_or_no_signal"),
                },
            ),
        }

    def node_generate_answer(state: ChatbotState) -> ChatbotState:
        selected = state.get("selected_candidates", [])
        top_candidate = selected[0] if selected else None
        abstain_reason = state.get("abstain_reason", "")
        
        # On round 1 intake engagement, generate a simple welcome message
        # The questions will be presented separately via suggested_questions UI element
        if abstain_reason in {"round_1_intake_engagement", "greeting_intake_engagement"}:
            answer = (
                "Hello, I can help with this. "
                "Please describe your skin concern (where it is, how it looks, and symptoms like itch or pain), "
                "or upload a photo so I can guide you safely."
            )
        else:
            answer = medgemma_service.generate_chat_answer(
                user_message=state.get("user_message", ""),
                top_candidate=top_candidate,
                candidates=selected,
                evidence=state.get("supporting_evidence", []),
                visual_features=state.get("visual_features", {}),
                graph_context=state.get("graph_context", []),
                memory_summary=state.get("memory_summary", ""),
                recent_messages=state.get("recent_messages", []),
                suggested_questions=state.get("suggested_questions", []),
            )

        return {
            "draft_answer": answer,
            "tool_trace": _append_trace(state, {"step": "draft_response", "agent": "response_agent"}),
        }

    def node_medical_safety_guard(state: ChatbotState) -> ChatbotState:
        answer = state.get("draft_answer", "").strip()
        disclaimer = " This is decision support only and not a confirmed medical diagnosis."
        if "not a" in answer.lower() and "diagnosis" in answer.lower():
            safe_answer = answer
        else:
            safe_answer = (answer + disclaimer).strip()

        return {
            "safe_answer": safe_answer,
            "tool_trace": _append_trace(state, {"step": "medical_safety_guard", "agent": "medical_safety_agent"}),
        }

    def node_finalize(state: ChatbotState) -> ChatbotState:
        selected = state.get("selected_candidates", [])
        top_candidate = None if state.get("should_abstain") else (selected[0] if selected else None)
        hierarchy_path = []
        if top_candidate:
            hierarchy_path = [
                str(top_candidate.get("main_class", "Other") or "Other"),
                str(top_candidate.get("sub_class", "Unspecified") or "Unspecified"),
                str(top_candidate.get("disease", "Insufficient Evidence") or "Insufficient Evidence"),
            ]

        explainability = {
            "hierarchy_path": hierarchy_path,
            "reasoning": {
                "matched_inputs": {
                    "descriptors": state.get("descriptors", []),
                    "body_part": state.get("body_part", ""),
                    "symptoms": state.get("symptoms", []),
                    "effects": state.get("effects", []),
                },
                "confidence": {
                    "should_abstain": bool(state.get("should_abstain", False)),
                    "reason": state.get("abstain_reason", ""),
                    "conversation_round": int(state.get("conversation_round", 1) or 1),
                    "forced_on_round_limit": bool(state.get("abstain_reason") == "forced_diagnosis_after_round_limit"),
                    "top_score": float(selected[0].get("score", 0.0)) if selected else 0.0,
                },
                "retrieval": {
                    "graph_rag_attempted": bool(state.get("graph_rag_attempted", False)),
                    "graph_rag_context_count": len(state.get("graph_context", [])),
                    "graph_rag_diseases": [str(row.get("disease", "")) for row in state.get("graph_context", [])[:5]],
                    "evidence_count": len(state.get("supporting_evidence", [])),
                    "used_fallback": bool(state.get("used_fallback", False)),
                },
                "neo4j_candidate_matches": [
                    {
                        "disease": item.get("disease", ""),
                        "score": float(item.get("score", 0.0)),
                        "matched_descriptors": item.get("matched_descriptors", []),
                        "matched_body_regions": item.get("matched_body_regions", []),
                        "matched_symptoms": item.get("matched_symptoms", []),
                        "matched_effects": item.get("matched_effects", []),
                        "matched_visual_atoms": item.get("matched_visual_atoms", []),
                    }
                    for item in selected[:5]
                ],
            },
            "follow_up_questions": state.get("suggested_questions", []),
        }

        final = {
            "answer": state.get("safe_answer") or state.get("draft_answer") or "",
            "top_candidate": top_candidate,
            "candidate_list": [
                {
                    "main_class": item.get("main_class", "Other"),
                    "sub_class": item.get("sub_class", "Unspecified"),
                    "disease": item.get("disease", "Insufficient Evidence"),
                    "score": float(item.get("score", 0.0)),
                }
                for item in selected
            ] if not state.get("should_abstain") else [],
            "supporting_evidence": state.get("supporting_evidence", []),
            "graph_context": state.get("graph_context", []),
            "suggested_questions": state.get("suggested_questions", []),
            "memory_summary": state.get("memory_summary", ""),
            "used_fallback": bool(state.get("used_fallback", False)),
            "explainability": explainability,
            "tool_trace": _append_trace(state, {"step": "finalize", "agent": "supervisor_agent"})[-14:],
            "agent_roles": state.get("plan", {}).get("agents", []),
        }
        session_id = state.get("session_id") or "default"
        _safe_memory_append(
            session_id,
            "user",
            state.get("user_message", ""),
            {
                "image_path": state.get("image_path"),
                "descriptors": state.get("descriptors", []),
                "body_part": state.get("body_part", ""),
                "symptoms": state.get("symptoms", []),
                "effects": state.get("effects", []),
            },
        )
        _safe_memory_append(session_id, "assistant", final["answer"], {"top_candidate": top_candidate.get("disease") if top_candidate else None})
        return {"final": final}

    graph = StateGraph(ChatbotState)
    graph.add_node("supervisor_plan", node_supervisor_plan)
    graph.add_node("extract_query", node_extract_query)
    graph.add_node("diagnose_neo4j", node_diagnose_neo4j)
    graph.add_node("diagnose_fallback", node_diagnose_fallback)
    graph.add_node("select_primary", node_select_primary)
    graph.add_node("retrieve_evidence", node_retrieve_evidence)
    graph.add_node("generate_patient_questions", node_generate_questions)
    graph.add_node("assess_confidence", node_assess_confidence)
    graph.add_node("abstain_answer", node_abstain_answer)
    graph.add_node("generate_answer", node_generate_answer)
    graph.add_node("medical_safety_guard", node_medical_safety_guard)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("supervisor_plan")
    graph.add_edge("supervisor_plan", "extract_query")
    graph.add_edge("extract_query", "diagnose_neo4j")
    graph.add_conditional_edges(
        "diagnose_neo4j",
        _route_after_neo4j,
        {
            "use_fallback": "diagnose_fallback",
            "continue": "select_primary",
        },
    )
    graph.add_edge("diagnose_fallback", "retrieve_evidence")
    graph.add_edge("select_primary", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "generate_patient_questions")
    graph.add_edge("generate_patient_questions", "assess_confidence")
    graph.add_conditional_edges(
        "assess_confidence",
        _route_after_assess,
        {
            "abstain": "abstain_answer",
            "generate": "generate_answer",
        },
    )
    graph.add_edge("abstain_answer", "medical_safety_guard")
    graph.add_edge("generate_answer", "medical_safety_guard")
    graph.add_edge("medical_safety_guard", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
