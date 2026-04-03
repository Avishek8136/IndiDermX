from __future__ import annotations

import json
import re
from typing import Any

from bytez import Bytez

from hyperderm.core.config import settings


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\\s*", "", stripped)
        stripped = re.sub(r"\\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


class MedGemmaChatService:
    BODY_PARTS = {
        "face",
        "scalp",
        "neck",
        "arm",
        "elbow",
        "hand",
        "chest",
        "back",
        "abdomen",
        "groin",
        "thigh",
        "leg",
        "knee",
        "foot",
        "trunk",
    }

    DESCRIPTORS = {
        "erythema",
        "plaque",
        "scaling",
        "papule",
        "pustule",
        "nodule",
        "patch",
        "spot",
        "macule",
        "hyperpigmentation",
        "comedone",
        "crust",
        "vesicle",
        "burrow",
        "pigmented",
        "atrophy",
        "induration",
        "annular",
    }

    SYMPTOMS = {
        "itch",
        "itchy",
        "itching",
        "pain",
        "burning",
        "dryness",
        "redness",
        "swelling",
        "rash",
    }

    EFFECTS = {
        "scarring",
        "spread",
        "spreading",
        "infection",
        "inflammation",
        "relapse",
        "bleeding",
        "fissure",
    }

    def __init__(self) -> None:
        self._sdk = Bytez(settings.bytez_api_key)
        self._model = self._sdk.model(settings.bytez_model)

    @staticmethod
    def _normalize_token(token: str) -> str:
        t = token.strip().lower()
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        t = re.sub(r"[^a-z]", "", t)
        synonyms = {
            "itchy": "itch",
            "itching": "itch",
            "cheek": "face",
            "cheeks": "face",
            "blackspot": "spot",
            "blackspots": "spot",
            "spots": "spot",
            "patchy": "patch",
            "round": "annular",
            "rounded": "annular",
            "scars": "scarring",
            "spreading": "spread",
        }
        return synonyms.get(t, t)

    @staticmethod
    def _deterministic_candidate_answer(top_candidate: dict[str, Any], evidence: list[dict[str, Any]] | None = None) -> str:
        disease = str(top_candidate.get("disease", "Unknown")).strip() or "Unknown"
        main_class = str(top_candidate.get("main_class", "Other")).strip() or "Other"
        sub_class = str(top_candidate.get("sub_class", "Unspecified")).strip() or "Unspecified"
        score = float(top_candidate.get("score", 0.0))
        confidence = "low" if score < 3.0 else ("moderate" if score < 6.0 else "higher")
        evidence_count = len(evidence or [])
        return (
            f"Most likely condition from available signals is {disease}. "
            f"Hierarchy: {main_class} > {sub_class} > {disease}. "
            f"This is a {confidence}-confidence decision-support estimate"
            f" based on the matched descriptors and location"
            f" with {evidence_count} supporting evidence item(s). "
            "Please confirm with a dermatologist. This is decision support only and not a confirmed medical diagnosis."
        )

    @classmethod
    def _clean_answer(
        cls,
        text: str,
        top_candidate: dict[str, Any] | None,
        candidates: list[dict[str, Any]] | None = None,
        evidence: list[dict[str, Any]] | None = None,
    ) -> str:
        clean = " ".join(text.strip().split())
        if not clean:
            if top_candidate:
                return cls._deterministic_candidate_answer(top_candidate, evidence)
            return (
                "I could not confidently identify a condition from the available signals. "
                "This is decision support only and not a confirmed medical diagnosis."
            )
        # If model ignores language/style instructions and returns noisy text, fall back to deterministic wording.
        if "muhtemelen" in clean.lower() or "kuyru" in clean.lower():
            if top_candidate:
                return cls._deterministic_candidate_answer(top_candidate, evidence)
            return (
                "I could not confidently identify a condition from the available signals. "
                "Please consult a dermatologist. This is decision support only and not a confirmed medical diagnosis."
            )

        if top_candidate:
            top_name = str(top_candidate.get("disease", "")).strip().lower()
            candidate_names = {
                str(item.get("disease", "")).strip().lower()
                for item in (candidates or [])
                if str(item.get("disease", "")).strip()
            }
            candidate_names.add(top_name)

            # Guardrail: if model answer does not mention the selected top diagnosis,
            # or explicitly mentions an out-of-candidate diagnosis, use deterministic text.
            if top_name and top_name not in clean.lower():
                return cls._deterministic_candidate_answer(top_candidate, evidence)

            tokens = {
                token.strip().lower()
                for token in re.findall(r"[A-Za-z][A-Za-z\- ]{3,}", clean)
                if token.strip()
            }
            mentioned_diseases = {
                name for name in candidate_names if name and name in clean.lower()
            }
            if not mentioned_diseases and any("onychomycosis" in token for token in tokens):
                return cls._deterministic_candidate_answer(top_candidate, evidence)

        return clean

    def _run_text(self, prompt: str) -> str:
        result = self._model.run([{"role": "user", "content": prompt}])
        if result.error:
            raise RuntimeError(str(result.error))

        output = result.output
        if isinstance(output, dict):
            content = output.get("content")
            return str(content) if content is not None else json.dumps(output)
        return str(output)

    def extract_query_features(self, message: str) -> dict[str, Any]:
        prompt = (
            "Extract structured dermatology query fields. Return only JSON with keys "
            "descriptors (array), body_part (string), symptoms (array), effects (array).\n"
            f"message: {message}"
        )
        try:
            raw = self._run_text(prompt)
            parsed = _extract_json(raw)
        except Exception:  # noqa: BLE001
            parsed = {}

        if parsed:
            raw_descriptors = [str(item).strip().lower() for item in parsed.get("descriptors", []) if str(item).strip()]
            raw_symptoms = [str(item).strip().lower() for item in parsed.get("symptoms", []) if str(item).strip()]
            raw_effects = [str(item).strip().lower() for item in parsed.get("effects", []) if str(item).strip()]
            raw_body = str(parsed.get("body_part", "")).strip().lower()

            descriptors = sorted({self._normalize_token(item) for item in raw_descriptors if self._normalize_token(item) in self.DESCRIPTORS})
            symptoms = sorted({self._normalize_token(item) for item in raw_symptoms if self._normalize_token(item) in self.SYMPTOMS})
            effects = sorted({self._normalize_token(item) for item in raw_effects if self._normalize_token(item) in self.EFFECTS})
            body_part_norm = self._normalize_token(raw_body)
            body_part = body_part_norm if body_part_norm in self.BODY_PARTS else ""

            return {
                "descriptors": descriptors,
                "body_part": body_part,
                "symptoms": symptoms,
                "effects": effects,
                "source": "medgemma",
            }

        text = message.lower()
        tokens = re.findall(r"[a-zA-Z]+", text)
        normalized = [self._normalize_token(token) for token in tokens]
        descriptors = sorted({token for token in normalized if token in self.DESCRIPTORS})
        symptoms = sorted({token for token in normalized if token in self.SYMPTOMS})
        effects = sorted({token for token in normalized if token in self.EFFECTS})
        body_candidates = [token for token in normalized if token in self.BODY_PARTS]
        body_part = body_candidates[0] if body_candidates else ""

        return {
            "descriptors": descriptors,
            "body_part": body_part,
            "symptoms": symptoms,
            "effects": effects,
            "source": "heuristic",
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
        prompt = (
            "You are a dermatology decision-support chatbot. "
            "Respond in ENGLISH only. "
            "Use concise plain text with exactly three parts: probable condition, short rationale grounded only in provided candidates/evidence, and caution to consult a clinician. "
            "Do not invent details. Do not mention unsupported body parts. Do not claim certainty.\n\n"
            f"user_message: {user_message}\n"
            f"top_candidate: {top_candidate or {}}\n"
            f"candidate_list: {candidates[:3]}\n"
            f"supporting_evidence: {evidence[:5]}\n"
            f"visual_features: {visual_features or {}}\n"
            f"graph_context: {graph_context[:3] if graph_context else []}\n"
            f"memory_summary: {memory_summary}\n"
            f"suggested_questions: {suggested_questions or []}\n"
        )

        try:
            raw = self._run_text(prompt)
            return self._clean_answer(raw, top_candidate, candidates=candidates, evidence=evidence)
        except Exception:  # noqa: BLE001
            if not top_candidate:
                return (
                    "I could not confidently identify a condition from the available signals. "
                    "Please consult a dermatologist for an in-person evaluation."
                )
            return self._deterministic_candidate_answer(top_candidate, evidence)
