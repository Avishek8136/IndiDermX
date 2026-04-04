from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

from bytez import Bytez

from hyperderm.core.config import settings

logger = logging.getLogger(__name__)


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


# Global lock to ensure sequential MedGemma API requests (free tier allows 1 request at a time)
_medgemma_api_lock = threading.Lock()

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
        "fungal",
        "tinea",
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
        self._request_lock = _medgemma_api_lock  # Use global lock for rate limiting

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
            "facial": "face",
            "face": "face",
            "fungal": "fungal",
            "fungus": "fungal",
            "fungalinfection": "fungal",
            "fungalinfections": "fungal",
            "ringworm": "annular",
            "ringworms": "annular",
            "ring": "annular",
            "ringshaped": "annular",
            "ringlike": "annular",
            "tinea": "tinea",
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
    def _normalize_message_text(message: str) -> str:
        text = message.lower()
        replacements = {
            "fungal infections": "fungal infection",
            "fungal infection": "fungal",
            "ring worm": "ringworm",
            "ring worms": "ringworm",
            "ring-shaped": "ringshaped",
            "ring like": "ringlike",
            "ring-like": "ringlike",
            "cheek area": "cheeks",
            "cheek region": "cheeks",
            "on the cheeks": "cheeks",
            "on cheeks": "cheeks",
            "face area": "face",
            "on the face": "face",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text

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
        """Call MedGemma API with rate limiting and exponential backoff retry."""
        max_retries = 5
        base_wait_time = 2  # seconds
        
        for attempt in range(max_retries):
            # Acquire lock to ensure sequential requests (free tier: 1 request at a time)
            with self._request_lock:
                try:
                    logger.debug(f"Calling MedGemma LLM API (attempt {attempt + 1}/{max_retries})")
                    result = self._model.run([{"role": "user", "content": prompt}])
                    
                    if result.error:
                        error_msg = str(result.error).lower()
                        # Check for rate limit error (429 or "rate limited" message)
                        if "rate limited" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                            if attempt < max_retries - 1:
                                # Calculate exponential backoff: 2s, 4s, 8s, 16s, 32s
                                wait_time = base_wait_time * (2 ** attempt)
                                logger.warning(
                                    f"MedGemma rate limited. Retrying after {wait_time}s "
                                    f"(attempt {attempt + 1}/{max_retries})"
                                )
                                time.sleep(wait_time)
                                continue  # Retry the request
                            else:
                                logger.error(f"MedGemma rate limited after {max_retries} retries")
                                raise RuntimeError(str(result.error))
                        else:
                            logger.error(f"MedGemma returned error: {result.error}")
                            raise RuntimeError(str(result.error))

                    output = result.output
                    if isinstance(output, dict):
                        content = output.get("content")
                        response = str(content) if content is not None else json.dumps(output)
                    else:
                        response = str(output)
                    logger.debug(f"MedGemma response length: {len(response)} chars")
                    return response
                    
                except RuntimeError:
                    # Re-raise RuntimeError from MedGemma API failures
                    raise
                except Exception as e:
                    error_msg = str(e).lower()
                    # Handle network/connection rate limit errors
                    if "rate limited" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = base_wait_time * (2 ** attempt)
                            logger.warning(
                                f"MedGemma rate limited (connection error). Retrying after {wait_time}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(wait_time)
                            continue  # Retry the request
                        else:
                            logger.error(f"MedGemma rate limited after {max_retries} retries: {str(e)}")
                            raise
                    else:
                        logger.error(f"MedGemma API exception: {str(e)}")
                        raise
        
        # Should not reach here, but just in case
        raise RuntimeError("MedGemma API call failed after all retries")

    def extract_query_features(self, message: str) -> dict[str, Any]:
        normalized_message = self._normalize_message_text(message)
        logger.info(f"Extracting query features from message: '{message}'")
        prompt = (
            "Extract structured dermatology query fields. Return only JSON with keys "
            "descriptors (array), body_part (string), symptoms (array), effects (array).\n"
            f"message: {normalized_message}"
        )
        try:
            logger.debug("Calling MedGemma to extract query features")
            raw = self._run_text(prompt)
            parsed = _extract_json(raw)
            logger.info(f"MedGemma extraction succeeded: {parsed}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MedGemma extraction failed: {str(e)}, falling back to heuristic")
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

            logger.info(f"Extracted features - descriptors: {descriptors}, body_part: {body_part}, symptoms: {symptoms}, effects: {effects}")
            
            return {
                "descriptors": descriptors,
                "body_part": body_part,
                "symptoms": symptoms,
                "effects": effects,
                "source": "medgemma",
            }

        logger.info("Using heuristic feature extraction (fallback from MedGemma)")
        text = normalized_message
        tokens = re.findall(r"[a-zA-Z]+", text)
        normalized = [self._normalize_token(token) for token in tokens]
        descriptors = sorted({token for token in normalized if token in self.DESCRIPTORS})
        symptoms = sorted({token for token in normalized if token in self.SYMPTOMS})
        effects = sorted({token for token in normalized if token in self.EFFECTS})
        body_candidates = [token for token in normalized if token in self.BODY_PARTS]
        body_part = body_candidates[0] if body_candidates else ""

        logger.info(f"Heuristic features - descriptors: {descriptors}, body_part: {body_part}, symptoms: {symptoms}, effects: {effects}")

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
        candidate_reasoning = [
            {
                "disease": item.get("disease", ""),
                "score": float(item.get("score", 0.0)),
                "matched_descriptors": item.get("matched_descriptors", []),
                "matched_body_regions": item.get("matched_body_regions", []),
                "matched_symptoms": item.get("matched_symptoms", []),
                "matched_effects": item.get("matched_effects", []),
                "matched_visual_atoms": item.get("matched_visual_atoms", []),
            }
            for item in candidates[:5]
        ]

        prompt = (
            "You are an experienced dermatologist speaking with a patient in clinic. "
            "Respond in ENGLISH only. "
            "Sound natural, calm, and clinically practical, not robotic. "
            "Use concise plain text with exactly three sections and these labels: "
            "Probable condition:, Why this matches:, Clinical caution:. "
            "Keep each section short (2-4 sentences) and avoid repeating the same phrasing. "
            "In 'Why this matches', explicitly reference matched descriptors/body/symptoms/effects from neo4j_candidate_reasoning. "
            "In 'Why this matches', you may also reference evidence and graph_context details only when they are available. "
            "Address the user directly with empathetic doctor-style language (for example: 'Thanks, that helps' or 'From what you've described'). "
            "If evidence is weak, explicitly say what additional patient details are needed. "
            "If candidate_list is empty, still provide a provisional model-only differential in 'Probable condition' based on user_message and visual_features, "
            "and avoid mentioning internal tools, graph retrieval, or system internals. "
            "If candidate_list is empty and suggested_questions are provided, include 1-2 of those questions conversationally in the 'Clinical caution' section. "
            "Do not invent details. Do not mention unsupported body parts. Do not claim certainty.\n\n"
            f"user_message: {user_message}\n"
            f"top_candidate: {top_candidate or {}}\n"
            f"candidate_list: {candidates[:3]}\n"
            f"neo4j_candidate_reasoning: {candidate_reasoning}\n"
            f"supporting_evidence: {evidence[:5]}\n"
            f"visual_features: {visual_features or {}}\n"
            f"graph_context: {graph_context[:5] if graph_context else []}\n"
            f"memory_summary: {memory_summary}\n"
            f"suggested_questions: {suggested_questions or []}\n"
        )

        try:
            raw = self._run_text(prompt)
            result = self._clean_answer(raw, top_candidate, candidates=candidates, evidence=evidence)
            # If we got empty or invalid output, and no candidates, try direct diagnosis
            if (not result or "could not confidently" in result.lower()) and not top_candidate and not candidates:
                return self._direct_medgemma_diagnosis(user_message, visual_features)
            return result
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MedGemma API call failed: {str(e)}")
            # If main prompt fails and no candidates, try direct diagnosis from MedGemma
            if not top_candidate and not candidates:
                logger.info("Attempting direct MedGemma diagnosis after main prompt failure")
                return self._direct_medgemma_diagnosis(user_message, visual_features)
            # Fallback to deterministic if we have a top candidate
            if top_candidate:
                logger.info("Returning deterministic answer based on top candidate")
                return self._deterministic_candidate_answer(top_candidate, evidence)
            # Last resort
            logger.error("No fallback available, returning generic message")
            return (
                "I could not confidently identify a condition from the available signals. "
                "Please consult a dermatologist for an in-person evaluation."
            )

    def _direct_medgemma_diagnosis(self, user_message: str, visual_features: dict[str, Any] | None = None) -> str:
        """When Neo4j has no candidates, directly ask MedGemma for diagnosis based on symptoms."""
        logger.info("Starting direct MedGemma diagnosis mode (no Neo4j candidates)")
        # Simple, clear prompt for direct diagnosis
        simplified_prompt = (
            "You are a medical AI assistant providing differential diagnosis for a skin condition. "
            "Based on the patient's description, provide likely conditions in a natural doctor-like tone. "
            "Respond with exactly these three sections:\n\n"
            "Probable condition: Give the most likely skin conditions (fungal, inflammatory, infectious, etc.)\n"
            "Why this matches: Explain what in the history supports these possibilities\n"
            "Clinical caution: Include one practical next step and when to seek in-person care\n\n"
            f"Patient says: {user_message}"
        )
        if visual_features and visual_features.get("visual_atoms"):
            simplified_prompt += f"\nVisible features: {visual_features.get('visual_atoms', [])}"

        try:
            logger.info("Calling MedGemma for direct diagnosis")
            raw = self._run_text(simplified_prompt)
            clean = " ".join(raw.strip().split()) if raw else ""
            
            logger.info(f"Direct diagnosis response length: {len(clean)} chars")
            
            # Verify we got real content, not just the model refusing or being empty
            if clean and len(clean) > 50 and "probable condition" in clean.lower():
                logger.info("Direct diagnosis returned substantial content")
                return clean
            
            # If response is too short or missing key sections, try even simpler approach
            if clean and len(clean) > 20:
                logger.info("Direct diagnosis returned minimal but usable content")
                # Still better than nothing
                return clean
            
            logger.warning(f"Direct diagnosis returned insufficient content: {clean[:50]}")
                
        except Exception as e:  # noqa: BLE001
            logger.error(f"Direct diagnosis API call failed: {str(e)}")

        # Fallback to structured response with common conditions
        logger.info("Using fallback diagnosis response")
        return (
            "Probable condition: From your description, likely possibilities include dermatitis (irritant/allergic), fungal infection (tinea), "
            "or another inflammatory rash. "
            "Why this matches: The pattern, symptoms, and progression suggest an inflammatory or infectious skin process, "
            "but there is not enough certainty yet to name one confirmed condition. "
            "Clinical caution: Please avoid scratching, keep the area clean and dry, and arrange an in-person dermatology exam, "
            "especially if it is spreading, painful, oozing, or not improving. "
            "This is decision support only and not a confirmed medical diagnosis."
        )
