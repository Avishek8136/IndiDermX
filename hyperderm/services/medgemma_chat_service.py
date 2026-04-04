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
        evidence_count = len(evidence or [])
        if evidence_count == 0:
            return (
                "I do not have enough reliable evidence yet to name a specific condition. "
                "Please share a bit more detail about appearance, location, duration, and symptoms, "
                "or upload a clear image so I can guide you better. "
                "This is decision support only and not a confirmed medical diagnosis."
            )
        return (
            f"Based on what you have shared, this might be {disease}. "
            "I cannot confirm this in chat alone, so please monitor for worsening and seek an in-person dermatology exam if needed. "
            "This is decision support only and not a confirmed medical diagnosis."
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

    @staticmethod
    def _detect_advice_intent(user_message: str) -> bool:
        """Detect if user is asking for management/care advice rather than diagnosis."""
        message_lower = user_message.lower()
        advice_keywords = [
            "avoid", "care", "what should", "tips", "prevent",
            "treat", "manage", "do i", "how to", "what can i",
            "should i", "wash", "apply", "dry", "keep", "don't",
            "guidance", "recommendation", "best practice"
        ]
        return any(keyword in message_lower for keyword in advice_keywords)

    @staticmethod
    def _extract_asked_questions_from_history(memory_summary: str) -> set[str]:
        """Extract previously asked questions from memory summary to avoid repetition."""
        asked = set()
        history_lines = memory_summary.lower().split("\n")
        question_indicators = ["?", "asked", "question", "asked about", "do you"]
        for line in history_lines:
            if any(indicator in line for indicator in question_indicators):
                # Extract question words to avoid similar questions
                words = line.split()
                if words:
                    asked.add(" ".join(words[:3]))  # Store first 3 words as topic
        return asked

    def _generate_advice_response(self, user_message: str, top_candidate: dict[str, Any] | None) -> str:
        """Generate practical management and care advice based on suspected condition."""
        disease = str(top_candidate.get("disease", "")).strip().lower() if top_candidate else ""
        
        # General fungal infection advice
        if "fungal" in disease or "tinea" in disease or "ringworm" in disease:
            advice = (
                "**Management Tips for Fungal Infections:**\n\n"
                "- Keep the area clean and dry (fungal thrive in moisture)\n"
                "- Wash gently with mild soap daily\n"
                "- Pat dry thoroughly, especially in folds\n"
                "- Wear breathable, loose-fitting clothing\n"
                "- Avoid sharing washcloths, towels, or personal items\n"
                "- Wash hands thoroughly after touching the affected area\n"
                "- Consider using antifungal powder (OTC) after consulting product instructions\n"
                "- Keep fingernails trimmed to avoid scratching and spreading infection\n"
                "- Monitor for spread; if worsening, seek dermatology evaluation\n"
                "- Avoid swimming in public pools until cleared by a doctor\n\n"
                "⚠️ This is general guidance only. Confirm with a dermatologist before using any treatments."
            )
        # General inflammatory/dermatitis advice
        elif any(term in disease for term in ["dermatitis", "eczema", "psoriasis", "inflammatory"]):
            advice = (
                "**Management Tips for Inflammatory Skin Conditions:**\n\n"
                "- Avoid known triggers (harsh soaps, allergens, irritants)\n"
                "- Use fragrance-free, hypoallergenic moisturizer regularly\n"
                "- Apply moisturizer to damp skin within 3 minutes of bathing\n"
                "- Use lukewarm (not hot) water for washing\n"
                "- Avoid excessive cleaning or scrubbing\n"
                "- Wear soft, breathable fabrics (cotton when possible)\n"
                "- Keep stress low (stress can worsen conditions)\n"
                "- Patch test any new products on a small area first\n"
                "- If itching is severe, avoid scratching; consider wearing gloves at night\n"
                "- Track what seems to trigger or improve the condition\n\n"
                "⚠️ If symptoms worsen or persist beyond 2 weeks, consult a dermatologist."
            )
        else:
            # Generic advice
            advice = (
                "**General Skin Care Recommendations:**\n\n"
                "- Keep the area clean and dry\n"
                "- Avoid scratching or picking\n"
                "- Use gentle, fragrance-free cleaners\n"
                "- Apply moisturizer as needed\n"
                "- Avoid irritants and known allergens\n"
                "- Monitor for changes or worsening\n"
                "- See a dermatologist if not improving in 1-2 weeks\n\n"
                "⚠️ This is assistive guidance only. A proper diagnosis requires an in-person exam."
            )
        return advice

    def generate_chat_answer(
        self,
        user_message: str,
        top_candidate: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        visual_features: dict[str, Any] | None = None,
        graph_context: list[dict[str, Any]] | None = None,
        memory_summary: str = "",
        recent_messages: list[dict[str, Any]] | None = None,
        suggested_questions: list[str] | None = None,
    ) -> str:
        # Check if user is asking for advice/management rather than diagnosis
        is_advice_request = self._detect_advice_intent(user_message)
        if is_advice_request and top_candidate:
            logger.info("User requesting advice/management guidance")
            return self._generate_advice_response(user_message, top_candidate)

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

        # Build conversation history context
        recent_context = ""
        if recent_messages:
            recent_context = "Recent conversation history:\n"
            for msg in recent_messages[-6:]:  # Last 6 messages for context
                role = str(msg.get("role", "")).capitalize()
                content = str(msg.get("content", "")).strip()[:300]  # Truncate long messages
                recent_context += f"{role}: {content}\n"
            recent_context += "\n"

        # Identify previously asked questions to avoid repetition
        asked_questions = self._extract_asked_questions_from_history(memory_summary) if memory_summary else set()

        prompt = (
            "You are a helpful, empathetic dermatology assistant in a clinic setting.\n\n"
            "IMPORTANT RULES:\n"
            "1. ANSWER THE USER'S DIRECT QUESTION FIRST before asking your own questions.\n"
            "2. NEVER ask the same question twice. Review the conversation history below and avoid asking about things already addressed.\n"
            "3. Ask ONE follow-up question at a time to gather missing diagnostic information.\n"
            "4. Use natural, empathetic language. Avoid robotic templates.\n"
            "5. Respond in ENGLISH only.\n\n"
            "RESPONSE STRUCTURE:\n"
            "Start with your conversational response addressing the current message.\n"
            "If more information is needed, ask a SINGLE focused follow-up question.\n"
            "End with: '---\nThis is decision support only and not a confirmed medical diagnosis.'"
            "\n\n"
            f"{recent_context}"
            f"Current user message: {user_message}\n"
            f"Previously discussed topics (avoid re-asking): {asked_questions or 'none yet'}\n\n"
            f"Available diagnostic reasoning:\n"
            f"  Top candidate: {top_candidate or 'None'}\n"
            f"  Other candidates: {candidates[:2] if candidates else 'None'}\n"
            f"  Supporting evidence: {len(evidence or [])} items\n"
            f"  Visual features: {visual_features.get('visual_atoms', []) if visual_features else 'None'}\n\n"
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
                "I'm having some difficulty processing that. Let me try a different approach.\n\n"
                "Based on what you've described, the most important next step is to consult with a dermatologist "
                "for an in-person evaluation, especially if the condition is worsening.\n\n"
                "---\n"
                "This is decision support only and not a confirmed medical diagnosis."
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
