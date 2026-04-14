from __future__ import annotations

import json
import re
from typing import Any

from hyperderm.core.config import settings
from hyperderm.services.model_clients import run_text_with_fallback


def _parse_json_object(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        # Chat completions can return a wrapper: {role, content}. Parse JSON from content.
        content = output.get("content")
        if isinstance(content, str):
            return _parse_json_object(content)
        return output

    if not isinstance(output, str):
        return {}

    text = output.strip()
    # Accept common fenced responses from chat models.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _normalize_tokens(values: list[Any]) -> list[str]:
    synonym_map = {
        "pimple": "papule",
        "pimples": "papule",
        "whitehead": "comedone",
        "blackhead": "comedone",
        "red": "erythema",
        "scaly": "scaling",
        "scar": "scarring",
        "itchy": "itch",
    }

    noise_tokens = {
        "skin",
        "lesion",
        "change",
        "problem",
        "finding",
        "sign",
        "symptom",
    }

    def _canonical_atom(raw: str) -> str:
        token = raw.strip().lower()
        # Keep only ascii letters/spaces to avoid noisy multilingual/free-form generations.
        token = token.encode("ascii", "ignore").decode("ascii")
        token = re.sub(r"[^a-z\s\-]", " ", token)
        token = re.sub(r"\s+", " ", token)
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "pus", "ous")):
            token = token[:-1]
        token = synonym_map.get(token, token)
        return token.strip()

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        parts = re.split(r"[,;|/]", str(value))
        for part in parts:
            token = _canonical_atom(part)
            if not token or token in noise_tokens or token in seen:
                continue
            # Skip very long/free-form fragments, keep compact reusable node labels.
            if len(token) < 3 or len(token) > 28:
                continue
            if " " in token and len(token.split()) > 3:
                continue
            seen.add(token)
            output.append(token)
    return output[:12]


SYMPTOM_KEYWORDS = [
    "itch",
    "pruritus",
    "pain",
    "burning",
    "redness",
    "erythema",
    "swelling",
    "rash",
    "scaling",
    "scale",
    "plaque",
    "papule",
    "pustule",
    "nodule",
    "vesicle",
    "blister",
    "crust",
    "dryness",
    "lichenification",
    "excoriation",
    "pigmentation",
]

EFFECT_KEYWORDS = [
    "scarring",
    "hyperpigmentation",
    "hypopigmentation",
    "disfigurement",
    "infection",
    "inflammation",
    "ulceration",
    "bleeding",
    "fissure",
    "relapse",
    "chronic",
    "psychological distress",
]

DISEASE_FALLBACKS: dict[str, dict[str, list[str]]] = {
    "acne": {
        "symptoms": ["papule", "pustule", "nodule", "comedone", "redness"],
        "effects": ["scarring", "hyperpigmentation", "inflammation"],
    },
    "tinea": {
        "symptoms": ["itch", "scaling", "erythema", "annular rash"],
        "effects": ["spread", "relapse", "hyperpigmentation"],
    },
    "eczema": {
        "symptoms": ["itch", "dryness", "erythema", "lichenification"],
        "effects": ["excoriation", "infection", "chronic"],
    },
    "psoriasis": {
        "symptoms": ["plaque", "scaling", "erythema", "itch"],
        "effects": ["fissure", "bleeding", "chronic"],
    },
    "vitiligo": {
        "symptoms": ["depigmented patch", "well-defined patch"],
        "effects": ["psychological distress", "sun sensitivity"],
    },
    "dermatitis": {
        "symptoms": ["itch", "rash", "erythema", "burning"],
        "effects": ["lichenification", "excoriation", "infection"],
    },
    "scabies": {
        "symptoms": ["severe itch", "papule", "burrow", "rash"],
        "effects": ["secondary infection", "excoriation", "sleep disturbance"],
    },
}


def _fallback_from_context(context_text: str) -> dict[str, list[str]]:
    text = context_text.lower()
    symptoms = [token for token in SYMPTOM_KEYWORDS if token in text]
    effects = [token for token in EFFECT_KEYWORDS if token in text]
    return {
        "symptoms": _normalize_tokens(symptoms),
        "effects": _normalize_tokens(effects),
    }


def _fallback_from_disease_name(disease_name: str) -> dict[str, list[str]]:
    lowered = disease_name.lower()
    for key, value in DISEASE_FALLBACKS.items():
        if key in lowered:
            return {
                "symptoms": _normalize_tokens(value.get("symptoms", [])),
                "effects": _normalize_tokens(value.get("effects", [])),
            }
    return {"symptoms": ["rash", "itch", "erythema"], "effects": ["inflammation"]}


class SymptomEffectService:
    def __init__(self) -> None:
        self._model_id = settings.bytez_model
        self._timeout_seconds = int(settings.bytez_timeout_seconds)

    def _run_text(self, prompt: str, provider_mode: str = "bytez_only") -> tuple[str, bool, str, str | None]:
        text, source, error = run_text_with_fallback(
            prompt,
            timeout_seconds=self._timeout_seconds,
            temperature=0.2,
            validator=lambda output: bool(output.strip()),
        )
        return text, bool(text.strip()), source, error

    def generate(self, disease_name: str, context_text: str, provider_mode: str = "bytez_only") -> dict[str, Any]:
        prompt_with_context = (
            "Extract concise dermatology symptom and effect tokens for graph nodes. "
            "Return only JSON with keys: symptoms (array of short strings), effects (array of short strings). "
            "Use dermatology terminology and short reusable tokens. "
            "Do not include diagnosis labels or treatment advice.\n"
            f"disease: {disease_name}\n"
            f"context: {context_text[:5000]}"
        )
        prompt_disease_only = (
            "You are a dermatology clinical knowledge model. "
            "Generate the most likely symptoms and clinical effects/complications for the disease below "
            "using your dermatology knowledge, even if context is limited. "
            "Return only JSON with keys: symptoms (array of short strings), effects (array of short strings).\n"
            f"disease: {disease_name}"
        )
        prompt_disease_strict = (
            "You are a dermatology clinical extraction assistant. "
            "Return ONLY valid JSON with keys: disease, symptoms, effects. "
            "Symptoms and effects must be atomic short reusable tokens in lowercase. "
            "No markdown and no commentary.\n"
            f"Disease: {disease_name}"
        )

        fallback_context = _fallback_from_context(context_text)
        fallback_disease = _fallback_from_disease_name(disease_name)

        def _call_model(prompt: str) -> dict[str, Any]:
            output_text, ok, source, error = self._run_text(prompt, provider_mode=provider_mode)
            if not ok:
                return {"symptoms": [], "effects": [], "ok": False, "source": source, "error": error}

            parsed = _parse_json_object(output_text)
            symptoms = parsed.get("symptoms", [])
            effects = parsed.get("effects", [])
            if not isinstance(symptoms, list):
                symptoms = []
            if not isinstance(effects, list):
                effects = []
            return {
                "symptoms": _normalize_tokens(symptoms),
                "effects": _normalize_tokens(effects),
                "ok": True,
                "source": source,
            }

        primary = _call_model(prompt_with_context)
        if primary["symptoms"] or primary["effects"]:
            return {
                "symptoms": primary["symptoms"],
                "effects": primary["effects"],
                "generation_mode": f"{primary.get('source', 'bytez')}_context",
                "bytez_attempted": primary.get("source") == "bytez",
                "bytez_success": primary.get("source") == "bytez",
                "bytez_attempted_modes": ["context"],
            }

        secondary = _call_model(prompt_disease_only)
        if secondary["symptoms"] or secondary["effects"]:
            return {
                "symptoms": secondary["symptoms"],
                "effects": secondary["effects"],
                "generation_mode": f"{secondary.get('source', 'bytez')}_disease_only",
                "bytez_attempted": secondary.get("source") == "bytez",
                "bytez_success": secondary.get("source") == "bytez",
                "bytez_attempted_modes": ["context", "disease_only"],
            }

        strict = _call_model(prompt_disease_strict)
        if strict["symptoms"] or strict["effects"]:
            return {
                "symptoms": strict["symptoms"],
                "effects": strict["effects"],
                "generation_mode": f"{strict.get('source', 'bytez')}_disease_strict",
                "bytez_attempted": strict.get("source") == "bytez",
                "bytez_success": strict.get("source") == "bytez",
                "bytez_attempted_modes": ["context", "disease_only", "disease_strict"],
            }

        if fallback_context["symptoms"] or fallback_context["effects"]:
            return {
                **fallback_context,
                "generation_mode": "context_fallback",
                "bytez_attempted": False,
                "bytez_success": False,
                "bytez_attempted_modes": ["context", "disease_only", "disease_strict"],
            }
        return {
            **fallback_disease,
            "generation_mode": "disease_fallback",
            "bytez_attempted": False,
            "bytez_success": False,
            "bytez_attempted_modes": ["context", "disease_only", "disease_strict"],
        }
