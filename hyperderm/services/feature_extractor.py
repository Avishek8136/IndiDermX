from __future__ import annotations

import hashlib
import json
import time
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache
from pathlib import Path
from typing import Any

from bytez import Bytez

from hyperderm.core.config import settings


def _log_bytez_call(event: dict[str, Any]) -> None:
    log_path = Path(settings.backup_dir) / "bytez_calls.jsonl"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with log_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=True) + "\n")


@lru_cache(maxsize=1)
def _get_model():
    sdk = Bytez(settings.bytez_api_key)
    return sdk.model(settings.bytez_model)


def _parse_json_object(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, str):
            return _parse_json_object(content)
        return output

    if not isinstance(output, str):
        return {}

    text = output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Recover JSON if model wraps output in extra text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _normalize_descriptor_tokens(values: list[Any]) -> list[str]:
    synonym_map = {
        "red": "erythema",
        "scaly": "scaling",
        "pimples": "papule",
        "pimple": "papule",
        "whiteheads": "comedone",
        "blackheads": "comedone",
    }

    def _canonical_atom(raw: str) -> str:
        token = raw.strip().lower()
        token = re.sub(r"\s+", " ", token)
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "pus", "ous")):
            token = token[:-1]
        token = synonym_map.get(token, token)
        return token.strip()

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        parts = re.split(r"[,;|/]", str(value))
        for part in parts:
            token = _canonical_atom(part)
            if not token or token in seen:
                continue
            seen.add(token)
            normalized.append(token)
    return normalized


def _compact_error_code(message: str | None) -> str:
    if not message:
        return "unknown"
    lowered = message.lower()
    if "you only have 0 open" in lowered or "rate limits" in lowered or "gpu seats" in lowered:
        return "seat_limit"
    if "timed out" in lowered:
        return "timeout"
    if "remotedisconnected" in lowered or "connection aborted" in lowered:
        return "network"
    return "bytez_error"


def extract_visual_features(image_path: str | None, descriptors: list[str], disease_name: str | None = None) -> dict[str, Any]:
    descriptor_tokens = _normalize_descriptor_tokens(descriptors)
    image_ref = image_path or "not_provided"
    disease_ref = (disease_name or "unknown_disease").strip().lower()

    prompt = (
        "You are extracting dermatology visual features for graph storage. "
        "Given the provided image reference and descriptors, return ONLY a JSON object with keys: "
        "mu_ref (string), kappa (number 0-100), descriptor_tokens (array of strings), "
        "morphology_summary (string).\n"
        f"image_reference: {image_ref}\n"
        f"descriptors: {descriptor_tokens}"
    )

    model = _get_model()
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    def _invoke_once():
        # Standard Bytez model call without scale-up.
        result = model.run(messages)

        return result

    def _invoke_with_timeout(timeout_seconds: int):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_invoke_once)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as error:
            # Do not block waiting for a stuck worker thread.
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"Bytez call timed out after {timeout_seconds}s") from error
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    results = None
    last_error = None
    for attempt in range(1, 5):
        try:
            results = _invoke_with_timeout(settings.bytez_timeout_seconds)
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
            _log_bytez_call(
                {
                    "status": "exception",
                    "attempt": attempt,
                    "model": settings.bytez_model,
                    "image_path_ref": image_ref,
                    "error": last_error,
                }
            )
            time.sleep(min(8, 2 * attempt))
            continue

        if not results.error:
            _log_bytez_call(
                {
                    "status": "success",
                    "attempt": attempt,
                    "model": settings.bytez_model,
                    "image_path_ref": image_ref,
                }
            )
            break

        last_error = str(results.error)
        _log_bytez_call(
            {
                "status": "error",
                "attempt": attempt,
                "model": settings.bytez_model,
                "image_path_ref": image_ref,
                "error": last_error,
            }
        )
        transient_markers = ["RemoteDisconnected", "Connection aborted", "timed out", "502", "503", "504"]
        if any(marker in last_error for marker in transient_markers) and attempt < 4:
            time.sleep(min(8, 2 * attempt))
            continue

        break

    if results is None or results.error:
        # Keep ingestion moving even when Bytez has transient outages/seat blocks.
        signature = "|".join(descriptor_tokens)
        fallback_ref = hashlib.sha256(f"{disease_ref}|{signature}".encode("utf-8")).hexdigest()[:24]
        error_code = _compact_error_code(last_error)
        morphology_summary = ", ".join(descriptor_tokens[:6]) if descriptor_tokens else "unknown_morphology"
        return {
            "feature_id": f"vf:{fallback_ref}",
            "feature_natural_key": f"{disease_ref}|{'|'.join(descriptor_tokens)}",
            "mu_ref": f"fallback:{fallback_ref}",
            "kappa": float(5 + max(len(descriptor_tokens), 1)),
            "descriptor_tokens": descriptor_tokens,
            "descriptor_signature": " | ".join(descriptor_tokens),
            "descriptor_count": len(descriptor_tokens),
            "condition_name": disease_name or "Unknown Disease",
            "condition_key": f"cond:{disease_ref.replace(' ', '_')}",
            "morphology_summary": morphology_summary,
            "extracted_by": "fallback",
            "extraction_status": "fallback",
            "extraction_error_code": error_code,
            "extraction_error_detail": last_error or "",
        }

    parsed = _parse_json_object(results.output)
    if not parsed:
        parsed = {
            "mu_ref": f"parsed_fallback:{hashlib.sha256(image_ref.encode('utf-8')).hexdigest()[:12]}",
            "kappa": 10,
            "descriptor_tokens": descriptor_tokens,
            "morphology_summary": "fallback_non_json_output",
        }

    raw_kappa = parsed.get("kappa", 10)
    try:
        kappa = float(raw_kappa)
    except (TypeError, ValueError):
        kappa = 10.0

    kappa = max(0.0, min(100.0, kappa))
    parsed_tokens = parsed.get("descriptor_tokens", descriptor_tokens)
    if not isinstance(parsed_tokens, list):
        parsed_tokens = descriptor_tokens
    curated_tokens = _normalize_descriptor_tokens(parsed_tokens)
    if not curated_tokens:
        curated_tokens = descriptor_tokens

    stable_ref_source = f"{disease_ref}|{'|'.join(curated_tokens)}"
    feature_id = hashlib.sha256(stable_ref_source.encode("utf-8")).hexdigest()[:24]

    morphology_summary = str(parsed.get("morphology_summary", "")).strip()
    if not morphology_summary:
        morphology_summary = ", ".join(curated_tokens[:6]) if curated_tokens else "unknown_morphology"

    return {
        "feature_id": f"vf:{feature_id}",
        "feature_natural_key": f"{disease_ref}|{'|'.join(curated_tokens)}",
        "mu_ref": str(parsed.get("mu_ref", f"medgemma:{feature_id}")),
        "kappa": kappa,
        "descriptor_tokens": curated_tokens,
        "descriptor_signature": " | ".join(curated_tokens),
        "descriptor_count": len(curated_tokens),
        "condition_name": disease_name or "Unknown Disease",
        "condition_key": f"cond:{disease_ref.replace(' ', '_')}",
        "morphology_summary": morphology_summary,
        "extracted_by": settings.bytez_model,
        "extraction_status": "success",
        "extraction_error_code": "",
        "extraction_error_detail": "",
    }
