from __future__ import annotations

import os
import threading
import time
from functools import lru_cache
from typing import Any, Callable

from hyperderm.core.config import settings


Validator = Callable[[str], bool]
_bytez_call_lock = threading.Lock()
_last_bytez_call_at = 0.0


@lru_cache(maxsize=1)
def get_bytez_model() -> Any:
    try:
        from bytez import Bytez
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("Bytez package is not available") from error

    api_key = settings.bytez_api_key or os.getenv("BYTEZ_API_KEY")
    if not api_key:
        raise RuntimeError("Bytez token missing. Set BYTEZ_API_KEY in .env")

    model_id = settings.bytez_model or os.getenv("BYTEZ_MODEL") or "ASAIs-TDDI-2025/MedTurk-MedGemma-4b"
    sdk = Bytez(api_key)
    return sdk.model(model_id)


def _extract_bytez_text(result: Any) -> str:
    if result is None:
        return ""
    output = getattr(result, "output", result)
    if isinstance(output, dict):
        content = output.get("content")
        if content is not None:
            return str(content)
        generated_text = output.get("generated_text")
        if generated_text is not None:
            return str(generated_text)
        return str(output)
    return str(output or "")


def run_text_with_fallback(
    prompt: str,
    *,
    validator: Validator | None = None,
    temperature: float = 0.2,
    timeout_seconds: int | None = None,
) -> tuple[str, str, str | None]:
    _ = temperature, timeout_seconds
    validate = validator or (lambda text: bool(text.strip()))

    max_retries = max(int(settings.bytez_max_retries), 1)
    min_interval = max(float(settings.bytez_min_interval_seconds), 0.0)

    last_error = "unknown_error"
    for attempt in range(1, max_retries + 1):
        try:
            model = get_bytez_model()
            with _bytez_call_lock:
                global _last_bytez_call_at
                now = time.monotonic()
                wait_for = (_last_bytez_call_at + min_interval) - now
                if wait_for > 0:
                    time.sleep(wait_for)

                result = model.run([
                    {"role": "user", "content": prompt},
                ])
                _last_bytez_call_at = time.monotonic()

            if getattr(result, "error", None):
                last_error = str(getattr(result, "error"))
                lowered = last_error.lower()
                if attempt < max_retries and ("rate limited" in lowered or "429" in lowered):
                    time.sleep(min(6.0, 1.2 * attempt))
                    continue
                return "", "bytez", last_error

            bytez_text = _extract_bytez_text(result)
            if validate(bytez_text):
                return bytez_text, "bytez", None
            last_error = "invalid_or_empty_bytez_response"
            if attempt < max_retries:
                time.sleep(min(4.0, 0.8 * attempt))
                continue
            return "", "bytez", last_error
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
            lowered = last_error.lower()
            if attempt < max_retries and ("rate limited" in lowered or "429" in lowered):
                time.sleep(min(6.0, 1.2 * attempt))
                continue
            if attempt < max_retries:
                time.sleep(min(3.0, 0.6 * attempt))
                continue
            return "", "bytez", last_error

    return "", "bytez", last_error
