from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalRagService:
    def __init__(self, backup_dir: str) -> None:
        self._evidence_path = Path(backup_dir) / "evidence_cards.jsonl"

    def retrieve(self, query_terms: list[str], disease_hint: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        if not self._evidence_path.exists():
            return []

        normalized_terms = [term.strip().lower() for term in query_terms if term and term.strip()]
        disease_key = (disease_hint or "").strip().lower()

        scored: list[tuple[int, dict[str, Any]]] = []
        with self._evidence_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                row_disease = str(row.get("disease", "")).strip().lower()
                if disease_key and row_disease != disease_key:
                    continue

                title = str(row.get("title", "")).strip().lower()
                journal = str(row.get("journal", "")).strip().lower()
                joined = f"{row_disease} {title} {journal}"

                score = 0
                if disease_key and disease_key in joined:
                    score += 2
                for term in normalized_terms:
                    if term in joined:
                        score += 1

                if score == 0 and not disease_key:
                    continue

                scored.append(
                    (
                        score,
                        {
                            "title": row.get("title", ""),
                            "source": row.get("evidence_id") or row.get("source", "LocalBackup"),
                            "doi": row.get("doi", ""),
                            "disease": row.get("disease", ""),
                        },
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]
