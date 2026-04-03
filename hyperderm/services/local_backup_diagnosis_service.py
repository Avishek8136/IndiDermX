from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalBackupDiagnosisService:
    def __init__(self, backup_dir: str) -> None:
        self._kg_snapshot_path = Path(backup_dir) / "kg_snapshot.jsonl"

    def _load_disease_rows(self) -> list[dict[str, Any]]:
        if not self._kg_snapshot_path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with self._kg_snapshot_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if record.get("nodeType") != "Disease":
                    continue
                props = record.get("properties") or {}
                rows.append(
                    {
                        "main_class": props.get("mainClass", "Other"),
                        "sub_class": props.get("subClass", "Unspecified"),
                        "disease": props.get("disease", "Insufficient Evidence"),
                        "morphologies": [str(item).lower() for item in props.get("morphologies", [])],
                        "body_regions": [str(item).lower() for item in props.get("bodyRegions", [])],
                    }
                )
        return rows

    def retrieve_candidates(
        self,
        descriptors: list[str],
        body_part: str,
        symptoms: list[str],
        effects: list[str],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        diseases = self._load_disease_rows()
        if not diseases:
            return []

        normalized_desc = [item.strip().lower() for item in descriptors if item.strip()]
        normalized_body = body_part.strip().lower()
        normalized_symptoms = [item.strip().lower() for item in symptoms if item.strip()]
        normalized_effects = [item.strip().lower() for item in effects if item.strip()]

        candidates: list[dict[str, Any]] = []
        for row in diseases:
            morphologies = row.get("morphologies", [])
            body_regions = row.get("body_regions", [])
            disease_name = str(row.get("disease", "")).lower()

            desc_score = sum(1 for item in normalized_desc if item in morphologies)
            body_score = 1 if normalized_body and normalized_body in body_regions else 0
            lexical_score = sum(1 for item in [*normalized_symptoms, *normalized_effects] if item in disease_name)
            total = float(desc_score + body_score + lexical_score)

            if total <= 0:
                continue

            candidates.append(
                {
                    "main_class": row["main_class"],
                    "sub_class": row["sub_class"],
                    "disease": row["disease"],
                    "score": total,
                    "evidence": [{"title": "Local backup match", "source": "backup:kg_snapshot", "doi": ""}],
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]
