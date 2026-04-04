from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalBackupDiagnosisService:
    def __init__(self, backup_dir: str) -> None:
        self._kg_snapshot_path = Path(backup_dir) / "kg_snapshot.jsonl"
        self._ensure_snapshot_exists()

    def _ensure_snapshot_exists(self) -> None:
        if self._kg_snapshot_path.exists():
            return
        self._kg_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self._default_disease_rows()
        with self._kg_snapshot_path.open("w", encoding="utf-8") as file_obj:
            for row in rows:
                record = {
                    "nodeType": "Disease",
                    "properties": {
                        "mainClass": row["main_class"],
                        "subClass": row["sub_class"],
                        "disease": row["disease"],
                        "morphologies": row["morphologies"],
                        "bodyRegions": row["body_regions"],
                    },
                }
                file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _load_disease_rows(self) -> list[dict[str, Any]]:
        if not self._kg_snapshot_path.exists():
            return self._default_disease_rows()

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
        return rows if rows else self._default_disease_rows()

    @staticmethod
    def _default_disease_rows() -> list[dict[str, Any]]:
        return [
            {
                "main_class": "Infectious Disorders",
                "sub_class": "Infectious skin conditions -Fungal",
                "disease": "Tinea Corporis",
                "morphologies": ["annular", "patch", "scaling", "erythema", "fungal", "tinea"],
                "body_regions": ["face", "neck", "arm", "trunk", "leg", "groin"],
                "symptoms": ["itch", "redness", "rash"],
                "effects": ["spread", "inflammation", "scarring"],
            },
            {
                "main_class": "Inflammatory Disorders",
                "sub_class": "Inflammatory skin diseases (Eczema and Dermatitis)",
                "disease": "Contact Dermatitis",
                "morphologies": ["patch", "erythema", "scaling", "papule", "spot"],
                "body_regions": ["face", "neck", "hand", "arm", "trunk"],
                "symptoms": ["itch", "burning", "redness", "rash"],
                "effects": ["inflammation", "fissure", "scarring"],
            },
            {
                "main_class": "Inflammatory Disorders",
                "sub_class": "Psoriasiform disorders",
                "disease": "Plaque Psoriasis",
                "morphologies": ["plaque", "scaling", "erythema", "patch"],
                "body_regions": ["elbow", "knee", "scalp", "trunk"],
                "symptoms": ["itch", "dryness", "redness"],
                "effects": ["scarring", "inflammation"],
            },
            {
                "main_class": "Skin Appendages Disorders",
                "sub_class": "Sebacious Glands and Acneiform Disorders",
                "disease": "Acne",
                "morphologies": ["papule", "pustule", "nodule", "spot", "comedone"],
                "body_regions": ["face", "chest", "back"],
                "symptoms": ["pain", "redness", "rash"],
                "effects": ["scarring", "inflammation"],
            },
        ]

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
            symptom_hints = [str(item).lower() for item in row.get("symptoms", [])]
            effect_hints = [str(item).lower() for item in row.get("effects", [])]
            disease_name = str(row.get("disease", "")).lower()
            subclass_name = str(row.get("sub_class", "")).lower()
            main_class_name = str(row.get("main_class", "")).lower()

            desc_score = sum(1 for item in normalized_desc if item in morphologies)
            body_score = 1 if normalized_body and normalized_body in body_regions else 0
            symptom_score = sum(1 for item in normalized_symptoms if item in symptom_hints)
            effect_score = sum(1 for item in normalized_effects if item in effect_hints)
            lexical_score = sum(
                1
                for item in [*normalized_desc, *normalized_symptoms, *normalized_effects]
                if item and (item in disease_name or item in subclass_name or item in main_class_name)
            )
            total = float(desc_score + body_score + symptom_score + effect_score + lexical_score)

            if total <= 0:
                continue

            matched_descriptors = [item for item in normalized_desc if item in morphologies]
            matched_body = [normalized_body] if body_score else []
            matched_symptoms = [item for item in normalized_symptoms if item in symptom_hints]
            matched_effects = [item for item in normalized_effects if item in effect_hints]

            candidates.append(
                {
                    "main_class": row["main_class"],
                    "sub_class": row["sub_class"],
                    "disease": row["disease"],
                    "score": total,
                    "matched_descriptors": matched_descriptors,
                    "matched_body_regions": matched_body,
                    "matched_symptoms": matched_symptoms,
                    "matched_effects": matched_effects,
                    "matched_visual_atoms": [],
                    "evidence": [{"title": "Local backup match", "source": "backup:kg_snapshot", "doi": ""}],
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]
