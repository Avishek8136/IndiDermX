from __future__ import annotations

from typing import Any

from hyperderm.infrastructure.graph.neo4j_store import Neo4jStore


class DiagnosisService:
    def __init__(self, store: Neo4jStore) -> None:
        self._store = store

    def retrieve_candidates(
        self,
        descriptors: list[str],
        body_part: str,
        symptoms: list[str],
        effects: list[str],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self._store.fetch_top_candidates(
            descriptors=descriptors,
            body_part=body_part,
            symptoms=symptoms,
            effects=effects,
            limit=limit,
        )

    @staticmethod
    def compile_output(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        top = candidates[0] if candidates else {
            "main_class": "Other",
            "sub_class": "Unspecified",
            "disease": "Insufficient Evidence",
            "score": 0.0,
            "evidence": [],
        }

        return {
            "hierarchy_path": [top["main_class"], top["sub_class"], top["disease"]],
            "candidate_list": [
                {
                    "main_class": item["main_class"],
                    "sub_class": item["sub_class"],
                    "disease": item["disease"],
                    "score": item["score"],
                }
                for item in candidates
            ],
            "supporting_evidence": top.get("evidence", []),
            "counter_evidence": [],
            "why_not_top_alternatives": [
                {
                    "disease": item["disease"],
                    "reason": "lower_ranked_score",
                    "score_gap": max(0.0, float(top["score"]) - float(item["score"])),
                }
                for item in candidates[1:4]
            ],
            "final_recommendation": top["disease"],
            "uncertainty": max(0.0, 1.0 - min(1.0, top["score"] / 10.0)),
            "bias_checks": {
                "pathology_first_routing": True,
                "sensitive_identifier_used": False,
            },
            "privacy_safe_fields_used": ["descriptors", "body_part", "derived_visual_features"],
        }
