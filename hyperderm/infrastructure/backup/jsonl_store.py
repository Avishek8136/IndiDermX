from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlBackupStore:
    def __init__(self, backup_dir: str) -> None:
        self.base = Path(backup_dir)
        self.base.mkdir(parents=True, exist_ok=True)

        self.kg_snapshot = self.base / "kg_snapshot.jsonl"
        self.evidence_cards = self.base / "evidence_cards.jsonl"
        self.inference_audit = self.base / "inference_audit.jsonl"
        self.gap_audit = self.base / "gap_audit.jsonl"
        self.schema_version = self.base / "schema_version.json"

    def append(self, file_path: Path, record: dict[str, Any]) -> None:
        with file_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")

    def write_schema_version(self, schema_version: str, pipeline_version: str) -> None:
        payload = {
            "schemaVersion": schema_version,
            "pipelineVersion": pipeline_version,
        }
        self.schema_version.write_text(json.dumps(payload, indent=2), encoding="utf-8")
