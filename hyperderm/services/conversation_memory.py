from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConversationMemoryService:
    def __init__(self, backup_dir: str) -> None:
        self._path = Path(backup_dir) / "conversation_memory.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        with self._path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")

    def load_recent(self, session_id: str, limit: int = 8) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("session_id", "")) != session_id:
                    continue
                rows.append(row)
        return rows[-limit:]

    def summary(self, session_id: str, limit: int = 8) -> str:
        rows = self.load_recent(session_id, limit=limit)
        if not rows:
            return ""
        segments: list[str] = []
        for row in rows:
            role = str(row.get("role", "user")).strip().lower()
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            label = "User" if role == "user" else "Assistant"
            segments.append(f"{label}: {content}")
        return "\n".join(segments)
