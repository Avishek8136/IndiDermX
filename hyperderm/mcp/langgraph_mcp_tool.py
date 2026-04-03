from __future__ import annotations

from typing import Any


class LangGraphMCPTool:
    def __init__(self, name: str, description: str, graph: Any) -> None:
        self.name = name
        self.description = description
        self._graph = graph

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._graph.invoke(arguments)
        final = state.get("final", state)
        return {
            "tool": self.name,
            "description": self.description,
            "result": final,
        }
