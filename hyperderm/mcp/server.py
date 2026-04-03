from __future__ import annotations

import json
import sys
from typing import Any

from hyperderm.api.dependencies import AppContainer, create_container
from hyperderm.mcp.langgraph_mcp_tool import LangGraphMCPTool


class HyperDermMCPServer:
    def __init__(self, tool: LangGraphMCPTool) -> None:
        self._tool = tool

    def _tool_schema(self) -> dict[str, Any]:
        return {
            "name": self._tool.name,
            "description": self._tool.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_message": {"type": "string", "description": "User dermatology query"},
                    "session_id": {"type": "string", "description": "Conversation session identifier"},
                    "image_path": {"type": "string", "description": "Optional local image path"},
                },
                "required": ["user_message"],
            },
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "hyperderm-mcp", "version": "0.1.0"},
                    },
                }

            if method == "notifications/initialized":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}

            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": [self._tool_schema()]},
                }

            if method == "tools/call":
                name = str(params.get("name", "")).strip()
                arguments = params.get("arguments") or {}
                if name != self._tool.name:
                    raise ValueError(f"Unknown tool: {name}")
                result = self._tool.invoke(arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result.get("result", result), ensure_ascii=True),
                            }
                        ]
                    },
                }

            raise ValueError(f"Unsupported method: {method}")
        except Exception as error:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32000,
                    "message": str(error),
                },
            }


def build_mcp_server_from_container(container: AppContainer | None = None) -> HyperDermMCPServer:
    if container is None:
        container = create_container()
    return HyperDermMCPServer(tool=container.chat_mcp_tool)


def run_stdio_server() -> None:
    server = build_mcp_server_from_container()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
            sys.stdout.flush()
            continue

        response = server.handle(payload)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
            sys.stdout.flush()
