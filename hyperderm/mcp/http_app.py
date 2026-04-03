from __future__ import annotations

from fastapi import FastAPI

from hyperderm.mcp.server import build_mcp_server_from_container


def create_mcp_http_app() -> FastAPI:
    app = FastAPI(title="HyperDerm MCP HTTP", version="0.1.0")
    server = build_mcp_server_from_container()

    @app.post("/mcp")
    def mcp_endpoint(payload: dict):
        return server.handle(payload)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
