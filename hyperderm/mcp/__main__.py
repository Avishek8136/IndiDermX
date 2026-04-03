from __future__ import annotations

import argparse

import uvicorn

from hyperderm.mcp.http_app import create_mcp_http_app
from hyperderm.mcp.server import run_stdio_server


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperDerm MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()

    if args.transport == "stdio":
        run_stdio_server()
        return

    uvicorn.run(
        "hyperderm.mcp.http_app:create_mcp_http_app",
        host=args.host,
        port=args.port,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
