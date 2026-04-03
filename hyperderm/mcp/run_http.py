from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "hyperderm.mcp.http_app:create_mcp_http_app",
        host="0.0.0.0",
        port=9001,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
