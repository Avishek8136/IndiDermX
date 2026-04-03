from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


def create_frontend_app() -> FastAPI:
    app = FastAPI(title="HYPERDERM Chat Frontend", version="0.1.0")
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/config")
    def config() -> JSONResponse:
        return JSONResponse({"apiBaseUrl": os.getenv("HYPERDERM_API_URL", "http://127.0.0.1:8000")})

    return app


def main() -> None:
    uvicorn.run("frontend_service.app:create_frontend_app", host="0.0.0.0", port=8080, factory=True, reload=False)


if __name__ == "__main__":
    main()
