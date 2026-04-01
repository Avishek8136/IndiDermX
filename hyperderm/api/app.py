from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from hyperderm.api.dependencies import create_container
from hyperderm.api.routes.diagnosis import router as diagnosis_router
from hyperderm.api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = create_container()
    app.state.container = container
    try:
        yield
    finally:
        container.store.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="HYPERDERM-Graph API",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(diagnosis_router)
    return app
