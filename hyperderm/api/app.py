from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hyperderm.api.dependencies import create_container
from hyperderm.api.routes.chatbot import router as chatbot_router
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(diagnosis_router)
    app.include_router(chatbot_router)
    return app
