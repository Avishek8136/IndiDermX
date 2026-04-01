from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from hyperderm.core.config import settings
from hyperderm.domain.schemas import DiagnoseRequest, DiagnoseResponse
from hyperderm.api.dependencies import AppContainer

router = APIRouter(tags=["diagnosis"])


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


@router.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(payload: DiagnoseRequest, container: AppContainer = Depends(get_container)) -> DiagnoseResponse:
    state = container.workflow.invoke(
        {
            "descriptors": payload.descriptors,
            "body_part": payload.body_part,
            "symptoms": payload.symptoms,
            "effects": payload.effects,
            "image_path": payload.image_path,
        }
    )

    final = state.get("final", {})

    container.backup.append(
        container.backup.inference_audit,
        {
            "runId": str(uuid.uuid4()),
            "model": settings.bytez_model,
            "input": payload.model_dump(),
            "output": final,
        },
    )

    return DiagnoseResponse(**final)
