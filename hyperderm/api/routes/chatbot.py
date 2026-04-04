from __future__ import annotations

import logging
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from hyperderm.api.dependencies import AppContainer
from hyperderm.core.config import settings
from hyperderm.domain.schemas import ChatbotRequest, ChatbotResponse

router = APIRouter(tags=["chatbot"])
logger = logging.getLogger(__name__)

# Free-tier MedGemma allows one in-flight request. Enforce a strict app-level queue.
_chat_request_lock = threading.Lock()
_queue_counter_lock = threading.Lock()
_queued_or_running_requests = 0


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


@router.post("/upload-image")
def upload_image(file: UploadFile = File(...)) -> dict[str, str]:
    uploads_dir = Path(settings.backup_dir).parent / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "uploaded_image").name
    if not original_name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    extension = Path(original_name).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{extension}"
    destination = uploads_dir / safe_name

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "image_path": str(destination),
        "filename": original_name,
    }


@router.post("/chat/mcp", response_model=ChatbotResponse)
def chat_via_mcp_tool(
    payload: ChatbotRequest,
    container: AppContainer = Depends(get_container),
) -> ChatbotResponse:
    global _queued_or_running_requests

    session_id = payload.session_id or str(uuid.uuid4())

    with _queue_counter_lock:
        _queued_or_running_requests += 1
        queued_ahead = _queued_or_running_requests - 1

    if queued_ahead > 0:
        logger.info("Session %s queued behind %s request(s)", session_id, queued_ahead)

    try:
        with _chat_request_lock:
            result = container.chat_mcp_tool.invoke(
                {
                    "user_message": payload.message,
                    "session_id": session_id,
                    "image_path": payload.image_path,
                }
            )
    finally:
        with _queue_counter_lock:
            _queued_or_running_requests = max(0, _queued_or_running_requests - 1)

    final = result.get("result", {})
    return ChatbotResponse(
        session_id=session_id,
        answer=final.get("answer", ""),
        top_candidate=final.get("top_candidate"),
        candidate_list=final.get("candidate_list", []),
        supporting_evidence=final.get("supporting_evidence", []),
        graph_context=final.get("graph_context", []),
        suggested_questions=final.get("suggested_questions", []),
        memory_summary=final.get("memory_summary", ""),
        explainability=final.get("explainability", {}),
        used_fallback=bool(final.get("used_fallback", False)),
        tool_trace=final.get("tool_trace", []),
        model=settings.bytez_model,
    )
