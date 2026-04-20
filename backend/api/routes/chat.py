from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.api.schemas import ChatRequest
from backend.db.models import Notebook
from backend.db.session import get_db
from backend.rag.chat import stream_chat

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    nb = db.get(Notebook, req.notebook_id)
    if nb is None:
        raise HTTPException(404, "notebook not found")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    async def gen():
        async for event in stream_chat(
            db, req.notebook_id, messages, model=req.model or nb.openrouter_model
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
