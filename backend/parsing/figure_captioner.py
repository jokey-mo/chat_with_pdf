from __future__ import annotations

import base64
from pathlib import Path

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.openrouter import async_client

log = get_logger(__name__)

PROMPT = (
    "You are assisting a scientific-paper retrieval system. Describe the figure "
    "concisely (4-6 sentences) so a search over captions finds it. Include: what "
    "it depicts, variables/axes if a chart, key quantitative findings visible, "
    "and its apparent role (method diagram, results plot, schematic, photo, etc). "
    "Use the author-provided caption when helpful but do not copy it verbatim."
)


async def caption(
    image_path: Path | str, original_caption: str | None = None
) -> str | None:
    p = Path(image_path)
    if not p.exists():
        return None
    b64 = base64.b64encode(p.read_bytes()).decode()
    content: list[dict] = [
        {"type": "text", "text": PROMPT},
    ]
    if original_caption:
        content.append({"type": "text", "text": f"Author caption: {original_caption}"})
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
    )

    try:
        client = async_client()
        resp = await client.chat.completions.create(
            model=settings.openrouter_vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as e:
        log.warning("vision caption failed for %s: %s", p, e)
        return None
