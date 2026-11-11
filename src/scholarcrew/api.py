"""FastAPI app: streams agent progress as NDJSON and serves a small web UI."""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .config import Settings
from .graph import stream_crew
from .llm import make_llm

STATIC = Path(__file__).parent / "static"
MAX_QUESTION_CHARS = 300
RATE_LIMIT = 5  # requests per IP per hour, to protect a free-tier API key on a public demo
WINDOW_SECONDS = 3600

app = FastAPI(title="ScholarCrew", version="0.1.0")
_hits: dict[str, deque[float]] = defaultdict(deque)


def _check_rate_limit(client: str) -> None:
    now = time.time()
    hits = _hits[client]
    while hits and now - hits[0] > WINDOW_SECONDS:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit reached. Try again later.")
    hits.append(now)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/research")
def research(q: str, request: Request) -> StreamingResponse:
    question = q.strip()
    if not question:
        raise HTTPException(400, "Question is empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Question must be under {MAX_QUESTION_CHARS} characters.")
    _check_rate_limit(request.client.host if request.client else "unknown")

    settings = Settings.from_env()

    def events() -> Iterator[str]:
        try:
            for node, update in stream_crew(question, make_llm(settings), settings):
                payload = {"agent": node, "log": update.get("log", [])}
                if "report" in update:
                    payload["report"] = update["report"]
                yield json.dumps(payload) + "\n"
        except Exception as exc:  # noqa: BLE001 - report to the client instead of a broken stream
            yield json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
