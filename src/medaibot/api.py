"""HTTP API. Failures return real status codes, never an apology dressed up as a 200 answer."""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from medaibot import __version__
from medaibot.config import Settings, get_settings
from medaibot.logging_setup import configure_logging, log_event
from medaibot.models import AgentAnswer
from medaibot.runtime import Runtime, load_runtime

logger = logging.getLogger("medaibot.api")
STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_json)
        app.state.runtime = runtime
        app.state.startup_error = None
        if app.state.runtime is None:
            try:
                app.state.runtime = load_runtime(settings)
            except Exception as exc:  # keep serving /health so the orchestrator can see why
                app.state.startup_error = f"{type(exc).__name__}: {exc}"
                logger.exception("startup failed")
        yield

    app = FastAPI(title="MedAIBot", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_log(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log_event(
            logger,
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response

    def _runtime(request: Request) -> Runtime:
        rt: Runtime | None = request.app.state.runtime
        if rt is None:
            raise HTTPException(status_code=503, detail=request.app.state.startup_error or "not ready")
        return rt

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness: the process is up. Says nothing about the index or the model."""
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready(request: Request) -> dict[str, object]:
        rt = _runtime(request)
        return {
            "status": "ready",
            "chunks": len(rt.index.chunks),
            "embedder": rt.index.meta["embedder"],
            "index_built_at": rt.index.meta.get("built_at"),
            "label_versions": rt.index.meta.get("label_versions", {}),
            "llm_provider": settings.llm_provider,
            "live_openfda": settings.enable_live_openfda,
        }

    @app.post("/ask", response_model=AgentAnswer)
    def ask(body: AskRequest, request: Request) -> AgentAnswer:
        rt = _runtime(request)
        try:
            answer = rt.agent.run(body.question)
        except ValueError as exc:  # configuration problems, e.g. no model name
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if any(s.kind == "error" and s.summary.startswith("model call failed") for s in answer.trace):
            raise HTTPException(status_code=503, detail=answer.abstain_reason)
        return answer

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
