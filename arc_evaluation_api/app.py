"""FastAPI application factory and leak-resistant public endpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings
from .database import AttemptResult, SubmissionDatabase, TerminalTaskError
from .dataset import DatasetIntegrityError, DatasetStore, UnknownTaskError
from .models import SubmissionRequest

SESSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")


class RequestTooLargeError(HTTPException):
    """Raised while incrementally reading a request body beyond its configured limit."""

    def __init__(self) -> None:
        super().__init__(status_code=413, detail="request too large")


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_request_bytes: int):
        if max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_lengths = [
            value for name, value in scope["headers"] if name.lower() == b"content-length"
        ]
        transfer_encodings = [
            value for name, value in scope["headers"] if name.lower() == b"transfer-encoding"
        ]
        if (
            len(content_lengths) > 1
            or (content_lengths and (not content_lengths[0] or not content_lengths[0].isdigit()))
            or (content_lengths and transfer_encodings)
        ):
            await JSONResponse(status_code=400, content={"detail": "invalid content length"})(
                scope, receive, send
            )
            return
        if content_lengths:
            try:
                declared_length = int(content_lengths[0])
            except ValueError:
                await JSONResponse(status_code=400, content={"detail": "invalid content length"})(
                    scope, receive, send
                )
                return
            if declared_length > self.max_request_bytes:
                await JSONResponse(status_code=413, content={"detail": "request too large"})(
                    scope, receive, send
                )
                return

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_request_bytes:
                    raise RequestTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestTooLargeError:
            await JSONResponse(status_code=413, content={"detail": "request too large"})(
                scope, receive, send
            )


class RequestAuditMiddleware:
    """Record one safe audit row for every completed HTTP request."""

    def __init__(self, app: ASGIApp, database: SubmissionDatabase):
        self.app = app
        self.database = database

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        status_code = 500

        async def capture_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            route_object = scope.get("route")
            route = getattr(route_object, "path", None)
            method = str(scope.get("method", "UNKNOWN")).upper()
            # Persistence accepts only the method, server-defined route template (for
            # accepted requests), status, and timestamp.  Never pass the request object.
            self.database.audit_request(method=method, route=route, status_code=status_code)


def _result_payload(result: AttemptResult) -> dict[str, Any]:
    return {
        "correct": result.correct,
        "attempts_used": result.attempts_used,
        "attempts_remaining": result.attempts_remaining,
        "status": result.status,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    dataset = DatasetStore(resolved_settings.dataset_root)
    database = SubmissionDatabase(resolved_settings.database_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        database.initialize()
        yield

    app = FastAPI(
        title="ARC Evaluation API",
        version="0.1.0",
        description=(
            "Internal ARC-AGI service. Evaluation responses deliberately omit hidden test "
            "outputs. Submit all outputs for a task together; three attempts are allowed per "
            "task and X-Session-ID. X-Session-ID is a caller-supplied logical identity, not "
            "authentication. Deploy behind a trusted reverse proxy/network boundary and set "
            "ARC_API_KEY for shared bearer authentication."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestSizeLimitMiddleware, max_request_bytes=resolved_settings.max_request_bytes
    )
    # Added last so it wraps request-size checks as well as FastAPI routing/validation.
    app.add_middleware(RequestAuditMiddleware, database=database)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        # Locations and messages can contain caller-controlled field names or values.
        return JSONResponse(status_code=422, content={"detail": "invalid request"})

    @app.exception_handler(DatasetIntegrityError)
    async def dataset_error(_: Request, __: DatasetIntegrityError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "dataset unavailable"})

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, __: Exception) -> JSONResponse:
        # Keep internal exception details out of all client-visible error serialization.
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = resolved_settings.api_key
        if expected is None:
            return
        supplied = ""
        if authorization is not None and authorization.startswith("Bearer "):
            supplied = authorization[7:]
        # Byte comparison also makes non-ASCII header bytes a clean authentication failure.
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="invalid credentials")

    def identify_session(x_session_id: str = Header()) -> str:
        if SESSION_PATTERN.fullmatch(x_session_id) is None:
            raise HTTPException(status_code=400, detail="invalid X-Session-ID")
        return x_session_id

    def evaluation_task(task_id: str) -> dict[str, Any]:
        try:
            return dataset.load("evaluation", task_id)
        except UnknownTaskError as exc:
            raise HTTPException(status_code=404, detail="evaluation task not found") from exc

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", dependencies=[Depends(authenticate)])
    def ready() -> dict[str, str]:
        if not dataset.ready() or not database.ready():
            raise HTTPException(status_code=503, detail="service not ready")
        return {"status": "ready"}

    @app.get("/v1/training/tasks/{task_id}", dependencies=[Depends(authenticate)])
    def get_training_task(task_id: str) -> dict[str, Any]:
        try:
            task = dataset.load("training", task_id)
        except UnknownTaskError as exc:
            raise HTTPException(status_code=404, detail="training task not found") from exc
        return {"task_id": task_id, "train": task["train"], "test": task["test"]}

    @app.get("/v1/training/tasks", dependencies=[Depends(authenticate)])
    def list_training_tasks() -> dict[str, Any]:
        return {"task_ids": dataset.task_ids("training")}

    @app.get("/v1/evaluation/tasks/{task_id}", dependencies=[Depends(authenticate)])
    def get_evaluation_task(task_id: str) -> dict[str, Any]:
        task = evaluation_task(task_id)
        # Explicit allowlist: evaluation test outputs never enter the response object.
        return {
            "task_id": task_id,
            "train": task["train"],
            "test": [
                {"index": index, "input": pair["input"]} for index, pair in enumerate(task["test"])
            ],
        }

    @app.get("/v1/evaluation/tasks", dependencies=[Depends(authenticate)])
    def list_evaluation_tasks() -> dict[str, Any]:
        return {"task_ids": dataset.task_ids("evaluation")}

    @app.get("/v1/evaluation/tasks/{task_id}/status", dependencies=[Depends(authenticate)])
    def get_task_status(
        task_id: str, session_id: str = Depends(identify_session)
    ) -> dict[str, Any]:
        evaluation_task(task_id)
        return {"task_id": task_id, **_result_payload(database.status(session_id, task_id))}

    @app.post("/v1/evaluation/tasks/{task_id}/submissions", dependencies=[Depends(authenticate)])
    def submit(
        task_id: str,
        submission: SubmissionRequest,
        session_id: str = Depends(identify_session),
    ) -> dict[str, Any]:
        task = evaluation_task(task_id)
        expected_outputs = [pair["output"] for pair in task["test"]]
        if len(submission.outputs) != len(expected_outputs):
            raise HTTPException(
                status_code=422, detail="output count does not match test case count"
            )
        proposed_digest = hashlib.sha256(
            json.dumps(submission.outputs, separators=(",", ":")).encode()
        ).digest()
        expected_digest = hashlib.sha256(
            json.dumps(expected_outputs, separators=(",", ":")).encode()
        ).digest()
        correct = hmac.compare_digest(proposed_digest, expected_digest)
        try:
            result = database.submit(
                session_id=session_id,
                task_id=task_id,
                outputs=submission.outputs,
                reasoning=submission.reasoning,
                correct=correct,
            )
        except TerminalTaskError as exc:
            raise HTTPException(status_code=409, detail=_result_payload(exc.result)) from exc
        return {"task_id": task_id, **_result_payload(result)}

    return app


app = create_app()
