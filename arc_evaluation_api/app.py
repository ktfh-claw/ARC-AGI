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
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import RequestResponseEndpoint

from .config import Settings
from .database import AttemptResult, SubmissionDatabase, TerminalTaskError
from .dataset import DatasetIntegrityError, DatasetStore, UnknownTaskError
from .models import SubmissionRequest

SESSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")


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

    @app.middleware("http")
    async def request_size_limit(request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > resolved_settings.max_request_bytes:
                    return JSONResponse(status_code=413, content={"detail": "request too large"})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "invalid content length"})
        # Cover chunked requests that do not carry Content-Length as well.
        if len(await request.body()) > resolved_settings.max_request_bytes:
            return JSONResponse(status_code=413, content={"detail": "request too large"})
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Never echo request values: a submitted value may equal a hidden evaluation solution.
        errors = [
            {"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

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
        if not hmac.compare_digest(supplied, expected):
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
