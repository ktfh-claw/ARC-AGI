from __future__ import annotations

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from arc_evaluation_api.app import RequestSizeLimitMiddleware, create_app
from arc_evaluation_api.config import Settings

EVALUATION_ID = "abc12345"
TRAINING_ID = "def67890"
SESSION = "test-session-0001"
SYNTHETIC_OUTPUTS = [[[9, 8], [7, 6]]]


def generated_hidden_grid(seed: int = 37) -> list[list[int]]:
    """Generate synthetic test data; no corpus evaluation solution is a test fixture."""
    return [[(seed + row * 3 + column * 7) % 10 for column in range(3)] for row in range(2)]


def write_task(path: Path, *, hidden: list[list[int]]) -> None:
    task = {
        "train": [
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
            {"input": [[2]], "output": [[3]]},
        ],
        "test": [{"input": [[4, 5, 4]], "output": hidden}],
    }
    path.write_text(json.dumps(task), encoding="utf-8")


@pytest.fixture
def service(tmp_path: Path) -> tuple[TestClient, list[list[int]], Path]:
    data = tmp_path / "data"
    (data / "training").mkdir(parents=True)
    (data / "evaluation").mkdir()
    hidden = generated_hidden_grid()
    write_task(data / "evaluation" / f"{EVALUATION_ID}.json", hidden=hidden)
    write_task(data / "training" / f"{TRAINING_ID}.json", hidden=[[6]])
    database = tmp_path / "state" / "submissions.sqlite3"
    settings = Settings(dataset_root=data, database_path=database, api_key=None)
    with TestClient(create_app(settings)) as client:
        yield client, hidden, database


def headers(session: str = SESSION) -> dict[str, str]:
    return {"X-Session-ID": session}


def wrong_output(value: int = 0) -> dict[str, Any]:
    return {"outputs": [[[value]]], "reasoning": "A non-empty explanation."}


def assert_secret_absent(response_text: str, hidden: list[list[int]]) -> None:
    compact = json.dumps(hidden, separators=(",", ":"))
    spaced = json.dumps(hidden)
    assert compact not in response_text
    assert spaced not in response_text


def test_training_and_evaluation_views_are_intentionally_different(
    service: tuple[TestClient, list[list[int]], Path],
) -> None:
    client, hidden, _ = service
    assert client.get("/v1/training/tasks").json() == {"task_ids": [TRAINING_ID]}
    assert client.get("/v1/evaluation/tasks").json() == {"task_ids": [EVALUATION_ID]}
    training = client.get(f"/v1/training/tasks/{TRAINING_ID}")
    assert training.status_code == 200
    assert training.json()["train"][0]["output"] == [[1, 0], [0, 1]]
    assert training.json()["test"][0]["output"] == [[6]]

    evaluation = client.get(f"/v1/evaluation/tasks/{EVALUATION_ID}")
    assert evaluation.status_code == 200
    assert evaluation.json()["test"] == [{"index": 0, "input": [[4, 5, 4]]}]
    assert "output" not in evaluation.json()["test"][0]
    assert_secret_absent(evaluation.text, hidden)


def test_exact_three_attempts_and_persisted_audit_fields(
    service: tuple[TestClient, list[list[int]], Path],
) -> None:
    client, _, database = service
    for attempt in range(1, 4):
        response = client.post(
            f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
            headers=headers(),
            json=wrong_output(attempt),
        )
        assert response.status_code == 200
        assert response.json() == {
            "task_id": EVALUATION_ID,
            "correct": False,
            "attempts_used": attempt,
            "attempts_remaining": 3 - attempt,
            "status": "exhausted" if attempt == 3 else "active",
        }

    fourth = client.post(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
        headers=headers(),
        json=wrong_output(9),
    )
    assert fourth.status_code == 409
    assert fourth.json()["detail"]["attempts_used"] == 3

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT attempt_count, reasoning, correct, submitted_at, proposed_output_json "
            "FROM submissions ORDER BY id"
        ).fetchall()
    assert [row[0] for row in rows] == [1, 2, 3]
    assert all(row[1] == "A non-empty explanation." for row in rows)
    assert all(row[2] == 0 and "T" in row[3] for row in rows)
    assert [json.loads(row[4]) for row in rows] == [
        wrong_output(value)["outputs"] for value in (1, 2, 3)
    ]


def test_success_is_terminal_and_does_not_consume_an_extra_attempt(
    service: tuple[TestClient, list[list[int]], Path],
) -> None:
    client, hidden, _ = service
    solved = client.post(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
        headers=headers(),
        json={"outputs": [hidden], "reasoning": "Derived the transformation."},
    )
    assert solved.status_code == 200
    assert solved.json()["correct"] is True
    assert solved.json()["status"] == "solved"
    assert solved.json()["attempts_used"] == 1
    assert_secret_absent(solved.text, hidden)

    retry = client.post(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
        headers=headers(),
        json=wrong_output(),
    )
    assert retry.status_code == 409
    assert retry.json()["detail"]["attempts_used"] == 1


def test_state_survives_application_restart(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "training").mkdir(parents=True)
    (data / "evaluation").mkdir()
    hidden = generated_hidden_grid()
    write_task(data / "evaluation" / f"{EVALUATION_ID}.json", hidden=hidden)
    database = tmp_path / "submissions.sqlite3"
    settings = Settings(dataset_root=data, database_path=database, api_key=None)

    with TestClient(create_app(settings)) as first:
        response = first.post(
            f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
            headers=headers(),
            json=wrong_output(),
        )
        assert response.json()["attempts_used"] == 1
    with TestClient(create_app(settings)) as restarted:
        status = restarted.get(f"/v1/evaluation/tasks/{EVALUATION_ID}/status", headers=headers())
        assert status.json()["attempts_used"] == 1
        second = restarted.post(
            f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
            headers=headers(),
            json=wrong_output(8),
        )
        assert second.json()["attempts_used"] == 2


def test_concurrent_requests_atomically_cap_attempts(
    service: tuple[TestClient, list[list[int]], Path],
) -> None:
    client, _, database = service

    def submit(index: int) -> tuple[int, int]:
        response = client.post(
            f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
            headers=headers("concurrent-user-01"),
            json=wrong_output(index % 10),
        )
        payload = response.json()
        attempts = payload.get("attempts_used", payload.get("detail", {}).get("attempts_used"))
        return response.status_code, attempts

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(submit, range(10)))
    assert sum(code == 200 for code, _ in results) == 3
    assert sum(code == 409 for code, _ in results) == 7
    assert sorted(attempt for code, attempt in results if code == 200) == [1, 2, 3]
    with sqlite3.connect(database) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM submissions WHERE session_id = ? AND task_id = ?",
            ("concurrent-user-01", EVALUATION_ID),
        ).fetchone()[0]
    assert count == 3


@pytest.mark.parametrize(
    "body",
    [
        {"outputs": [], "reasoning": "reason"},
        {"outputs": [[[0], [0, 1]]], "reasoning": "reason"},
        {"outputs": [[[10]]], "reasoning": "reason"},
        {"outputs": [[[True]]], "reasoning": "reason"},
        {"outputs": SYNTHETIC_OUTPUTS, "reasoning": "   "},
        {"outputs": [[[0] * 31]], "reasoning": "reason"},
        {"outputs": SYNTHETIC_OUTPUTS, "reasoning": "reason", "unexpected": "field"},
    ],
)
def test_malformed_submissions_are_rejected_without_consuming_attempts(
    service: tuple[TestClient, list[list[int]], Path], body: dict[str, Any]
) -> None:
    client, _, _ = service
    assert (
        client.post(
            f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions", headers=headers(), json=body
        ).status_code
        == 422
    )
    status = client.get(f"/v1/evaluation/tasks/{EVALUATION_ID}/status", headers=headers())
    assert status.json()["attempts_used"] == 0


def test_output_count_and_oversized_request_are_rejected(
    service: tuple[TestClient, list[list[int]], Path],
) -> None:
    client, _, _ = service
    count_mismatch = client.post(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
        headers=headers(),
        json={"outputs": [[[0]], [[1]]], "reasoning": "reason"},
    )
    assert count_mismatch.status_code == 422
    oversized = client.post(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
        headers=headers(),
        content=json.dumps({"outputs": SYNTHETIC_OUTPUTS, "reasoning": "x" * 70_000}),
    )
    assert oversized.status_code == 413

    def chunks() -> Any:
        yield json.dumps({"outputs": SYNTHETIC_OUTPUTS, "reasoning": "y" * 70_000}).encode()

    oversized_chunked = client.post(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
        headers=headers(),
        content=chunks(),
    )
    assert oversized_chunked.status_code == 413

    understated = client.post(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
        headers={**headers(), "Content-Length": "1"},
        content=json.dumps({"outputs": SYNTHETIC_OUTPUTS, "reasoning": "z" * 70_000}),
    )
    assert understated.status_code == 413

    invalid_length = client.post(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
        headers={**headers(), "Content-Length": "-1"},
        content=b"{}",
    )
    assert invalid_length.status_code == 400


def test_request_size_limit_stops_consuming_a_chunked_body_at_the_limit() -> None:
    received_chunks = 0
    sent_messages: list[dict[str, Any]] = []
    chunks = [b"1234", b"5678", b"should-not-be-read"]

    async def receive() -> dict[str, Any]:
        nonlocal received_chunks
        body = chunks[received_chunks]
        received_chunks += 1
        return {
            "type": "http.request",
            "body": body,
            "more_body": received_chunks < len(chunks),
        }

    async def send(message: dict[str, Any]) -> None:
        sent_messages.append(message)

    async def consume_body(_: dict[str, Any], receive: Any, __: Any) -> None:
        while (await receive()).get("more_body", False):
            pass

    scope = {"type": "http", "headers": []}
    middleware = RequestSizeLimitMiddleware(consume_body, max_request_bytes=7)
    asyncio.run(middleware(scope, receive, send))  # type: ignore[arg-type]

    assert received_chunks == 2
    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 413


@pytest.mark.parametrize(
    "request_headers",
    [
        [(b"content-length", b"1"), (b"content-length", b"1")],
        [(b"content-length", b"1"), (b"transfer-encoding", b"chunked")],
        [(b"content-length", b"9" * 5_000)],
    ],
)
def test_request_size_limit_rejects_ambiguous_framing(
    request_headers: list[tuple[bytes, bytes]],
) -> None:
    downstream_called = False
    sent_messages: list[dict[str, Any]] = []

    async def downstream(_: Any, __: Any, ___: Any) -> None:
        nonlocal downstream_called
        downstream_called = True

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent_messages.append(message)

    middleware = RequestSizeLimitMiddleware(downstream, max_request_bytes=7)
    scope = {"type": "http", "headers": request_headers}
    asyncio.run(middleware(scope, receive, send))  # type: ignore[arg-type]

    assert downstream_called is False
    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 400


@pytest.mark.parametrize(
    "path",
    [
        "/v1/evaluation/tasks/not-an-id",
        "/v1/evaluation/tasks/deadbeef",
        "/v1/evaluation/tasks/..%2Ftraining%2Fdef67890",
        "/v1/evaluation/tasks/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "/data/evaluation/abc12345.json",
    ],
)
def test_unknown_ids_traversal_and_static_file_probes_are_safe(
    service: tuple[TestClient, list[list[int]], Path], path: str
) -> None:
    client, hidden, _ = service
    response = client.get(path)
    assert response.status_code in {404, 422}
    assert_secret_absent(response.text, hidden)


def test_solution_is_absent_from_every_read_or_error_surface(
    service: tuple[TestClient, list[list[int]], Path],
) -> None:
    client, hidden, _ = service
    probes = [
        client.get(f"/v1/evaluation/tasks/{EVALUATION_ID}"),
        client.get(f"/v1/evaluation/tasks/{EVALUATION_ID}/status", headers=headers()),
        client.get("/openapi.json"),
        client.get("/docs"),
        client.get("/ready"),
        client.post(
            f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
            headers=headers(),
            json={"outputs": [hidden], "reasoning": 42},
        ),
        client.post(
            f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
            headers=headers(),
            json={"outputs": [hidden], "reasoning": "ok", "extra": hidden},
        ),
        client.post(
            f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
            headers=headers(),
            json={
                "outputs": SYNTHETIC_OUTPUTS,
                "reasoning": "ok",
                json.dumps(hidden, separators=(",", ":")): "attacker-controlled field",
            },
        ),
    ]
    for response in probes:
        assert_secret_absent(response.text, hidden)


def test_sessions_are_explicit_and_isolated(
    service: tuple[TestClient, list[list[int]], Path],
) -> None:
    client, _, _ = service
    missing = client.get(f"/v1/evaluation/tasks/{EVALUATION_ID}/status")
    invalid = client.get(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/status", headers={"X-Session-ID": "short"}
    )
    assert missing.status_code == 422
    assert invalid.status_code == 400
    client.post(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/submissions",
        headers=headers("isolated-session-1"),
        json=wrong_output(),
    )
    other = client.get(
        f"/v1/evaluation/tasks/{EVALUATION_ID}/status",
        headers=headers("isolated-session-2"),
    )
    assert other.json()["attempts_used"] == 0


def test_optional_bearer_api_key(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "training").mkdir(parents=True)
    (data / "evaluation").mkdir()
    write_task(data / "training" / f"{TRAINING_ID}.json", hidden=[[5]])
    write_task(data / "evaluation" / f"{EVALUATION_ID}.json", hidden=generated_hidden_grid())
    app = create_app(
        Settings(
            dataset_root=data,
            database_path=tmp_path / "db.sqlite3",
            api_key="a-secure-internal-key",
        )
    )
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 401
        assert (
            client.get(
                "/ready", headers={b"Authorization": b"Bearer \xffxxxxxxxxxxxxxxxx"}
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/ready", headers={"Authorization": "Bearer a-secure-internal-key"}
            ).status_code
            == 200
        )
