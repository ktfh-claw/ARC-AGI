# ARC Evaluation API

This is a small internal service for serving ARC tasks and scoring evaluation attempts without
returning hidden evaluation test outputs.

## Trust boundary and identity

Every evaluation status/submission request requires an `X-Session-ID` (16–128 characters from
letters, digits, `.`, `_`, `:`, or `-`). It is a caller-selected logical identity used to scope the
three-attempt budget; **it is not authentication**. In the initial internal deployment, keep the
service on a trusted network or behind an authenticating reverse proxy and set `ARC_API_KEY` to a
random value of at least 16 characters. With that setting, protected routes also require
`Authorization: Bearer <key>`. A shared key does not stop one authorized caller from impersonating
another session; use proxy-issued identity or per-client credentials before exposing this service
to mutually untrusted clients.

No static file route is installed. Dataset roots and split names are fixed server-side, and task IDs
must be exactly eight lowercase hexadecimal characters.
The application does not log request bodies, reasoning, guesses, or expected outputs. Uvicorn's
normal access log contains request paths and response status only; do not add request-body logging
at a reverse proxy or run production workers with debugger-style traceback-local-variable capture.

## Endpoints

- `GET /health`: process liveness (intentionally unauthenticated).
- `GET /ready`: dataset/database readiness.
- `GET /v1/training/tasks` and `GET /v1/evaluation/tasks`: available task IDs.
- `GET /v1/training/tasks/{task_id}`: complete training task, including demonstration and test
  inputs/outputs.
- `GET /v1/evaluation/tasks/{task_id}`: demonstrations plus indexed test inputs. Evaluation test
  outputs are omitted.
- `GET /v1/evaluation/tasks/{task_id}/status`: session-scoped attempt status.
- `POST /v1/evaluation/tasks/{task_id}/submissions`: score all test outputs as one task attempt.
- `GET /docs` and `GET /openapi.json`: interactive and machine-readable documentation.

A submission body is:

```json
{
  "outputs": [[[0, 1], [1, 0]]],
  "reasoning": "The demonstrated transformation is applied to the test input."
}
```

`outputs` must contain one rectangular 1×1 through 30×30 integer grid per test case; cells are
0–9. `reasoning` is required, non-blank, and at most 10,000 characters. Responses reveal only
`correct`, `attempts_used`, `attempts_remaining`, and `status` (`active`, `solved`, or `exhausted`).
Solved/exhausted tasks return HTTP 409 without consuming an attempt. Validation errors never echo
submitted values.

## Ubuntu deployment

These commands assume a checkout at `/opt/arc-evaluation-api` containing the unmodified `data/`
directory:

```bash
sudo useradd --system --home /opt/arc-evaluation-api --shell /usr/sbin/nologin arc-api
sudo install -d -o arc-api -g arc-api -m 0750 /var/lib/arc-evaluation-api
cd /opt/arc-evaluation-api
sudo python3 -m venv .venv
sudo .venv/bin/pip install --requirement requirements.lock
sudo install -o root -g arc-api -m 0640 deployment/arc-evaluation-api.env.example /etc/arc-evaluation-api.env
sudoedit /etc/arc-evaluation-api.env
sudo install -o root -g root -m 0644 deployment/arc-evaluation-api.service /etc/systemd/system/arc-evaluation-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now arc-evaluation-api
curl --fail http://127.0.0.1:8000/health
curl --fail -H 'Authorization: Bearer YOUR_KEY' http://127.0.0.1:8000/ready
```

The default bind is `0.0.0.0:8000`. Apply an Ubuntu firewall rule and/or reverse proxy appropriate
to the trusted internal network; do not publish this port directly to untrusted networks. SQLite
state lives in `/var/lib/arc-evaluation-api` and should be included in host backups.

`requirements.txt` pins the direct runtime dependencies and `requirements.lock` records the fully
resolved Python 3.12 Ubuntu/Linux runtime. Regenerate the lock in a clean environment whenever a
direct dependency changes. Development-only tools are pinned in `requirements-dev.txt`.
