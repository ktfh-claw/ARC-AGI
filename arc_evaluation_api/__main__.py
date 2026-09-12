"""Development and simple VM entrypoint."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "arc_evaluation_api.app:app",
        host=os.getenv("ARC_HOST", "0.0.0.0"),
        port=int(os.getenv("ARC_PORT", "8000")),
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
