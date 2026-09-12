"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    dataset_root: Path
    database_path: Path
    api_key: str | None
    max_request_bytes: int = 65_536

    @classmethod
    def from_env(cls) -> Settings:
        repository_root = Path(__file__).resolve().parent.parent
        api_key = os.getenv("ARC_API_KEY")
        if api_key is not None and len(api_key) < 16:
            raise ValueError("ARC_API_KEY must contain at least 16 characters")
        return cls(
            dataset_root=Path(os.getenv("ARC_DATASET_ROOT", repository_root / "data")),
            database_path=Path(
                os.getenv("ARC_DATABASE_PATH", repository_root / "var" / "arc-api.sqlite3")
            ),
            api_key=api_key,
            max_request_bytes=int(os.getenv("ARC_MAX_REQUEST_BYTES", "65536")),
        )
