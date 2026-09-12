"""Validated, fixed-root access to ARC JSON task files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeAlias

Grid: TypeAlias = list[list[int]]
Task: TypeAlias = dict[str, Any]

TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")
MAX_GRID_DIMENSION = 30


class UnknownTaskError(LookupError):
    """The requested task is not present in the selected split."""


class DatasetIntegrityError(RuntimeError):
    """A repository dataset file is invalid."""


def validate_grid(grid: object, *, location: str) -> Grid:
    if not isinstance(grid, list) or not 1 <= len(grid) <= MAX_GRID_DIMENSION:
        raise DatasetIntegrityError(f"invalid grid height at {location}")
    width: int | None = None
    validated: Grid = []
    for row in grid:
        if not isinstance(row, list) or not 1 <= len(row) <= MAX_GRID_DIMENSION:
            raise DatasetIntegrityError(f"invalid grid width at {location}")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise DatasetIntegrityError(f"non-rectangular grid at {location}")
        clean_row: list[int] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                raise DatasetIntegrityError(f"invalid grid cell at {location}")
            clean_row.append(cell)
        validated.append(clean_row)
    return validated


class DatasetStore:
    """Loads only named JSON files below two predetermined dataset directories."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._cache: dict[tuple[str, str], Task] = {}

    def task_ids(self, split: str) -> list[str]:
        directory = self._split_directory(split)
        if not directory.is_dir():
            raise DatasetIntegrityError(f"dataset split is unavailable: {split}")
        return sorted(path.stem for path in directory.glob("[0-9a-f]" * 8 + ".json"))

    def load(self, split: str, task_id: str) -> Task:
        if TASK_ID_PATTERN.fullmatch(task_id) is None:
            raise UnknownTaskError(task_id)
        cache_key = (split, task_id)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        directory = self._split_directory(split)
        path = (directory / f"{task_id}.json").resolve()
        if path.parent != directory or not path.is_file():
            raise UnknownTaskError(task_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetIntegrityError("unable to load dataset task") from exc
        task = self._validate_task(raw, split=split)
        self._cache[cache_key] = task
        return task

    def ready(self) -> bool:
        try:
            for split in ("training", "evaluation"):
                task_ids = self.task_ids(split)
                if not task_ids:
                    return False
                self.load(split, task_ids[0])
            return True
        except (DatasetIntegrityError, UnknownTaskError):
            return False

    def _split_directory(self, split: str) -> Path:
        if split not in {"training", "evaluation"}:
            raise ValueError("invalid dataset split")
        directory = (self.root / split).resolve()
        if directory.parent != self.root:
            raise DatasetIntegrityError("dataset split escaped configured root")
        return directory

    @staticmethod
    def _validate_task(raw: object, *, split: str) -> Task:
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("train"), list)
            or not isinstance(raw.get("test"), list)
        ):
            raise DatasetIntegrityError("invalid task document")
        result: Task = {"train": [], "test": []}
        for section in ("train", "test"):
            pairs = raw[section]
            if not pairs:
                raise DatasetIntegrityError(f"empty {section} section")
            for index, pair in enumerate(pairs):
                if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
                    raise DatasetIntegrityError(f"invalid {section} pair")
                clean_pair = {
                    "input": validate_grid(
                        pair["input"], location=f"{split}.{section}.{index}.input"
                    ),
                    "output": validate_grid(
                        pair["output"], location=f"{split}.{section}.{index}.output"
                    ),
                }
                result[section].append(clean_pair)
        return result
