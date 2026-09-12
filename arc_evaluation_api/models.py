"""Public API request models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Reasoning = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10_000)
]
Cell = Annotated[int, Field(strict=True, ge=0, le=9)]
OutputGrid = list[list[Cell]]


class SubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outputs: list[OutputGrid] = Field(min_length=1, max_length=3)
    reasoning: Reasoning

    @field_validator("outputs")
    @classmethod
    def validate_outputs(cls, outputs: list[OutputGrid]) -> list[OutputGrid]:
        for grid in outputs:
            if not 1 <= len(grid) <= 30:
                raise ValueError("each grid must contain 1 to 30 rows")
            width: int | None = None
            for row in grid:
                if not 1 <= len(row) <= 30:
                    raise ValueError("each grid row must contain 1 to 30 cells")
                if width is None:
                    width = len(row)
                elif len(row) != width:
                    raise ValueError("grids must be rectangular")
        return outputs
