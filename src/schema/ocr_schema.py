"""OCR schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

from pydantic import BaseModel, Field


class OcrResultSchema(BaseModel):
    """OCR result schema."""

    boxes: List[List[int]] = Field(
        ...,
        example=[
            [0, 0, 10, 10, 20, 20, 30, 30, 40, 40],
            [50, 50, 60, 60, 70, 70, 80, 80, 90, 90],
        ],
    )
    texts: List[str] = Field(..., example=["Hello", "World"])
    oris: List[str] = Field([], example=["up", "down"])
    scores: List[float] = Field(..., example=[0.99, 0.98])
