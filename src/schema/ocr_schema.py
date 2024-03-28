"""OCR schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

from pydantic import BaseModel, Field


class OcrResultSchema(BaseModel):
    """OCR result schema."""

    texts: List[str] = Field(..., example=["Hello", "World"])
    scores: List[float] = Field(..., example=[0.99, 0.98])
    boxes: List[List[int]] = Field(..., example=[[0, 0, 10, 10], [10, 10, 20, 20]])
