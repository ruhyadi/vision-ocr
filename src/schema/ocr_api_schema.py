"""OCR API schema."""

import rootutils

ROOT = rootutils.autosetup()

from fastapi import File, UploadFile
from pydantic import BaseModel, Field


class OcrSnapshotRequestSchema(BaseModel):
    """OCR snapshot request schema."""

    image: UploadFile = Field(..., description="Image file")
