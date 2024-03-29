"""OCR API using ONNXruntime."""

import rootutils

ROOT = rootutils.autosetup()

from io import BytesIO

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI
from omegaconf import DictConfig
from PIL import Image

from src.engine.ocr_onnx_engine import OcrOnnxEngine
from src.schema.ocr_api_schema import OcrSnapshotRequestSchema
from src.schema.ocr_schema import OcrResultSchema
from src.utils.logger import get_logger

log = get_logger()


class OcrOnnxApi:
    """OCR API using ONNXruntime."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize OCR API."""
        self.cfg = cfg
        self.app = FastAPI()
        self.router = APIRouter()

        self.setup_engine()
        self.setup()

    def setup_engine(self) -> None:
        """Setup OCR engine."""
        self.ocr_engine = OcrOnnxEngine(
            det_engine_path=self.cfg.engine.ocr.det_engine_path,
            rec_engine_path=self.cfg.engine.ocr.rec_engine_path,
            ori_engine_path=self.cfg.engine.ocr.ori_engine_path,
            provider=self.cfg.engine.ocr.provider,
            max_batch_size=self.cfg.engine.ocr.max_batch_size,
        )
        self.ocr_engine.setup()

    def setup(self) -> None:
        """Setup router."""

        @self.router.post(
            "/api/v1/engine/ocr/snapshot",
            tags=["ocr"],
            summary="OCR from snapshot",
            response_model=OcrResultSchema,
        )
        async def ocr_snapshot(
            form: OcrSnapshotRequestSchema = Depends(),
        ) -> OcrResultSchema:
            """OCR from snapshot."""
            log.log(21, f"Request OCR from snapshot")
            img = await self.preprocess_img_bytes(await form.image.read())
            result = self.ocr_engine.predict(img)

            return result

    async def preprocess_img_bytes(self, img_bytes: bytes) -> np.ndarray:
        """Preprocess image bytes."""
        img = Image.open(BytesIO(img_bytes))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # if PNG, convert to RGB
        if img.shape[-1] == 4:
            img = img[..., :3]

        return img
