"""End2End OCR engine with ONNX runtime."""

import rootutils

ROOT = rootutils.autosetup()

import time

import numpy as np

from src.engine.ocr_det_onnx_engine import OcrDetOnnxEngine
from src.engine.ocr_rec_onnx_engine import OcrRecOnnxEngine
from src.schema.ocr_schema import OcrResultSchema
from src.utils.logger import get_logger

log = get_logger()


class OcrOnnxEngine:
    """End2end OCR engine with ONNX runtime."""

    def __init__(
        self,
        det_engine_path: str,
        rec_engine_path: str,
        ori_engine_path: str,
        provider: str = "cpu",
        max_batch_size: int = 8,
    ) -> None:
        """Initialize OCR engine with ONNX runtime."""
        self.det_engine_path = det_engine_path
        self.rec_engine_path = rec_engine_path
        self.ori_engine_path = ori_engine_path
        self.provider = provider
        self.max_batch_size = max_batch_size

    def setup(self) -> None:
        """Setup OCR engines."""
        log.info(f"Setup OCR engines with provider {self.provider}")
        self.det_engine = OcrDetOnnxEngine(
            engine_path=self.det_engine_path, provider=self.provider
        )
        self.det_engine.setup()

        self.rec_engine = OcrRecOnnxEngine(
            rec_engine_path=self.rec_engine_path,
            ori_engine_path=self.ori_engine_path,
            provider=self.provider,
            max_batch_size=self.max_batch_size,
        )
        self.rec_engine.setup()

        log.info("OCR engines are ready")

    def predict(self, img: np.ndarray) -> OcrResultSchema:
        """Predict OCR from image."""
        log.info(f"Predict OCR from image with shape {img.shape[:2]}")
        t0 = time.time()

        # detect text boxes
        boxes = self.det_engine.predict(img)

        # recognize text
        result = self.rec_engine.predict(img, boxes)

        t1 = time.time()
        log.info(f"OCR prediction time: {(t1 - t0)*1000:.3f}ms")

        return result
