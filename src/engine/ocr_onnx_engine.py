"""Paddle OCR engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import Tuple

import cv2
import numpy as np

from src.engine.onnx_engine import OnnxEngine
from src.schema.ocr_schema import OcrResultSchema
from src.utils.logger import get_logger

log = get_logger()


class OcrOnnxEngine(OnnxEngine):
    """PaddleOCR engine with ONNX runtime."""

    def __init__(
        self,
        det_engine_path: str,
        rec_engine_path: str,
        rec_batch_size: int = 8,
        dict_path: str = "assets/en_dict.txt",
        provider: str = "cpu",
    ) -> None:
        """Initialize PaddleOCR engine with ONNX runtime."""
        self.det_engine_path = det_engine_path
        self.rec_engine_path = rec_engine_path
        self.dict_path = dict_path
        self.rec_batch_size = rec_batch_size
        self.provider = provider

        # recognition postprocess params
        self.rec_postprocess_params = {
            "name": "ocr_utils.CTCLabelDecode",
            "character_type": "ch",
            "character_dict_path": self.dict_path,
            "use_space_char": True,
        }

    def setup(self) -> None:
        """Setup PaddleOCR engine with ONNX runtime."""
        log.info(f"Setup PaddleOCR engine with ONNXRuntime...")

        # setup text detection engine
        self.det_engine = OnnxEngine(
            engine_path=self.det_engine_path, provider=self.provider
        )
        self.det_engine.setup()

        # setup text recognition engine
        self.rec_engine = OnnxEngine(
            engine_path=self.rec_engine_path, provider=self.provider
        )
        self.rec_engine.setup()

        # setup metadata
        self.det_input_name = self.det_engine.metadata[0].input_name
        self.det_output_name = self.det_engine.metadata[0].output_name
        self.rec_input_name = self.rec_engine.metadata[0].input_name
        self.rec_output_name = self.rec_engine.metadata[0].output_name

        log.info(f"Setup PaddleOCR engine with ONNXRuntime completed")

    def predict(self, img: np.ndarray) -> OcrResultSchema:
        """Predict text from image."""
        img0 = img.copy()

    def detect(self, img: np.ndarray):
        """Detect text from image."""
        results = self.det_engine.run()
