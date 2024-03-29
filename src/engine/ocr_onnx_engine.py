"""End2End OCR engine with ONNX runtime."""

import rootutils

ROOT = rootutils.autosetup()

from typing import Tuple

import cv2
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

    def predict(self, img: np.ndarray) -> OcrResultSchema:
        """Predict OCR from image."""
        # detect text boxes
        boxes = self.det_engine.predict(img)

        # recognize text
        result = self.rec_engine.predict(img, boxes)

        return result

if __name__ == "__main__":
    """Debugging."""
    engine = OcrOnnxEngine(
        det_engine_path="tmp/models/ocr_det.onnx",
        rec_engine_path="tmp/models/ocr_rec.onnx",
        ori_engine_path="tmp/models/ocr_ori.onnx",
        provider="cpu",
        max_batch_size=1,
    )
    engine.setup()

    img = cv2.imread("tmp/sample001.jpg")
    results = engine.predict(img)

    # draw boxes
    for text, box, ori in zip(results.texts, results.boxes, results.oris):
        cv2.polylines(
            img,
            [np.array(box).reshape(-1, 1, 2).astype(np.int32)],
            True,
            (0, 255, 0),
            2,
        )
        color = (0, 255, 0) if ori == "up" else (255, 0, 0)
        cv2.putText(
            img, text, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    cv2.imwrite("tmp/ocr_result.jpg", img)