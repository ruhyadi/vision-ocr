"""OCR detection ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Tuple

import cv2
import numpy as np

from src.engine.onnx_engine import OnnxEngine
from src.schema.ocr_schema import OcrResultSchema
from src.utils.logger import get_logger

log = get_logger()


class OcrDetOnnxEngine(OnnxEngine):
    """PaddleOCR detection engine with ONNX runtime."""

    def __init__(self, engine_path: str, provider: str = "cpu") -> None:
        """Initialize PaddleOCR detection engine with ONNX runtime."""
        super().__init__(engine_path, provider)

    def predict(
        self, img: np.ndarray, shape: Tuple[int, int] = (640, 640)
    ) -> OcrResultSchema:
        """Detect text from image."""
        img = self.preprocess_img(img, shape)
        results: List[np.ndarray] = self.engine.run(
            [self.metadata[0].output_name], {self.metadata[0].input_name: img}
        )
        self._visualize(results)

        return results

    def preprocess_img(
        self, img: np.ndarray, shape: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """Preprocess image for detection model."""
        img = cv2.resize(img, shape)
        log.warning(f"img shape: {img.shape}")
        img = np.transpose(img, (2, 0, 1)) / 255.0  # HWC -> CHW
        img = np.expand_dims(img, axis=0)

        # normalize image
        img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std

        return img.astype(np.float32)

    def _visualize(self, results: List[np.ndarray]) -> None:
        """Visualize detection results."""
        result = results[0].squeeze(0).transpose((1, 2, 0)) * 255.0  # CHW -> HWC

        cv2.imwrite("tmp/det.jpg", result)


if __name__ == "__main__":
    """Debugging."""

    engine = OcrDetOnnxEngine(
        engine_path="tmp/models/ocr_det.onnx",
        provider="cpu",
    )
    engine.setup()
    log.warning(f"Engine metadata: {engine.metadata}")

    img = cv2.imread("tmp/sample001.jpg")
    log.warning(f"img shape: {img.shape}")
    # results = engine.predict(img, shape=(img.shape[1], img.shape[0]))
    results = engine.predict(img, shape=(640 * 5, 640 * 5))
