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

    def predict(self, img: np.ndarray) -> OcrResultSchema:
        """Detect text from image."""
        img0 = img.copy()
        img, pad = self.preprocess_img(img)
        results: List[np.ndarray] = self.engine.run(
            [self.metadata[0].output_name], {self.metadata[0].input_name: img}
        )
        self._visualize(results)
        result = self.postprocess_det(
            results, img0_h=img0.shape[1], img0_w=img0.shape[0], pad=pad
        )

        return results

    def preprocess_img(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for detection model"""
        src_h, src_w = img.shape[:2]
        dst_h = 640 * ((src_h // 640) + 1)
        dst_w = 640 * ((src_w // 640) + 1)

        # resize image with ratio preserved
        ratio = min(dst_w / src_w, dst_h / src_h)
        resized_w, resized_h = int(src_w * ratio), int(src_h * ratio)
        dw, dh = (dst_w - resized_w) / 2, (dst_h - resized_h) / 2
        img = cv2.resize(img, (resized_w, resized_h))

        # pad image to target size
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114
        )

        # normalize image
        img = np.transpose(img, (2, 0, 1)) / 255.0  # HWC -> CHW
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32), (left, top)

    def postprocess_det(
        self,
        results: List[np.ndarray],
        img0_h: int,
        img0_w: int,
        pad: Tuple[int, int],
    ) -> OcrResultSchema:
        """Postprocess detection results."""
        # convert batch to single result
        result = results[0].squeeze(0).transpose((1, 2, 0)) * 255.0  # CHW -> HWC
        res_h, res_w = result.shape[:2]

        # unpad image
        result = result[
            int(round(pad[1])) : int(round(pad[1])) + res_h,
            int(round(pad[0])) : int(round(pad[0])) + res_w,
        ]

        # resize image to original size
        result = cv2.resize(result, (img0_h, img0_w))

        return result

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

    img = cv2.imread("tmp/sample001.jpg")
    # results = engine.predict(img, shape=(img.shape[1], img.shape[0]))
    results = engine.predict(img)
