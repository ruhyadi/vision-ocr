"""OCR detection ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

import time
from typing import List, Tuple

import cv2
import numpy as np

from src.engine.onnx_engine import OnnxEngine
from src.schema.ocr_schema import OcrResultSchema
from src.utils.logger import get_logger
from src.utils.ocr_utils import get_mini_boxes, unclip

log = get_logger()


class OcrDetOnnxEngine(OnnxEngine):
    """PaddleOCR detection engine with ONNX runtime."""

    def __init__(self, engine_path: str, provider: str = "cpu") -> None:
        """Initialize PaddleOCR detection engine with ONNX runtime."""
        super().__init__(engine_path, provider)

    def predict(self, img: np.ndarray) -> List[List[int]]:
        """Detect text from image."""
        t0 = time.time()
        img0 = img.copy()
        img, pads = self.preprocess_img(img)
        results: List[np.ndarray] = self.engine.run(
            [self.metadata[0].output_name], {self.metadata[0].input_name: img}
        )
        boxes = self.postprocess_det(
            results, img0_h=img0.shape[1], img0_w=img0.shape[0], pads=pads
        )

        t1 = time.time()
        log.info(f"Detection time: {(t1 - t0)*1000:.3f}ms")

        return [[int(x) for x in box.flatten()] for box in boxes]

    def preprocess_img(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
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

        return img.astype(np.float32), (top, bottom, left, right)

    def postprocess_det(
        self,
        results: List[np.ndarray],
        img0_h: int,
        img0_w: int,
        pads: Tuple[int, int, int, int],
    ) -> List[np.ndarray]:
        """Postprocess detection results."""
        # convert batch to single result
        result = results[0].squeeze(0).transpose((1, 2, 0)) * 255.0  # CHW -> HWC
        res_h, res_w = result.shape[:2]

        # unpad image
        # pads in the order of top, bottom, left, right
        top, bottom, left, right = pads
        result = result[top : res_h - bottom, left : res_w - right]

        # resize image to original size
        result = cv2.resize(result, (img0_h, img0_w))

        # find contours
        outputs = cv2.findContours(
            image=result.astype(np.uint8),
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(outputs) == 3:
            _, contours, _ = outputs
        else:
            contours, _ = outputs
        n_contours = len(contours)

        boxes = []
        for i in range(n_contours):
            pts, sside = get_mini_boxes(contour=contours[i])
            if sside < 3:
                continue
            box = unclip(pts).reshape(-1, 1, 2)
            box, sside = get_mini_boxes(contour=box)
            if sside < 3 + 2:
                continue

            boxes.append(box)

        return boxes

    def _visualize_contours(self, results: List[np.ndarray]) -> None:
        """Visualize detection results."""
        result = results[0].squeeze(0).transpose((1, 2, 0)) * 255.0  # CHW -> HWC

        cv2.imwrite("tmp/det_contours.jpg", result)

        return result

    def _visualize_boxes(self, img: np.ndarray, boxes: List[np.ndarray]) -> None:
        """Visualize boxes."""
        img1 = img.copy()
        for box in boxes:
            cv2.polylines(img1, [box.astype(np.int32)], True, (0, 255, 0), 2)

        cv2.imwrite("tmp/det_boxes.jpg", img)

        return img1


if __name__ == "__main__":
    """Debugging."""

    engine = OcrDetOnnxEngine(
        engine_path="tmp/models/ocr_det.onnx",
        provider="cpu",
    )
    engine.setup()

    img = cv2.imread("tmp/sample001.jpg")
    results = engine.predict(img)

    log.warning(f"Results: {results}")