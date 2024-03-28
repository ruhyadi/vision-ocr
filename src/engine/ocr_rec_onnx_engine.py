"""OCR recognition ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

import time
from typing import List

import numpy as np
from tqdm import tqdm

from src.engine.onnx_engine import OnnxEngine
from src.utils.logger import get_logger

log = get_logger()


class OcrRecOnnxEngine(OnnxEngine):
    """OCR recognition engine with ONNX runtime."""

    def __init__(
        self, engine_path: str, provider: str = "cpu", max_batch_size: int = 8
    ) -> None:
        """Initialize OCR recognition engine with ONNX runtime."""
        super().__init__(engine_path, provider)
        self.max_batch_size = max_batch_size

    def predict(self, img: np.ndarray, boxes: List[np.ndarray]):
        """
        Predict transcription from text images.

        Args:
            img (np.ndarray): Original non-preprocessed image
            boxes (List[np.ndarray]): Detected text boxes in shape (n, 4, 2)

        Returns:
            List[np.ndarray]: Predicted transcriptions
        """
        t0 = time.time()
        imgs = self.preprocess_imgs(img, boxes)

        # iterate per batch
        results = []
        for i in tqdm(range(0, len(imgs), self.max_batch_size), desc="Recognition"):
            batch_imgs = imgs[i : i + self.max_batch_size]
            batch_results: List[np.ndarray] = self.engine.run(
                [self.metadata[0].output_name],
                {self.metadata[0].input_name: batch_imgs},
            )
            results.extend(batch_results)

        t1 = time.time()
        log.info(f"Recognition time: {(t1 - t0)*1000:.3f}ms")

        return results

    def preprocess_imgs(
        self, img: np.ndarray, boxes: List[np.ndarray], dst_h: int = 48
    ) -> np.ndarray:
        """Preprocess image for recognition model."""
        imgs = []
        for box in boxes:
            # crop with rotated box
            img_crop = self.rotated_crop(img, points=box)
            # resize
            dst_w = int(dst_h * img_crop.shape[1] / img_crop.shape[0])
            img_crop = cv2.resize(img_crop, (dst_w, dst_h))
            imgs.append(img_crop.transpose(2, 0, 1))  # HWC -> CHW

        return imgs

    def rotated_crop(self, img: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Crop image with rotated box (points)."""
        assert len(points) == 4, "Rotated box must have 4 points"
        width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [width, 0],
                [width, height],
                [0, height],
            ]
        )
        M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
        dst_image = cv2.warpPerspective(
            img,
            M,
            (width, height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_LINEAR,
        )

        # rotate image
        dst_img_h, dst_img_w = dst_image.shape[:2]
        if dst_img_h * 1.0 / dst_img_w > 1.5:
            dst_image = np.rot90(dst_image)

        return dst_image

    def _visualize_boxes(self, img: np.ndarray, boxes: List[np.ndarray]) -> None:
        """Visualize boxes."""
        img1 = img.copy()
        for box in boxes:
            cv2.polylines(img1, [box.astype(np.int32)], True, (0, 255, 0), 2)

        cv2.imwrite("tmp/det_boxes.jpg", img)

        return img1


if __name__ == "__main__":
    """Debugging."""
    import cv2

    from src.engine.ocr_det_onnx_engine import OcrDetOnnxEngine

    det_engine = OcrDetOnnxEngine(
        engine_path="tmp/models/ocr_det.onnx",
        provider="cpu",
    )
    det_engine.setup()

    rec_engine = OcrRecOnnxEngine(
        engine_path="tmp/models/ocr_rec.onnx",
        provider="cpu",
        max_batch_size=1,
    )
    rec_engine.setup()
    log.warning(f"Recognition metadata: {rec_engine.metadata}")

    img = cv2.imread("tmp/sample001.jpg")
    boxes = det_engine.predict(img)
    results = rec_engine.predict(img, boxes)
