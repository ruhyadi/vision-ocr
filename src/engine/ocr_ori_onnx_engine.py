"""OCR text orientation classifier ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

import time
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from src.engine.onnx_engine import OnnxEngine
from src.utils.logger import get_logger

log = get_logger()


class OcrOriOnnxEngine(OnnxEngine):
    """OCR text orientation classifier engine with ONNX runtime."""

    def __init__(
        self,
        engine_path: str,
        provider: str = "cpu",
        max_batch_size: int = 8,
        categories: List[str] = ["up", "down"],
    ) -> None:
        """Initialize OCR text orientation classifier engine."""
        super().__init__(engine_path, provider)
        self.max_batch_size = max_batch_size
        self.categories = categories

    def predict(self, imgs: List[np.ndarray]):
        """Predict text orientation from text images."""
        # iterate per batch
        oris: List[str] = []
        for i in tqdm(range(0, len(imgs), self.max_batch_size), desc="Orientation"):
            batch_imgs = imgs[i : i + self.max_batch_size]
            # batch_oris has shape (n, 2)
            batch_oris: np.ndarray = self.engine.run(
                [self.metadata[0].output_name],
                {self.metadata[0].input_name: batch_imgs},
            )[0]
            results = self.postprocess_oris(batch_oris)
            oris.extend(results)

        return oris

    def preprocess_imgs(self, imgs: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess images.
        Images should be in form [-1, 3, 48, 192].
        """
        resized_imgs = np.zeros((len(imgs), 3, 48, 192), dtype=np.float32)
        for i, img in enumerate(imgs):
            img = img.transpose(1, 2, 0)  # CHW -> HWC
            img = cv2.resize(img, (192, 48))
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            resized_imgs[i] = img

        return resized_imgs

    def postprocess_oris(self, oris: np.ndarray) -> List[str]:
        """
        Post-process orientation results.

        Args:
            oris (np.ndarray): Orientation results in shape (n, 2)

        Returns:
            List[str]: List of orientation categories
        """

        return [self.categories[np.argmax(ori)] for ori in oris]


if __name__ == "__main__":

    engine = OcrOriOnnxEngine(
        engine_path="tmp/models/ocr_ori_cls.onnx",
        provider="cpu",
    )
    engine.setup()
    log.warning(f"Metadata: {engine.metadata}")
