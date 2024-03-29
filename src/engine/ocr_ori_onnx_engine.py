"""OCR text orientation classifier ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

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
        oris: List[str] = []
        for i in tqdm(range(0, len(imgs), self.max_batch_size), desc="Orientation"):
            batch_imgs = imgs[i : i + self.max_batch_size]
            batch_oris: np.ndarray = self.engine.run(
                [self.metadata[0].output_name],
                {self.metadata[0].input_name: batch_imgs},
            )[0]
            results = self.postprocess_oris(batch_oris)
            oris.extend(results)

        return oris

    def postprocess_oris(self, oris: np.ndarray) -> List[str]:
        """
        Post-process orientation results.

        Args:
            oris (np.ndarray): Orientation results in shape (n, 2)

        Returns:
            List[str]: List of orientation categories
        """

        return [self.categories[np.argmax(ori)] for ori in oris]
