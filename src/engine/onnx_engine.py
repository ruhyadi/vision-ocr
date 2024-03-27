"""ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

from pathlib import Path
from typing import List, Union

import onnxruntime as ort

from src.schema.onnx_schema import OnnxMetadataSchema
from src.utils.logger import get_logger

log = get_logger()


class OnnxEngine:
    """ONNX runtime engine module."""

    def __init__(self, engine_path: str, provider: str = "cpu") -> None:
        """
        Initialize ONNX runtime engine.

        Args:
            engine_path (str): Path to ONNX runtime engine file.
            provider (str): Provider for ONNX runtime engine. Defaults to "cpu".
        """
        self.engine_path = Path(engine_path)
        self.provider = provider
        self.provider = self.check_providers(provider)

    def setup(self) -> None:
        """Setup ONNX runtime engine."""
        self.engine = ort.InferenceSession(
            str(self.engine_path), providers=self.provider
        )
        self.metadata = self.get_metadata()

    def get_metadata(self) -> List[OnnxMetadataSchema]:
        """
        Get model metadata.

        Returns:
            List[OnnxMetadataSchema]: List of model metadata.
        """
        inputs = self.engine.get_inputs()
        outputs = self.engine.get_outputs()

        result: List[OnnxMetadataSchema] = []
        for inp, out in zip(inputs, outputs):
            result.append(
                OnnxMetadataSchema(
                    input_name=inp.name,
                    input_shape=inp.shape,
                    output_name=out.name,
                    output_shape=out.shape,
                )
            )

        return result

    def check_providers(self, provider: Union[str, list]) -> list:
        """
        Check available providers. If provider is not available, use CPU instead.

        Args:
            provider (Union[str, list]): Provider for ONNX runtime engine.

        Returns:
            list: List of available providers.
        """
        assert provider in ["cpu", "gpu"], "Invalid provider"
        available_providers = ort.get_available_providers()
        log.debug(f"Available providers: {available_providers}")
        if provider == "cpu" and "OpenVINOExecutionProvider" in available_providers:
            provider = ["CPUExecutionProvider", "OpenVINOExecutionProvider"]
        elif provider == "gpu" and "CUDAExecutionProvider" in available_providers:
            provider = ["CUDAExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        return provider
