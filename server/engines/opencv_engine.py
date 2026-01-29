"""
OpenCV-based inpainting engine.
Uses classical algorithms (Telea/Navier-Stokes) - no hallucinations.
Best for simple texture continuation.
"""
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)


class OpenCVEngine(BaseEngine):
    """
    Classical inpainting using OpenCV.
    Fast, no AI hallucinations, good for simple patterns.
    """

    def __init__(self, method: str = "telea"):
        """
        Args:
            method: "telea" or "ns" (Navier-Stokes)
        """
        self.method = method
        self._loaded = False
        logger.info(f"OpenCVEngine initialized, method: {method}")

    @property
    def name(self) -> str:
        return "opencv"

    @property
    def supports_controlnet(self) -> bool:
        return False

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """No model to load for OpenCV"""
        self._loaded = True
        logger.info("OpenCV engine ready")

    def unload(self) -> None:
        """Nothing to unload"""
        self._loaded = False
        logger.info("OpenCV engine stopped")

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 0.85,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        controlnet_scale: float = 0.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Performs inpainting using OpenCV.
        Most parameters are ignored (only used for API compatibility).
        """
        if not self.is_loaded():
            raise RuntimeError("Engine not loaded. Call load() first.")

        # Convert to numpy
        img_np = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Convert mask to grayscale
        mask_np = np.array(mask.convert("L"))
        # OpenCV expects white = inpaint area
        # Our mask is white = inpaint area, so it's correct

        # Inpaint radius based on mask size
        # Larger radius = better for bigger areas but slower
        radius = 5

        # Select algorithm
        if self.method == "ns":
            flags = cv2.INPAINT_NS
        else:
            flags = cv2.INPAINT_TELEA

        logger.info(f"Running OpenCV inpaint: method={self.method}, radius={radius}")

        # Perform inpainting
        result_bgr = cv2.inpaint(img_bgr, mask_np, radius, flags)

        # Convert back to RGB PIL Image
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(result_rgb)

        logger.info("OpenCV inpaint completed")

        return result
