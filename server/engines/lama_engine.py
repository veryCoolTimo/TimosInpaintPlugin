"""
Движок инпейнтинга на основе LaMa (Large Mask Inpainting).
Специализируется на удалении объектов — заполняет маску контекстом из окружения.
"""
import logging
from typing import Optional

import torch
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)


class LamaEngine(BaseEngine):
    """
    Инпейнтинг через LaMa — быстрое удаление объектов без промптов.
    Один forward pass (~1-3 сек на GPU/MPS, дольше на CPU).
    """

    def __init__(self, device: Optional[str] = None):
        self._model = None
        if device:
            self._device = device
        elif torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        logger.info(f"LamaEngine initialized, device: {self._device}")

    @property
    def name(self) -> str:
        return "lama"

    @property
    def supports_controlnet(self) -> bool:
        return False

    @property
    def device(self) -> str:
        return self._device

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded():
            logger.info("LaMa already loaded")
            return

        logger.info(f"Loading LaMa model on {self._device}...")
        from simple_lama_inpainting import SimpleLama
        try:
            # Not all versions of simple-lama-inpainting accept a device
            # kwarg — fall back to its own default (previously hardcoded
            # to CPU here regardless of what was actually available) if not.
            self._model = SimpleLama(device=torch.device(self._device))
        except TypeError:
            logger.warning(
                "Installed simple-lama-inpainting doesn't support a device "
                "argument — falling back to its default device."
            )
            self._model = SimpleLama()
        logger.info("LaMa model loaded")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        logger.info("LaMa model unloaded")

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 1.0,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        controlnet_scale: float = 0.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        if not self.is_loaded():
            raise RuntimeError("LaMa model not loaded. Call load() first.")

        # LaMa не использует промпты — только изображение + маска
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)

        # Маска должна быть L mode, white=inpaint
        if mask.mode != "L":
            mask = mask.convert("L")

        logger.info(f"Running LaMa inpaint: size={image.size}")
        result = self._model(image, mask)
        logger.info("LaMa inpaint completed")

        return result
