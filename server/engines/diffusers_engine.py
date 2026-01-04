"""
Движок инпейнтинга на основе Diffusers + SDXL
"""
import logging
from typing import Optional

import torch
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)


class DiffusersEngine(BaseEngine):
    """
    Инпейнтинг через Diffusers с SDXL Inpainting.
    Опционально поддерживает ControlNet для сохранения lineart.
    """

    def __init__(
        self,
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.controlnet_id = controlnet_id

        # Определяем устройство
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.pipe = None
        self.controlnet = None
        self.lineart_processor = None

        logger.info(f"DiffusersEngine initialized, device: {self.device}")

    @property
    def name(self) -> str:
        return "diffusers"

    @property
    def supports_controlnet(self) -> bool:
        return self.controlnet is not None

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def load(self) -> None:
        """Загружает SDXL Inpainting pipeline"""
        if self.is_loaded():
            logger.info("Model already loaded")
            return

        logger.info(f"Loading Inpainting model: {self.model_id}")

        from diffusers import StableDiffusionInpaintPipeline

        # Check if loading from local .ckpt file or HuggingFace
        is_local_ckpt = self.model_id.endswith('.ckpt') or self.model_id.endswith('.safetensors')

        # MPS (Apple Silicon) REQUIRES float32 - float16 causes NaN values
        dtype = torch.float32 if self.device in ["mps", "cpu"] else torch.float16

        if is_local_ckpt:
            logger.info(f"Loading from local checkpoint: {self.model_id}")
            self.pipe = StableDiffusionInpaintPipeline.from_single_file(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
            )
        else:
            logger.info(f"Loading from HuggingFace: {self.model_id}")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                local_files_only=True,
            )

        self.pipe.to(self.device)

        # Оптимизации для Mac
        if self.device == "mps":
            self.pipe.enable_attention_slicing()
            # Note: VAE tiling causes tensor shape errors on MPS, so we don't enable it
            # float32 alone should prevent NaN issues
            # Disable safety checker (often causes issues on MPS)
            self.pipe.safety_checker = None
            logger.info("MPS optimizations enabled: attention_slicing, safety_checker disabled")

        # Загружаем ControlNet если указан
        if self.controlnet_id:
            self._load_controlnet()

        logger.info("Model loaded successfully")

    def _load_controlnet(self) -> None:
        """Загружает ControlNet для lineart"""
        try:
            from controlnet_aux import LineartDetector

            logger.info(f"Loading ControlNet: {self.controlnet_id}")

            # Процессор для извлечения lineart
            self.lineart_processor = LineartDetector.from_pretrained(
                "lllyasviel/Annotators"
            )

            logger.info("ControlNet loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load ControlNet: {e}")
            self.controlnet = None
            self.lineart_processor = None

    def unload(self) -> None:
        """Выгружает модель из памяти"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        if self.controlnet is not None:
            del self.controlnet
            self.controlnet = None

        if self.lineart_processor is not None:
            del self.lineart_processor
            self.lineart_processor = None

        # Очищаем память
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")

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
        """Выполняет инпейнтинг"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        # Устанавливаем сид
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Убеждаемся что изображения правильного размера
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)

        # Промпт по умолчанию для манхвы
        if not prompt:
            prompt = "clean background, manga style, high quality lineart"

        if not negative_prompt:
            negative_prompt = (
                "blurry, low quality, watermark, signature, "
                "realistic, photo, 3d render, deformed"
            )

        logger.info(
            f"Running inpaint: size={image.size}, "
            f"strength={strength}, steps={num_inference_steps}"
        )

        # Запускаем инпейнтинг
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        logger.info("Inpaint completed")

        return result
