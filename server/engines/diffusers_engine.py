"""
Движок инпейнтинга на основе Diffusers (SD 1.5 / SDXL)
"""
import logging
from typing import Optional

import torch
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)


class DiffusersEngine(BaseEngine):
    """
    Инпейнтинг через Diffusers.
    Автоматически выбирает pipeline: SDXL или SD 1.5.
    """

    def __init__(
        self,
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.is_sdxl = "xl" in model_id.lower()
        self.max_resolution = 1024 if self.is_sdxl else 512

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

        logger.info(f"DiffusersEngine initialized, device: {self.device}, SDXL: {self.is_sdxl}")

    @property
    def name(self) -> str:
        return "sdxl" if self.is_sdxl else "sd15"

    @property
    def supports_controlnet(self) -> bool:
        return False

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def load(self) -> None:
        """Загружает Inpainting pipeline"""
        if self.is_loaded():
            logger.info("Model already loaded")
            return

        logger.info(f"Loading Inpainting model: {self.model_id}")

        # MPS (Apple Silicon) REQUIRES float32 - float16 causes NaN values
        dtype = torch.float32 if self.device in ["mps", "cpu"] else torch.float16

        if self.is_sdxl:
            from diffusers import StableDiffusionXLInpaintPipeline
            logger.info("Using SDXL Inpainting pipeline")
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )
        else:
            from diffusers import StableDiffusionInpaintPipeline
            logger.info("Using SD 1.5 Inpainting pipeline")
            is_local = self.model_id.endswith('.ckpt') or self.model_id.endswith('.safetensors')
            if is_local:
                self.pipe = StableDiffusionInpaintPipeline.from_single_file(
                    self.model_id, torch_dtype=dtype, safety_checker=None,
                )
            else:
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_id, torch_dtype=dtype, safety_checker=None,
                )

        # Заменяем scheduler на DPM++ 2M Karras — быстрая сходимость,
        # хорошее качество при 20 шагах (default PNDM требует 30+)
        from diffusers import DPMSolverMultistepScheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        logger.info("Scheduler set to DPM++ 2M Karras")

        self.pipe.to(self.device)

        # Оптимизации для Mac
        if self.device == "mps":
            self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, 'safety_checker'):
                self.pipe.safety_checker = None
            logger.info("MPS optimizations enabled: attention_slicing")

        logger.info("Model loaded successfully")

    def unload(self) -> None:
        """Выгружает модель из памяти"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        # gc.collect() до empty_cache(): на MPS empty_cache() часто не
        # освобождает память, пока Python не собрал сборщиком мусора сами
        # объекты (и их MPS-тензоры), на которые ссылался pipe.
        import gc
        gc.collect()
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
        strength: float = 1.0,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        controlnet_scale: float = 0.5,
        seed: Optional[int] = None,
        step_callback=None,
        **kwargs,
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

        logger.info(
            f"Running inpaint: size={image.size}, "
            f"strength={strength}, steps={num_inference_steps}, "
            f"guidance={guidance_scale}"
        )
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Negative: {negative_prompt}")

        # Callback для прогресса (diffusers 0.21: callback(step, timestep, latents))
        def on_step(step, timestep, latents):
            if step_callback:
                step_callback(step + 1, num_inference_steps)

        # Запускаем инпейнтинг
        w, h = image.size
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            height=h,
            width=w,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        if step_callback:
            pipe_kwargs["callback"] = on_step
            pipe_kwargs["callback_steps"] = 1

        result = self.pipe(**pipe_kwargs).images[0]

        logger.info("Inpaint completed")

        return result
