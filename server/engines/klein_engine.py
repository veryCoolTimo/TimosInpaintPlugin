"""
FLUX.2 [klein] 4B — инпейнтинг через diffusers Flux2KleinInpaintPipeline.

Почему klein вместо FLUX.1 Fill:
- 4B параметров вместо 12B, дистиллированная — 4 шага вместо 28;
- Apache 2.0 и не gated: не нужен HuggingFace-логин;
- вышла в январе 2026, пайплайн инпейнта в diffusers с 0.38 (апрель 2026).

Особенности пайплайна (проверено по исходникам diffusers 0.38–0.40):
- при переданном image пайплайн сам ужимает вход до ~1 МП и округляет
  стороны вниз до кратного 16 — поэтому max_resolution=1024 и
  size_divisor=16, тогда вход проходит без изменений;
- маска бинаризуется по 0.5 и сжимается до сетки латентов (1/16) билинейно
  без сглаживания — штрих тоньше 16 px пропадал целиком, поэтому маска
  заранее расширяется до сетки (grid_dilate_mask); мягкий край даёт наша
  склейка в pipeline.finish, а не модель;
- для дистиллированной модели CFG выключен, guidance_scale > 1 игнорируется.
"""
import gc
import logging
import platform
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)

# Только то, что нужно пайплайну: в корне репозитория лежит ещё дубль
# трансформера одним файлом (+7.75 ГБ), который from_pretrained не нужен
_DOWNLOAD_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
]

# Промпт по умолчанию: без него модель не знает, чем заполнять дыру
DEFAULT_FILL_PROMPT = "seamless continuation of the surrounding image, clean background"


def grid_dilate_mask(mask: Image.Image, cell: int = 16) -> Image.Image:
    """
    Каждая ячейка cell×cell становится белой, если в ней есть хоть один
    белый пиксель. Пайплайн сжимает маску до латентов билинейно, и каждая
    ячейка латента смотрит лишь на пару центральных пикселей блока: провод
    в 10 px или край объекта до ~8 px модель иначе не перерисовывала.
    """
    arr = np.array(mask.convert("L")) > 127
    h, w = arr.shape
    blocks = arr.reshape(h // cell, cell, w // cell, cell).any(axis=(1, 3))
    full = np.repeat(np.repeat(blocks, cell, axis=0), cell, axis=1)
    return Image.fromarray((full * 255).astype(np.uint8), "L")


class Flux2KleinEngine(BaseEngine):
    """
    Инпейнтинг через FLUX.2 [klein] 4B (дистиллированная).
    ~16 ГБ весов в bf16 (трансформер 7.75 + текстовый энкодер Qwen3 8 +
    VAE 0.2); на Mac работает целиком в unified memory без CPU offload.
    """

    max_resolution = 1024
    size_divisor = 16

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-4B",
        default_guidance_scale: float = 1.0,
        default_num_inference_steps: int = 4,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.default_guidance_scale = default_guidance_scale
        self.default_num_inference_steps = default_num_inference_steps

        if device:
            self._device = device
        elif torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self.pipe = None
        # Доп. аргументы пайплайна (тесты с крошечной моделью задают свои
        # text_encoder_out_layers)
        self.pipe_kwargs = {}
        logger.info(f"Flux2KleinEngine initialized, device: {self._device}")

    @property
    def name(self) -> str:
        return "flux2-klein-4b"

    @property
    def device(self) -> str:
        return self._device

    @property
    def supports_controlnet(self) -> bool:
        return False

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def load(self) -> None:
        if self.pipe is not None:
            return

        from diffusers import Flux2KleinInpaintPipeline
        from huggingface_hub import snapshot_download

        if self._device == "mps":
            major = int((platform.mac_ver()[0] or "0").split(".")[0])
            if major < 14:
                # bf16 на MPS — с macOS 14; на 13 падало бы с невнятной
                # ошибкой dtype на первом шаге
                raise RuntimeError(
                    f"FLUX.2 klein needs macOS 14 (Sonoma) or newer for bfloat16 on Apple GPU, "
                    f"found macOS {platform.mac_ver()[0]}. Switch ENGINE_TYPE in server/config.py."
                )

        # bf16: так модель обучена; fp16 у FLUX известен чёрными картинками.
        # На CPU тоже bf16 — float32 удвоил бы и без того 16 ГБ
        def from_dir(local_files_only):
            local_dir = snapshot_download(
                self.model_id, allow_patterns=_DOWNLOAD_PATTERNS, local_files_only=local_files_only
            )
            return Flux2KleinInpaintPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16)

        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            # Сначала без сети: модель уже скачана — старт без обращений к HF
            pipe = from_dir(local_files_only=True)
        except (LocalEntryNotFoundError, OSError, ValueError) as e:
            logger.info(f"{self.model_id} not fully cached ({type(e).__name__}) — downloading (~16 GB)...")
            pipe = from_dir(local_files_only=False)
        logger.info(f"Loading FLUX.2 klein on {self._device} (bf16)...")
        # Без enable_model_cpu_offload: на unified memory Mac он ничего не
        # экономит и только гоняет веса туда-обратно на каждом шаге
        self.pipe = pipe.to(self._device)
        logger.info("FLUX.2 klein loaded")

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        gc.collect()
        if self._device == "mps":
            torch.mps.empty_cache()
        elif self._device == "cuda":
            torch.cuda.empty_cache()
        logger.info("FLUX.2 klein unloaded")

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 1.0,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        controlnet_scale: float = 0.5,
        seed: Optional[int] = None,
        step_callback=None,
        **kwargs,
    ) -> Image.Image:
        """klein игнорирует negative_prompt (CFG выключен), controlnet_scale."""
        if self.pipe is None:
            raise RuntimeError("FLUX.2 klein not loaded. Call load() first.")

        if guidance_scale is None:
            guidance_scale = self.default_guidance_scale
        if not num_inference_steps:
            num_inference_steps = self.default_num_inference_steps

        image = image.convert("RGB")
        mask = mask.convert("L")
        if mask.size != image.size:
            raise ValueError(f"mask size {mask.size} != image size {image.size}")
        w, h = image.size
        if w % self.size_divisor or h % self.size_divisor or w * h > self.max_resolution ** 2:
            # pipeline.prepare уже даёт подходящий размер; это страховка от
            # тихого ресайза внутри пайплайна, после которого результат не
            # совпал бы с маской
            raise ValueError(f"klein input must be ≤1 MP with sides divisible by 16, got {w}x{h}")

        # CPU-генератор: torch.Generator("mps") для сидирования не годится
        generator = torch.Generator("cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        callback_fn = None
        if step_callback is not None:
            def callback_fn(pipe, step_index, timestep, callback_kwargs):
                # Число шагов зависит от strength по формуле пайплайна —
                # берём его у пайплайна. step_callback поднимает
                # JobCancelled при Stop — исключение прерывает генерацию
                step_callback(step_index + 1, pipe.num_timesteps)
                return callback_kwargs

        prompt = prompt.strip() or DEFAULT_FILL_PROMPT
        logger.info(
            f"klein inpaint: {w}x{h}, steps={num_inference_steps}, strength={strength}, "
            f"guidance={guidance_scale}, seed={seed}, prompt={prompt[:80]!r}"
        )

        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=grid_dilate_mask(mask, self.size_divisor),
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=callback_fn,
            **self.pipe_kwargs,
        ).images[0]

        logger.info(f"klein inpaint complete: {result.size}")
        return result
