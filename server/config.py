"""
Конфигурация сервера инпейнтинга
"""
from pathlib import Path
from typing import Literal

# Сервер
HOST = "127.0.0.1"
PORT = 7860

# Движок инпейнтинга
ENGINE_TYPE: Literal["diffusers", "comfyui"] = "diffusers"

# Модели
SDXL_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_lineart"

# Дефолтные параметры инпейнтинга
DEFAULT_STRENGTH = 0.85
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CONTROLNET_SCALE = 0.5
DEFAULT_NUM_INFERENCE_STEPS = 30

# Негативный промпт для манхвы
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, signature, "
    "realistic, photo, 3d render, deformed"
)

# Пути
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR_NAME = "_AI_CACHE"
OUTPUT_DIR_NAME = "_AI_OUT"

# Кэш
CACHE_ENABLED = True

# Логирование
LOG_LEVEL = "INFO"
