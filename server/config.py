"""
Конфигурация сервера инпейнтинга
"""
from pathlib import Path
from typing import Literal

# Сервер
HOST = "127.0.0.1"
PORT = 7860

# Движок инпейнтинга.
# "comfyui" сознательно не входит в список — ветки реализации для него нет,
# выбор недопустимого значения должен падать явной ошибкой при старте, а не
# тихо откатываться на diffusers.
ENGINE_TYPE: Literal["flux", "powerpaint", "diffusers"] = "flux"

# Модели — FLUX.1 Fill dev (default)
FLUX_GGUF_REPO = "YarvixPA/FLUX.1-Fill-dev-gguf"
FLUX_GGUF_FILENAME = "flux1-fill-dev-Q5_K_M.gguf"
FLUX_BASE_MODEL = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_DEFAULT_GUIDANCE_SCALE = 30.0
FLUX_DEFAULT_NUM_INFERENCE_STEPS = 28

# PowerPaint v2 — BrushNet-based, 73% human preference for object removal
POWERPAINT_MODEL = "JunhaoZhuang/PowerPaint_v2"

# Legacy SD models (not used when ENGINE_TYPE = "powerpaint")
SDXL_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_MODEL = ""

# Дефолтные параметры инпейнтинга
DEFAULT_STRENGTH = 1.0  # 1.0 = full regeneration from context (object removal)
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CONTROLNET_SCALE = 0.5
DEFAULT_NUM_INFERENCE_STEPS = 20

# Негативный промпт для манхвы - агрессивный против галлюцинаций
DEFAULT_NEGATIVE_PROMPT = (
    "person, human, people, man, woman, boy, girl, child, face, body, figure, "
    "character, animal, creature, monster, "
    "text, letters, words, logo, watermark, signature, "
    "blurry, low quality, artifacts, noise, "
    "realistic, photo, 3d render, deformed, disfigured"
)

# Пути
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR_NAME = "_AI_CACHE"
OUTPUT_DIR_NAME = "_AI_OUT"

# Кэш
CACHE_ENABLED = False

# Логирование
LOG_LEVEL = "INFO"

# Отладка: сохранять промежуточные PNG (raw image/mask, model input/output,
# final result) в /tmp на каждый запрос. Раньше это было включено всегда —
# лишний I/O и утечка кадров пользователя в системный temp на каждый инпейнт.
DEBUG_SAVE_INTERMEDIATE = False
