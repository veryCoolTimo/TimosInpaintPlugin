"""
Конфигурация сервера инпейнтинга
"""
from pathlib import Path
from typing import Literal

# Сервер
HOST = "127.0.0.1"
PORT = 7860

# Движок режима AI Gen.
# "klein" — FLUX.2 [klein] 4B (по умолчанию): 4 шага, Apache 2.0, без
#           HuggingFace-логина, ~16 ГБ весов.
# "flux"  — FLUX.1 Fill dev GGUF (прежний движок): 28 шагов, gated-репо,
#           нужен `hf auth login`.
# "diffusers" — SD 1.5 / SDXL inpaint, 512–1024 px.
ENGINE_TYPE: Literal["klein", "flux", "diffusers"] = "klein"

# FLUX.2 [klein] 4B (дистиллированная: 4 шага, guidance выключен)
KLEIN_MODEL = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_DEFAULT_GUIDANCE_SCALE = 1.0
KLEIN_DEFAULT_NUM_INFERENCE_STEPS = 4

# FLUX.1 Fill dev (ENGINE_TYPE = "flux")
FLUX_GGUF_REPO = "YarvixPA/FLUX.1-Fill-dev-gguf"
# Было "flux1-fill-dev-Q5_K_M.gguf" — этого файла в репозитории нет и,
# похоже, никогда не было под этим именем (проверено через HF API): доступны
# Q3_K_S/Q4_0/Q4_1/Q4_K_S/Q5_0/Q5_1/Q5_K_S/Q6_K/Q8_0. Q5_K_S — 8.29GB, что
# совпадает с комментарием "~8GB" в engines/flux_engine.py, поэтому это,
# похоже, и был исходный замысел (переименование апстримом или опечатка).
# Без этого фикса /load падал с 404 при любой попытке использовать FLUX.
FLUX_GGUF_FILENAME = "flux1-fill-dev-Q5_K_S.gguf"
FLUX_BASE_MODEL = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_DEFAULT_GUIDANCE_SCALE = 30.0
FLUX_DEFAULT_NUM_INFERENCE_STEPS = 28

# SD / SDXL inpaint (ENGINE_TYPE = "diffusers")
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

# Логирование
LOG_LEVEL = "INFO"

# Отладка: сохранять вход/маску/выход модели в $TMPDIR/ae-inpaint/ на
# каждый запрос. Выключено: лишний I/O и копии кадров пользователя.
DEBUG_SAVE_INTERMEDIATE = False
