"""
FastAPI сервер для инпейнтинга
"""
import asyncio
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from starlette.concurrency import run_in_threadpool

import config
from engines import DiffusersEngine, LamaEngine, UpscaleEngine, PowerPaintEngine, FluxFillEngine
from engines.opencv_engine import OpenCVEngine
from engines.base import BaseEngine
from utils import base64_to_image, image_to_base64, CacheManager
from utils.image import (
    ensure_rgb,
    ensure_mask_format,
    resize_for_model,
    apply_mask_feather,
    expand_mask,
    crop_to_mask,
    paste_back,
)


class JobCancelled(Exception):
    """Поднимается из step_callback, когда пользователь нажал Stop."""
    pass

# Настройка логирования — и в консоль, и в файл
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/ae_inpaint_server.log"),
    ]
)
logger = logging.getLogger(__name__)

# Глобальные объекты
ai_engine: Optional[BaseEngine] = None
lama_engine: Optional[LamaEngine] = None
opencv_engine: Optional[OpenCVEngine] = None
upscale_engine: Optional[UpscaleEngine] = None
cache_manager: Optional[CacheManager] = None

# Progress tracking
progress_info = {
    "step": 0,
    "total_steps": 0,
    "stage": "idle",  # idle, loading, inpainting, upscaling, done
}

# Единая блокировка вокруг load/infer/unload — сервер локальный,
# однопользовательский, поэтому просто сериализуем тяжёлые операции вместо
# полноценной очереди. Это же не даёт /unload выгрузить модель во время
# /inpaint и не даёт двум авто-загрузкам гоняться друг с другом.
engine_lock = asyncio.Lock()

# Состояние текущей job. cancel_event — threading.Event (не asyncio!), потому
# что проверяется из step_callback, который вызывается синхронно внутри
# threadpool-воркера, а не в event loop.
current_job = {
    "id": None,
    "active": False,
    "cancel_event": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: загрузка/выгрузка модели"""
    global ai_engine, lama_engine, opencv_engine, upscale_engine, cache_manager

    logger.info("Starting server...")

    # Инициализируем движки
    if config.ENGINE_TYPE == "flux":
        ai_engine = FluxFillEngine(
            gguf_repo=config.FLUX_GGUF_REPO,
            gguf_filename=config.FLUX_GGUF_FILENAME,
            base_model=config.FLUX_BASE_MODEL,
            default_guidance_scale=config.FLUX_DEFAULT_GUIDANCE_SCALE,
            default_num_inference_steps=config.FLUX_DEFAULT_NUM_INFERENCE_STEPS,
        )
    elif config.ENGINE_TYPE == "powerpaint":
        ai_engine = PowerPaintEngine(model_id=config.POWERPAINT_MODEL)
    elif config.ENGINE_TYPE == "diffusers":
        ai_engine = DiffusersEngine(
            model_id=config.SDXL_INPAINT_MODEL,
            controlnet_id=config.CONTROLNET_MODEL if config.CONTROLNET_MODEL else None,
        )
    else:
        raise RuntimeError(
            f"Unknown ENGINE_TYPE={config.ENGINE_TYPE!r} in config.py. "
            f"Expected 'flux', 'powerpaint' or 'diffusers'."
        )

    lama_engine = LamaEngine()
    # LaMa loads lazily on first request

    opencv_engine = OpenCVEngine(method="telea")
    opencv_engine.load()  # OpenCV doesn't need heavy loading

    upscale_engine = UpscaleEngine()
    # Upscale engine loads lazily on first request

    logger.info("Server started")
    yield

    # Cleanup
    if ai_engine and ai_engine.is_loaded():
        ai_engine.unload()
    if lama_engine and lama_engine.is_loaded():
        lama_engine.unload()
    if upscale_engine and upscale_engine.is_loaded():
        upscale_engine.unload()

    logger.info("Server stopped")


app = FastAPI(
    title="AE Inpaint Server",
    description="Локальный сервер инпейнтинга для After Effects плагина",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS для CEP панели.
# CEP-панель грузится как локальный файл и не шлёт полезного Origin, так что
# ограничить allow_origins конкретным значением тут особо нечем — но
# allow_credentials=True с открытым "*" вместе создают ситуацию, когда любая
# случайно открытая в браузере страница потенциально может дёргать этот
# локальный сервер. Credentials серверу не нужны (нет cookie/auth) — убираем.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# === Модели запросов ===

class InpaintRequest(BaseModel):
    """Запрос на инпейнтинг"""
    image: str = Field(..., description="Base64 PNG изображения")
    mask: str = Field(default="", description="Base64 PNG маски (белый = inpaint). Пусто = авто из альфа")
    mode: Literal["remove", "ai", "clean"] = Field(default="remove", description="Режим: 'remove' (LaMa), 'ai' (SD/FLUX/PowerPaint), 'clean' (OpenCV)")
    prompt: str = Field(default="", description="Текстовый промпт")
    negative_prompt: str = Field(default="", description="Негативный промпт")
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=50.0)
    num_steps: int = Field(default=20, ge=10, le=100)
    controlnet_scale: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: Optional[int] = Field(default=None)
    feather: int = Field(default=0, ge=0, le=50, description="Feather маски в px")
    expand: int = Field(default=0, ge=0, le=50, description="Expand маски в px")
    crop_to_mask: bool = Field(default=True, description="Обрезать по маске для ускорения")
    crop_padding: int = Field(default=128, ge=16, le=512, description="Отступ при обрезке по маске")
    invert_mask: bool = Field(default=False, description="Инвертировать маску")
    fill_transparent: bool = Field(default=False, description="Заполнить прозрачные области под маской")
    cache_dir: Optional[str] = Field(default=None, description="Путь к папке кэша проекта")


class InpaintResponse(BaseModel):
    """Ответ с результатом инпейнтинга"""
    result: str = Field(..., description="Base64 PNG результата")
    cached: bool = Field(default=False, description="Результат из кэша")
    width: int
    height: int


class UpscaleRequest(BaseModel):
    """Запрос на апскейл"""
    image: str = Field(..., description="Base64 PNG изображения")
    scale: int = Field(default=4, description="Множитель масштаба: 2 или 4")
    model_type: str = Field(default="anime", description="Тип модели: 'anime' или 'general'")


class UpscaleResponse(BaseModel):
    """Ответ с результатом апскейла"""
    result: str = Field(..., description="Base64 PNG результата")
    width: int
    height: int
    scale: int


class HealthResponse(BaseModel):
    """Статус сервера"""
    status: str
    engine: str
    engine_loaded: bool
    device: str
    model_cached: bool = False
    busy: bool = False


# === Эндпоинты ===

def is_model_cached() -> bool:
    """Проверяет есть ли модель в кэше HuggingFace"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if config.ENGINE_TYPE == "flux":
        # Check both GGUF file and base model
        gguf_name = config.FLUX_GGUF_REPO.replace("/", "--")
        base_name = config.FLUX_BASE_MODEL.replace("/", "--")
        return (cache_dir / f"models--{gguf_name}").exists() and (cache_dir / f"models--{base_name}").exists()
    elif config.ENGINE_TYPE == "powerpaint":
        model_name = config.POWERPAINT_MODEL.replace("/", "--")
    else:
        model_name = config.SDXL_INPAINT_MODEL.replace("/", "--")
    model_dir = cache_dir / f"models--{model_name}"
    return model_dir.exists()


@app.get("/progress")
async def get_progress():
    """Текущий прогресс инпейнтинга.

    Инференс теперь выполняется в threadpool (см. /inpaint), поэтому этот
    эндпоинт реально отвечает во время генерации, а не зависает вместе с
    event loop-ом, как раньше.
    """
    return progress_info


@app.post("/cancel")
async def cancel_job():
    """Отменяет текущую активную job (Stop в панели).

    Best-effort: выставляет cancel_event, который проверяется между шагами
    инференса в step_callback. Диффузионные движки (FLUX/PowerPaint/SD)
    остановятся на следующем шаге; LaMa/OpenCV успевают завершиться раньше,
    чем клиент вообще пришлёт /cancel, потому что работают за 1-3 секунды.
    """
    if not current_job["active"] or current_job["cancel_event"] is None:
        return {"status": "no_active_job"}
    current_job["cancel_event"].set()
    logger.info(f"Cancel requested for job {current_job['id']}")
    return {"status": "cancelling", "job_id": current_job["id"]}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервера"""
    return HealthResponse(
        status="ok",
        engine=ai_engine.name if ai_engine else "none",
        engine_loaded=ai_engine.is_loaded() if ai_engine else False,
        device=ai_engine.device if ai_engine else "unknown",
        model_cached=is_model_cached(),
        busy=current_job["active"],
    )


@app.post("/load")
async def load_model():
    """Загружает AI модель в память"""
    if ai_engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    if ai_engine.is_loaded():
        return {"status": "already_loaded"}

    if current_job["active"]:
        raise HTTPException(status_code=409, detail="Server is busy with another job")

    async with engine_lock:
        try:
            await run_in_threadpool(ai_engine.load)
            return {"status": "loaded"}
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_model():
    """Выгружает AI модель из памяти"""
    if ai_engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    if current_job["active"]:
        raise HTTPException(status_code=409, detail="Cannot unload while a job is running")

    async with engine_lock:
        ai_engine.unload()
    return {"status": "unloaded"}


@app.post("/inpaint", response_model=InpaintResponse)
async def inpaint(request: InpaintRequest):
    """Выполняет инпейнтинг"""
    global cache_manager, progress_info

    if current_job["active"]:
        raise HTTPException(status_code=409, detail="Server is busy with another job. Stop it first or wait.")

    # Выбираем движок по режиму
    mode = request.mode
    if mode == "clean":
        engine = opencv_engine
    elif mode == "remove":
        engine = lama_engine
    else:
        engine = ai_engine

    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    job_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    current_job.update({"id": job_id, "active": True, "cancel_event": cancel_event})
    progress_info = {"step": 0, "total_steps": 0, "stage": "preparing"}

    async with engine_lock:
      try:
        # Автозагрузка модели при первом запросе — в threadpool, чтобы не
        # блокировать event loop (иначе /progress и /cancel зависают вместе
        # с загрузкой модели на несколько минут).
        if not engine.is_loaded():
            logger.info(f"Auto-loading {engine.name} model on first request...")
            progress_info = {"step": 0, "total_steps": 0, "stage": "loading_model"}
            try:
                await run_in_threadpool(engine.load)
            except Exception as e:
                progress_info["stage"] = "error"
                logger.error(f"Failed to auto-load model: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

        # Debug: log received data sizes
        logger.info(f"Received image base64 length: {len(request.image)}")
        logger.info(f"Received mask base64 length: {len(request.mask)}")

        # Декодируем изображения
        image = base64_to_image(request.image)
        logger.info(f"Decoded image: {image.mode} {image.size}")

        if config.DEBUG_SAVE_INTERMEDIATE:
            image.save("/tmp/ae_debug_raw_image.png")

        if request.mask:
            mask = base64_to_image(request.mask)
            logger.info(f"Decoded mask: {mask.mode} {mask.size}")
            if config.DEBUG_SAVE_INTERMEDIATE:
                mask.save("/tmp/ae_debug_raw_mask.png")
        else:
            # No mask provided - generate from alpha channel (expand mode)
            logger.info("No mask provided, generating from alpha channel")
            import numpy as np
            if image.mode == 'RGBA':
                alpha = np.array(image.split()[3])
                # White where transparent (alpha=0), black where opaque
                mask_array = np.where(alpha < 128, 255, 0).astype(np.uint8)
                mask = Image.fromarray(mask_array, 'L')
                logger.info(f"Generated alpha mask: {mask.size}, white pixels: {np.sum(mask_array > 128)}")
            else:
                # No alpha channel - can't generate mask
                raise HTTPException(status_code=400, detail="No mask provided and image has no alpha channel")

        # Сохраняем альфа-канал для восстановления прозрачного фона
        original_alpha = image.split()[3] if image.mode == "RGBA" else None
        if original_alpha:
            logger.info("Saved alpha channel for later restoration")
            # Заполняем прозрачные пиксели средним цветом непрозрачных,
            # чтобы модель не видела белый фон и не генерировала поверх
            import numpy as np
            img_arr = np.array(image)
            alpha_arr = img_arr[:, :, 3]
            opaque = alpha_arr > 128
            if np.any(opaque):
                avg_color = img_arr[opaque][:, :3].mean(axis=0).astype(np.uint8)
            else:
                avg_color = np.array([128, 128, 128], dtype=np.uint8)
            img_arr[~opaque, :3] = avg_color
            img_arr[~opaque, 3] = 255  # Make fully opaque so ensure_rgb won't composite onto white
            image = Image.fromarray(img_arr)
            logger.info(f"Filled transparent pixels with avg color: {avg_color.tolist()}")

        # Подготавливаем изображения
        image = ensure_rgb(image)
        mask = ensure_mask_format(mask, invert=request.invert_mask)
        logger.info(f"Mask inverted: {request.invert_mask}")

        # Применяем feather/expand к маске
        if request.feather > 0:
            mask = apply_mask_feather(mask, request.feather)
        if request.expand > 0:
            mask = expand_mask(mask, request.expand)

        # Сохраняем оригиналы для paste_back
        original_image = image.copy()
        original_mask = mask.copy()
        original_size = image.size
        crop_bbox = None

        # Обрезаем по маске для ускорения (LaMa и SD)
        use_ai = mode in ("ai", "remove")
        # Было isinstance(ai_engine, ...) — проверяло глобальный AI-движок,
        # а не выбранный для этого запроса engine. При ENGINE_TYPE=flux и
        # mode=remove (LaMa) divisor ошибочно брался как для FLUX (32 вместо 8).
        crop_divisor = 32 if (isinstance(engine, FluxFillEngine) and mode == "ai") else 8
        if request.crop_to_mask and use_ai:
            image, mask, crop_bbox = crop_to_mask(image, mask, padding=request.crop_padding, divisor=crop_divisor)
            if crop_bbox:
                logger.info(f"Cropped to mask: {original_size} -> {image.size} (bbox: {crop_bbox})")

        # Ресайз для моделей
        cropped_size = image.size
        if use_ai:
            is_flux = isinstance(engine, FluxFillEngine)
            max_model_size = 1024 if is_flux else 512
            divisor = 32 if is_flux else 8
            image = resize_for_model(image, max_size=max_model_size, divisor=divisor)
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            logger.info(f"Resized for model: {cropped_size} -> {image.size}")

        # Resolve guidance_scale/steps ДО построения ключа кэша — иначе кэш
        # ключуется по тому, что прислал клиент, а не по тому, что реально
        # ушло в модель после FLUX-переопределений (см. ниже).
        effective_guidance = request.guidance_scale
        effective_steps = request.num_steps
        is_flux_engine = isinstance(engine, FluxFillEngine)

        if is_flux_engine:
            # If user sent SD defaults (7.5 / 20), override with FLUX defaults
            if abs(request.guidance_scale - 7.5) < 0.01:
                effective_guidance = config.FLUX_DEFAULT_GUIDANCE_SCALE
            if request.num_steps == 20:
                effective_steps = config.FLUX_DEFAULT_NUM_INFERENCE_STEPS

        # Combine user's negative prompt with default for stronger effect (not used by FLUX)
        if request.negative_prompt and request.negative_prompt != config.DEFAULT_NEGATIVE_PROMPT:
            neg_prompt = request.negative_prompt + ", " + config.DEFAULT_NEGATIVE_PROMPT
        else:
            neg_prompt = config.DEFAULT_NEGATIVE_PROMPT

        # Параметры для кэширования. Раньше сюда не входили engine/prompt/
        # negative_prompt/invert_mask/fill_transparent и брались "сырые"
        # guidance/steps из запроса вместо реально применённых — при смене
        # ENGINE_TYPE или срабатывании FLUX-дефолтов кэш мог отдать чужой
        # результат под тем же ключом.
        params = {
            "engine": engine.name,
            "mode": request.mode,
            "prompt": request.prompt,
            "negative_prompt": neg_prompt,
            "strength": request.strength,
            "guidance_scale": effective_guidance,
            "num_steps": effective_steps,
            "controlnet_scale": request.controlnet_scale,
            "feather": request.feather,
            "expand": request.expand,
            "seed": request.seed,
            "invert_mask": request.invert_mask,
            "fill_transparent": request.fill_transparent,
            "crop_to_mask": request.crop_to_mask,
            "crop_padding": request.crop_padding if request.crop_to_mask else 0,
        }

        # Проверяем кэш. Кэш теперь хранит уже полностью готовый финальный
        # результат (после blend/paste_back/alpha, см. сохранение ниже), так
        # что на попадании просто отдаём его как есть — раньше здесь
        # ресайзился промежуточный (pre-paste_back/pre-alpha) результат, что
        # могло вернуть визуально неверную картинку.
        if config.CACHE_ENABLED and request.cache_dir:
            cache_dir = Path(request.cache_dir) / config.CACHE_DIR_NAME
            output_dir = Path(request.cache_dir) / config.OUTPUT_DIR_NAME
            cache_manager = CacheManager(cache_dir, output_dir)

            cached_result = cache_manager.get_cached_result(
                image, mask, request.prompt, params
            )
            if cached_result is not None:
                logger.info("Returning cached result")
                return InpaintResponse(
                    result=image_to_base64(cached_result),
                    cached=True,
                    width=cached_result.width,
                    height=cached_result.height,
                )

        logger.info(f"=== INPAINT PARAMS ===")
        logger.info(f"  mode: {mode}")
        logger.info(f"  prompt: '{request.prompt}'")
        logger.info(f"  negative_prompt: '{neg_prompt}'" if not is_flux_engine else "  negative_prompt: (ignored by FLUX)")
        logger.info(f"  strength: {request.strength}")
        logger.info(f"  guidance_scale: {effective_guidance}")
        logger.info(f"  steps: {effective_steps}")
        logger.info(f"  seed: {request.seed}")
        logger.info(f"  image size: {image.size}")
        logger.info(f"  mask size: {mask.size}")

        if config.DEBUG_SAVE_INTERMEDIATE:
            image.save("/tmp/ae_debug_model_input.png")
            mask.save("/tmp/ae_debug_model_mask.png")

        progress_info = {"step": 0, "total_steps": effective_steps, "stage": "inpainting"}

        def on_step(step, total):
            progress_info["step"] = step
            progress_info["total_steps"] = total
            if cancel_event.is_set():
                raise JobCancelled()

        inpaint_kwargs = dict(
            image=image,
            mask=mask,
            prompt=request.prompt,
            negative_prompt=neg_prompt,
            strength=request.strength,
            guidance_scale=effective_guidance,
            num_inference_steps=effective_steps,
            controlnet_scale=request.controlnet_scale,
            seed=request.seed,
            step_callback=on_step if mode == "ai" else None,
        )

        # PowerPaint: always use object_removal (P_ctxt) — fills masked area with
        # surrounding context. This is the correct mode for AE inpainting/removal.
        if mode == "ai" and isinstance(engine, PowerPaintEngine):
            inpaint_kwargs["task"] = "object_removal"

        # Инференс — в threadpool, чтобы event loop оставался живым:
        # /progress и /cancel продолжают отвечать во время генерации, и
        # cancel_event внутри on_step реально успевает сработать между шагами.
        result = await run_in_threadpool(lambda: engine.inpaint(**inpaint_kwargs))

        if config.DEBUG_SAVE_INTERMEDIATE:
            result.save("/tmp/ae_debug_model_output.png")

        # Защищаем пиксели за пределами маски (модель может их менять)
        if use_ai:
            blend_mask = mask  # уже resized к размеру модели
            if result.size != blend_mask.size:
                blend_mask = blend_mask.resize(result.size, Image.Resampling.LANCZOS)
            result = Image.composite(result, image, blend_mask)
            logger.info("Blended result with original outside mask")

        # Возвращаем к размеру после crop — через Real-ESRGAN если нужно увеличить
        if result.size != cropped_size:
            target_w, target_h = cropped_size
            result_w, result_h = result.size
            scale_needed = max(target_w / result_w, target_h / result_h)

            if scale_needed > 1.2:
                # Апскейлим через Real-ESRGAN для качества
                try:
                    if not upscale_engine.is_loaded():
                        logger.info("Auto-loading upscale engine...")
                        await run_in_threadpool(upscale_engine.load)

                    esrgan_scale = 4 if scale_needed > 2.5 else 2
                    progress_info = {"step": 0, "total_steps": 0, "stage": "upscaling"}
                    logger.info(f"Upscaling result: {result.size} x{esrgan_scale} (need {scale_needed:.1f}x)")
                    result = await run_in_threadpool(
                        lambda: upscale_engine.upscale(result, scale=esrgan_scale, model_type="anime")
                    )
                except Exception as e:
                    logger.warning(f"Upscale failed, using LANCZOS: {e}")

            # Точный ресайз до нужного размера
            if result.size != cropped_size:
                result = result.resize(cropped_size, Image.Resampling.LANCZOS)

        # Вставляем обратно в оригинал если был crop
        if crop_bbox:
            result = paste_back(original_image, result, crop_bbox, original_mask)
            logger.info(f"Pasted back to original: {result.size}")

        # Финальная проверка размера
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)

        # Восстанавливаем прозрачный фон если был
        if original_alpha is not None:
            import numpy as np
            result = result.convert("RGBA")
            if request.fill_transparent:
                # Fill transparent: в области маски ставим alpha=255 (непрозрачно),
                # за пределами маски — оригинальный alpha
                alpha_arr = np.array(original_alpha)
                mask_arr = np.array(original_mask)
                alpha_arr[mask_arr > 128] = 255
                result.putalpha(Image.fromarray(alpha_arr))
                logger.info("Restored alpha with fill_transparent (mask area forced opaque)")
            else:
                result.putalpha(original_alpha)
                logger.info("Restored alpha channel (transparent background)")

        if config.DEBUG_SAVE_INTERMEDIATE:
            result.save("/tmp/ae_debug_final_result.png")

        # Сохраняем в кэш ФИНАЛЬНЫЙ результат (после blend/paste_back/alpha).
        # Раньше кэшировался промежуточный результат до этих шагов, и
        # cache-hit возвращал его без paste_back/alpha-коррекции.
        if config.CACHE_ENABLED and cache_manager:
            cache_manager.save_to_cache(image, mask, result, request.prompt, params)

        progress_info = {"step": 0, "total_steps": 0, "stage": "done"}

        return InpaintResponse(
            result=image_to_base64(result),
            cached=False,
            width=result.width,
            height=result.height,
        )

      except JobCancelled:
        progress_info = {"step": 0, "total_steps": 0, "stage": "cancelled"}
        logger.info(f"Job {job_id} cancelled by user")
        raise HTTPException(status_code=499, detail="Cancelled by user")
      except HTTPException:
        progress_info = {"step": 0, "total_steps": 0, "stage": "error"}
        raise
      except Exception as e:
        progress_info = {"step": 0, "total_steps": 0, "stage": "error"}
        logger.error(f"Inpaint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
      finally:
        current_job.update({"id": None, "active": False, "cancel_event": None})


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale(request: UpscaleRequest):
    """Выполняет апскейл изображения"""
    if upscale_engine is None:
        raise HTTPException(status_code=500, detail="Upscale engine not initialized")

    if current_job["active"]:
        raise HTTPException(status_code=409, detail="Server is busy with another job. Stop it first or wait.")

    current_job.update({"id": uuid.uuid4().hex[:12], "active": True, "cancel_event": None})

    async with engine_lock:
      try:
        # Декодируем изображение
        logger.info(f"Received image base64 length: {len(request.image)}")
        image = base64_to_image(request.image)
        logger.info(f"Decoded image: {image.mode} {image.size}")

        # Проверяем параметры
        scale = request.scale if request.scale in [2, 4] else 4
        model_type = request.model_type if request.model_type in ["anime", "general"] else "anime"

        logger.info(f"Upscaling: {image.size} x{scale} with {model_type} model")

        # В threadpool — Real-ESRGAN не даёт прогресс по шагам (один
        # enhance() без callback), поэтому cancel здесь не прерывает уже
        # запущенный upscale, но хотя бы не блокирует event loop и не даёт
        # запуститься параллельному /inpaint поверх той же GPU-памяти.
        if not upscale_engine.is_loaded() or upscale_engine.current_model != model_type:
            await run_in_threadpool(upscale_engine.load, model_type)

        result = await run_in_threadpool(
            lambda: upscale_engine.upscale(image=ensure_rgb(image), scale=scale, model_type=model_type)
        )

        logger.info(f"Upscale completed: {result.size}")

        return UpscaleResponse(
            result=image_to_base64(result),
            width=result.width,
            height=result.height,
            scale=scale,
        )

      except HTTPException:
        raise
      except Exception as e:
        logger.error(f"Upscale failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
      finally:
        current_job.update({"id": None, "active": False, "cancel_event": None})


@app.post("/clear-cache")
async def clear_cache(cache_dir: str):
    """Очищает кэш проекта"""
    try:
        cache_path = Path(cache_dir) / config.CACHE_DIR_NAME
        output_path = Path(cache_dir) / config.OUTPUT_DIR_NAME

        cm = CacheManager(cache_path, output_path)
        cm.clear_cache()

        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
