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
from starlette.concurrency import run_in_threadpool

import config
from engines import DiffusersEngine, LamaEngine, UpscaleEngine, PowerPaintEngine, FluxFillEngine
from engines.opencv_engine import OpenCVEngine
from engines.base import BaseEngine
import pipeline
from utils import base64_to_image, image_to_base64


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

# Progress tracking
progress_info = {
    "step": 0,
    "total_steps": 0,
    "stage": "idle",  # idle, loading_model, inpainting, done, error, cancelled
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
    global ai_engine, lama_engine, opencv_engine, upscale_engine

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


class InpaintResponse(BaseModel):
    """Ответ с результатом инпейнтинга"""
    result: str = Field(..., description="Base64 PNG результата")
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
    global progress_info

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

        try:
            image = await run_in_threadpool(base64_to_image, request.image)
            mask = await run_in_threadpool(base64_to_image, request.mask) if request.mask else None
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Can't decode image/mask: {e}")
        logger.info(f"Decoded image: {image.mode} {image.size}, mask: {mask.size if mask else 'from alpha'}")

        try:
            prepared = await run_in_threadpool(
                lambda: pipeline.prepare(
                    image,
                    mask,
                    max_resolution=engine.max_resolution,
                    size_divisor=engine.size_divisor,
                    invert_mask=request.invert_mask,
                    expand_px=request.expand,
                    feather_px=request.feather,
                    crop=request.crop_to_mask,
                    crop_padding=request.crop_padding,
                    fill_transparent=request.fill_transparent,
                )
            )
        except pipeline.InputError as e:
            raise HTTPException(status_code=400, detail=str(e))

        logger.info(
            f"Prepared: crop={prepared.bbox} {prepared.cropped_size} -> model {prepared.model_size} "
            f"(padded {prepared.model_image.size}), expand_mode={prepared.expand_mode}"
        )

        effective_guidance = request.guidance_scale
        effective_steps = request.num_steps
        if isinstance(engine, FluxFillEngine):
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

        logger.info(
            f"Inpaint: mode={mode} engine={engine.name} prompt={request.prompt!r} "
            f"guidance={effective_guidance} steps={effective_steps} seed={request.seed}"
        )

        if config.DEBUG_SAVE_INTERMEDIATE:
            prepared.model_image.save("/tmp/ae_debug_model_input.png")
            prepared.model_mask.save("/tmp/ae_debug_model_mask.png")

        progress_info = {"step": 0, "total_steps": effective_steps, "stage": "inpainting"}

        def on_step(step, total):
            progress_info["step"] = step
            progress_info["total_steps"] = total
            if cancel_event.is_set():
                raise JobCancelled()

        inpaint_kwargs = dict(
            image=prepared.model_image,
            mask=prepared.model_mask,
            prompt=request.prompt,
            negative_prompt=neg_prompt,
            strength=request.strength,
            guidance_scale=effective_guidance,
            num_inference_steps=effective_steps,
            controlnet_scale=request.controlnet_scale,
            seed=request.seed,
            step_callback=on_step if mode == "ai" else None,
        )

        # PowerPaint: без промпта — object_removal (заполнить контекстом), с
        # промптом — text_guided (раньше промпт молча игнорировался), для
        # expand — image_outpainting
        if mode == "ai" and isinstance(engine, PowerPaintEngine):
            if prepared.expand_mode:
                inpaint_kwargs["task"] = "image_outpainting"
            else:
                inpaint_kwargs["task"] = "text_guided" if request.prompt.strip() else "object_removal"

        # Инференс — в threadpool, чтобы event loop оставался живым:
        # /progress и /cancel продолжают отвечать во время генерации, и
        # cancel_event внутри on_step реально успевает сработать между шагами.
        result = await run_in_threadpool(lambda: engine.inpaint(**inpaint_kwargs))

        if config.DEBUG_SAVE_INTERMEDIATE:
            result.save("/tmp/ae_debug_model_output.png")

        result = await run_in_threadpool(lambda: pipeline.finish(prepared, result))
        logger.info(f"Composited onto original: {result.mode} {result.size}")

        progress_info = {"step": 0, "total_steps": 0, "stage": "done"}

        encoded = await run_in_threadpool(image_to_base64, result)
        return InpaintResponse(
            result=encoded,
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
            lambda: upscale_engine.upscale(image=image, scale=scale, model_type=model_type)
        )

        logger.info(f"Upscale completed: {result.size}")

        encoded = await run_in_threadpool(image_to_base64, result)
        return UpscaleResponse(
            result=encoded,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
