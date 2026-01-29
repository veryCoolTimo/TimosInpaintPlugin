"""
FastAPI сервер для инпейнтинга
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

import config
from engines import DiffusersEngine, UpscaleEngine
from engines.opencv_engine import OpenCVEngine
from utils import base64_to_image, image_to_base64, CacheManager
from utils.image import (
    ensure_rgb,
    ensure_mask_format,
    resize_for_model,
    apply_mask_feather,
    expand_mask,
    crop_to_mask,
    paste_back,
    blank_masked_area,
)

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Глобальные объекты
ai_engine: Optional[DiffusersEngine] = None
opencv_engine: Optional[OpenCVEngine] = None
upscale_engine: Optional[UpscaleEngine] = None
cache_manager: Optional[CacheManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: загрузка/выгрузка модели"""
    global ai_engine, opencv_engine, upscale_engine, cache_manager

    logger.info("Starting server...")

    # Инициализируем движки
    ai_engine = DiffusersEngine(
        model_id=config.SDXL_INPAINT_MODEL,
        controlnet_id=config.CONTROLNET_MODEL if hasattr(config, 'CONTROLNET_MODEL') else None,
    )

    opencv_engine = OpenCVEngine(method="telea")
    opencv_engine.load()  # OpenCV doesn't need heavy loading

    upscale_engine = UpscaleEngine()
    # Upscale engine loads lazily on first request

    logger.info("Server started")
    yield

    # Cleanup
    if ai_engine and ai_engine.is_loaded():
        ai_engine.unload()
    if upscale_engine and upscale_engine.is_loaded():
        upscale_engine.unload()

    logger.info("Server stopped")


app = FastAPI(
    title="AE Inpaint Server",
    description="Локальный сервер инпейнтинга для After Effects плагина",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS для CEP панели
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Модели запросов ===

class InpaintRequest(BaseModel):
    """Запрос на инпейнтинг"""
    image: str = Field(..., description="Base64 PNG изображения")
    mask: str = Field(default="", description="Base64 PNG маски (белый = inpaint). Пусто = авто из альфа")
    mode: str = Field(default="ai", description="Режим: 'ai' (SD) или 'clean' (OpenCV)")
    prompt: str = Field(default="", description="Текстовый промпт")
    negative_prompt: str = Field(default="", description="Негативный промпт")
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_steps: int = Field(default=30, ge=10, le=100)
    controlnet_scale: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: Optional[int] = Field(default=None)
    feather: int = Field(default=0, ge=0, le=50, description="Feather маски в px")
    expand: int = Field(default=0, ge=0, le=50, description="Expand маски в px")
    crop_to_mask: bool = Field(default=True, description="Обрезать по маске для ускорения")
    crop_padding: int = Field(default=128, ge=16, le=512, description="Отступ при обрезке по маске")
    invert_mask: bool = Field(default=False, description="Инвертировать маску")
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


# === Эндпоинты ===

def is_model_cached() -> bool:
    """Проверяет есть ли модель в кэше HuggingFace"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_name = config.SDXL_INPAINT_MODEL.replace("/", "--")
    model_dir = cache_dir / f"models--{model_name}"
    return model_dir.exists()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервера"""
    return HealthResponse(
        status="ok",
        engine=ai_engine.name if ai_engine else "none",
        engine_loaded=ai_engine.is_loaded() if ai_engine else False,
        device=ai_engine.device if ai_engine else "unknown",
        model_cached=is_model_cached(),
    )


@app.post("/load")
async def load_model():
    """Загружает AI модель в память"""
    if ai_engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    if ai_engine.is_loaded():
        return {"status": "already_loaded"}

    try:
        ai_engine.load()
        return {"status": "loaded"}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_model():
    """Выгружает AI модель из памяти"""
    if ai_engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    ai_engine.unload()
    return {"status": "unloaded"}


@app.post("/inpaint", response_model=InpaintResponse)
async def inpaint(request: InpaintRequest):
    """Выполняет инпейнтинг"""
    global cache_manager

    # Выбираем движок по режиму
    use_opencv = request.mode == "clean"
    engine = opencv_engine if use_opencv else ai_engine

    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    # Автозагрузка AI модели при первом запросе
    if not use_opencv and not engine.is_loaded():
        logger.info("Auto-loading AI model on first request...")
        try:
            engine.load()
        except Exception as e:
            logger.error(f"Failed to auto-load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    try:
        # Debug: log received data sizes
        logger.info(f"Received image base64 length: {len(request.image)}")
        logger.info(f"Received mask base64 length: {len(request.mask)}")

        # Декодируем изображения
        image = base64_to_image(request.image)
        logger.info(f"Decoded image: {image.mode} {image.size}")

        if request.mask:
            mask = base64_to_image(request.mask)
            logger.info(f"Decoded mask: {mask.mode} {mask.size}")
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

        # Обрезаем по маске для ускорения (только для AI режима)
        if request.crop_to_mask and not use_opencv:
            image, mask, crop_bbox = crop_to_mask(image, mask, padding=request.crop_padding)
            if crop_bbox:
                logger.info(f"Cropped to mask: {original_size} -> {image.size} (bbox: {crop_bbox})")

        # Ресайз только для AI модели (OpenCV работает с оригинальным размером)
        cropped_size = image.size  # Размер после crop (или оригинал если не обрезали)
        if not use_opencv:
            image = resize_for_model(image)
            mask = mask.resize(image.size)

        # Параметры для кэширования
        params = {
            "mode": request.mode,
            "strength": request.strength,
            "guidance_scale": request.guidance_scale,
            "num_steps": request.num_steps,
            "controlnet_scale": request.controlnet_scale,
            "feather": request.feather,
            "expand": request.expand,
            "seed": request.seed,
            "crop_to_mask": request.crop_to_mask,
            "crop_padding": request.crop_padding if request.crop_to_mask else 0,
        }

        # Проверяем кэш
        if config.CACHE_ENABLED and request.cache_dir:
            cache_dir = Path(request.cache_dir) / config.CACHE_DIR_NAME
            output_dir = Path(request.cache_dir) / config.OUTPUT_DIR_NAME
            cache_manager = CacheManager(cache_dir, output_dir)

            cached_result = cache_manager.get_cached_result(
                image, mask, request.prompt, params
            )
            if cached_result is not None:
                logger.info("Returning cached result")
                # Возвращаем к оригинальному размеру
                if cached_result.size != original_size:
                    cached_result = cached_result.resize(original_size)

                return InpaintResponse(
                    result=image_to_base64(cached_result),
                    cached=True,
                    width=cached_result.width,
                    height=cached_result.height,
                )

        # Выполняем инпейнтинг
        result = engine.inpaint(
            image=image,
            mask=mask,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or config.DEFAULT_NEGATIVE_PROMPT,
            strength=request.strength,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_steps,
            controlnet_scale=request.controlnet_scale,
            seed=request.seed,
        )

        # Сохраняем в кэш
        if config.CACHE_ENABLED and cache_manager:
            cache_manager.save_to_cache(
                image, mask, result, request.prompt, params
            )

        # Возвращаем к размеру после crop (до model resize)
        if result.size != cropped_size:
            result = result.resize(cropped_size, Image.Resampling.LANCZOS)

        # Вставляем обратно в оригинал если был crop
        if crop_bbox:
            result = paste_back(original_image, result, crop_bbox, original_mask)
            logger.info(f"Pasted back to original: {result.size}")

        # Финальная проверка размера
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)

        return InpaintResponse(
            result=image_to_base64(result),
            cached=False,
            width=result.width,
            height=result.height,
        )

    except Exception as e:
        logger.error(f"Inpaint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale(request: UpscaleRequest):
    """Выполняет апскейл изображения"""
    if upscale_engine is None:
        raise HTTPException(status_code=500, detail="Upscale engine not initialized")

    try:
        # Декодируем изображение
        logger.info(f"Received image base64 length: {len(request.image)}")
        image = base64_to_image(request.image)
        logger.info(f"Decoded image: {image.mode} {image.size}")

        # Проверяем параметры
        scale = request.scale if request.scale in [2, 4] else 4
        model_type = request.model_type if request.model_type in ["anime", "general"] else "anime"

        logger.info(f"Upscaling: {image.size} x{scale} with {model_type} model")

        # Выполняем апскейл
        result = upscale_engine.upscale(
            image=ensure_rgb(image),
            scale=scale,
            model_type=model_type,
        )

        logger.info(f"Upscale completed: {result.size}")

        return UpscaleResponse(
            result=image_to_base64(result),
            width=result.width,
            height=result.height,
            scale=scale,
        )

    except Exception as e:
        logger.error(f"Upscale failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
