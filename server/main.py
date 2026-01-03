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

import config
from engines import DiffusersEngine
from utils import base64_to_image, image_to_base64, CacheManager
from utils.image import (
    ensure_rgb,
    ensure_mask_format,
    resize_for_model,
    apply_mask_feather,
    expand_mask,
)

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Глобальные объекты
engine: Optional[DiffusersEngine] = None
cache_manager: Optional[CacheManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: загрузка/выгрузка модели"""
    global engine, cache_manager

    logger.info("Starting server...")

    # Инициализируем движок
    engine = DiffusersEngine(
        model_id=config.SDXL_INPAINT_MODEL,
        controlnet_id=config.CONTROLNET_MODEL if hasattr(config, 'CONTROLNET_MODEL') else None,
    )

    # Предзагрузка модели (опционально, можно отложить)
    # engine.load()

    logger.info("Server started")
    yield

    # Cleanup
    if engine and engine.is_loaded():
        engine.unload()

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
    mask: str = Field(..., description="Base64 PNG маски (белый = inpaint)")
    prompt: str = Field(default="", description="Текстовый промпт")
    negative_prompt: str = Field(default="", description="Негативный промпт")
    strength: float = Field(default=0.85, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_steps: int = Field(default=30, ge=10, le=100)
    controlnet_scale: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: Optional[int] = Field(default=None)
    feather: int = Field(default=0, ge=0, le=50, description="Feather маски в px")
    expand: int = Field(default=0, ge=0, le=50, description="Expand маски в px")
    cache_dir: Optional[str] = Field(default=None, description="Путь к папке кэша проекта")


class InpaintResponse(BaseModel):
    """Ответ с результатом инпейнтинга"""
    result: str = Field(..., description="Base64 PNG результата")
    cached: bool = Field(default=False, description="Результат из кэша")
    width: int
    height: int


class HealthResponse(BaseModel):
    """Статус сервера"""
    status: str
    engine: str
    engine_loaded: bool
    device: str


# === Эндпоинты ===

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервера"""
    return HealthResponse(
        status="ok",
        engine=engine.name if engine else "none",
        engine_loaded=engine.is_loaded() if engine else False,
        device=engine.device if engine else "unknown",
    )


@app.post("/load")
async def load_model():
    """Загружает модель в память"""
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    if engine.is_loaded():
        return {"status": "already_loaded"}

    try:
        engine.load()
        return {"status": "loaded"}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_model():
    """Выгружает модель из памяти"""
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    engine.unload()
    return {"status": "unloaded"}


@app.post("/inpaint", response_model=InpaintResponse)
async def inpaint(request: InpaintRequest):
    """Выполняет инпейнтинг"""
    global cache_manager

    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    # Автозагрузка модели при первом запросе
    if not engine.is_loaded():
        logger.info("Auto-loading model on first request...")
        try:
            engine.load()
        except Exception as e:
            logger.error(f"Failed to auto-load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    try:
        # Декодируем изображения
        image = base64_to_image(request.image)
        mask = base64_to_image(request.mask)

        # Подготавливаем изображения
        image = ensure_rgb(image)
        mask = ensure_mask_format(mask)

        # Применяем feather/expand к маске
        if request.feather > 0:
            mask = apply_mask_feather(mask, request.feather)
        if request.expand > 0:
            mask = expand_mask(mask, request.expand)

        # Ресайз для модели
        original_size = image.size
        image = resize_for_model(image)
        mask = mask.resize(image.size)

        # Параметры для кэширования
        params = {
            "strength": request.strength,
            "guidance_scale": request.guidance_scale,
            "num_steps": request.num_steps,
            "controlnet_scale": request.controlnet_scale,
            "feather": request.feather,
            "expand": request.expand,
            "seed": request.seed,
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

        # Возвращаем к оригинальному размеру
        if result.size != original_size:
            result = result.resize(original_size)

        return InpaintResponse(
            result=image_to_base64(result),
            cached=False,
            width=result.width,
            height=result.height,
        )

    except Exception as e:
        logger.error(f"Inpaint failed: {e}", exc_info=True)
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
