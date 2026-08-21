"""
Upscale Engine - Real-ESRGAN для апскейлинга аниме/манхвы
"""
import logging
import os
from pathlib import Path
from typing import Optional, Literal

import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class UpscaleEngine:
    """
    Апскейлер на основе Real-ESRGAN.
    Поддерживает модели anime (для манхвы) и general.
    """

    def __init__(self, device: Optional[str] = None):
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.upsampler = None
        self.current_model = None
        self.models_dir = Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)

        logger.info(f"UpscaleEngine initialized, device: {self.device}")

    @property
    def name(self) -> str:
        return "realesrgan"

    def is_loaded(self) -> bool:
        return self.upsampler is not None

    def _download_model(self, model_name: str) -> Path:
        """Скачивает модель если её нет"""
        import requests
        from tqdm import tqdm

        models = {
            'anime': {
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
                'filename': 'RealESRGAN_x4plus_anime_6B.pth'
            },
            'general': {
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                'filename': 'RealESRGAN_x4plus.pth'
            }
        }

        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")

        model_path = self.models_dir / models[model_name]['filename']

        if not model_path.exists():
            logger.info(f"Downloading model {models[model_name]['filename']}...")
            url = models[model_name]['url']
            # Скачиваем во временный файл и переименовываем только при
            # успехе — раньше частичная/битая загрузка (обрыв сети, 404 с
            # HTML-страницей ошибки) молча записывалась как валидный .pth,
            # и следующий запуск считал его валидным навсегда (проверялось
            # только model_path.exists()).
            tmp_path = model_path.with_suffix(model_path.suffix + ".part")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            try:
                with open(tmp_path, 'wb') as f:
                    for data in tqdm(response.iter_content(chunk_size=1024),
                                    total=total_size // 1024, unit='KB'):
                        f.write(data)
                tmp_path.rename(model_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            logger.info(f"Model downloaded: {model_path}")

        return model_path

    def load(self, model_type: Literal["anime", "general"] = "anime") -> None:
        """Загружает модель Real-ESRGAN"""
        if self.is_loaded() and self.current_model == model_type:
            logger.info(f"Model {model_type} already loaded")
            return

        # Выгружаем предыдущую модель
        if self.is_loaded():
            self.unload()

        logger.info(f"Loading Real-ESRGAN model: {model_type}")

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path = self._download_model(model_type)

        # Настройка архитектуры
        if model_type == 'anime':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                          num_block=6, num_grow_ch=32, scale=4)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                          num_block=23, num_grow_ch=32, scale=4)

        # Размер тайла
        tile_size = 256 if self.device == 'mps' else 512

        # Половинная точность
        half = self.device not in ['cpu', 'mps']

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=self.device
        )

        self.current_model = model_type
        logger.info(f"Model {model_type} loaded successfully")

    def unload(self) -> None:
        """Выгружает модель из памяти"""
        if self.upsampler is not None:
            del self.upsampler
            self.upsampler = None
            self.current_model = None

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Upscale model unloaded")

    def upscale(
        self,
        image: Image.Image,
        scale: Literal[2, 4] = 4,
        model_type: Literal["anime", "general"] = "anime"
    ) -> Image.Image:
        """
        Апскейлит изображение.

        Args:
            image: PIL Image для апскейла
            scale: Множитель масштаба (2 или 4)
            model_type: Тип модели (anime или general)

        Returns:
            PIL Image апскейленное изображение
        """
        # Загружаем модель если нужно
        if not self.is_loaded() or self.current_model != model_type:
            self.load(model_type)

        # Конвертируем PIL в numpy (BGR для OpenCV).
        # Раньше grayscale-вход (2D массив, нет image.shape[2]) падал с
        # IndexError — приводим explicitly к RGB/RGBA сначала.
        import cv2
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        img_np = np.array(image)
        if img_np.ndim == 2:  # на всякий случай, если convert выше не сработал
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        logger.info(f"Upscaling: {image.size} x{scale} with {model_type} model")

        # Апскейлим
        # Real-ESRGAN всегда делает x4, для x2 нужно потом уменьшить
        output, _ = self.upsampler.enhance(img_np, outscale=scale)

        # Конвертируем обратно в PIL (RGB)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(output_rgb)

        logger.info(f"Upscale completed: {result.size}")

        return result
