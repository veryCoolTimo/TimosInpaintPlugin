"""
Кэширование результатов инпейнтинга
"""
import hashlib
import json
from pathlib import Path
from typing import Optional
from PIL import Image

from .image import image_to_base64


class CacheManager:
    """Менеджер кэша для инпейнтинга"""

    def __init__(self, cache_dir: Path, output_dir: Path):
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _compute_hash(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        params: dict
    ) -> str:
        """Вычисляет hash от входных данных"""
        hasher = hashlib.md5()

        # Hash изображения
        hasher.update(image.tobytes())

        # Hash маски
        hasher.update(mask.tobytes())

        # Hash промпта
        hasher.update(prompt.encode("utf-8"))

        # Hash параметров
        params_str = json.dumps(params, sort_keys=True)
        hasher.update(params_str.encode("utf-8"))

        return hasher.hexdigest()[:16]

    def get_cache_key(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        params: dict,
        prefix: str = "inpaint"
    ) -> str:
        """Генерирует ключ кэша"""
        hash_str = self._compute_hash(image, mask, prompt, params)
        return f"{prefix}_{hash_str}"

    def get_cached_result(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        params: dict,
        prefix: str = "inpaint"
    ) -> Optional[Image.Image]:
        """Возвращает закэшированный результат или None"""
        cache_key = self.get_cache_key(image, mask, prompt, params, prefix)
        result_path = self.output_dir / f"{cache_key}_result.png"

        if result_path.exists():
            return Image.open(result_path)

        return None

    def save_to_cache(
        self,
        image: Image.Image,
        mask: Image.Image,
        result: Image.Image,
        prompt: str,
        params: dict,
        prefix: str = "inpaint"
    ) -> Path:
        """Сохраняет результат в кэш"""
        cache_key = self.get_cache_key(image, mask, prompt, params, prefix)

        # Сохраняем входные данные (для отладки)
        input_path = self.cache_dir / f"{cache_key}_input.png"
        mask_path = self.cache_dir / f"{cache_key}_mask.png"
        result_path = self.output_dir / f"{cache_key}_result.png"

        image.save(input_path)
        mask.save(mask_path)
        result.save(result_path)

        # Сохраняем метаданные
        meta_path = self.cache_dir / f"{cache_key}_meta.json"
        meta = {
            "prompt": prompt,
            "params": params,
            "input": str(input_path),
            "mask": str(mask_path),
            "result": str(result_path)
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return result_path

    def clear_cache(self):
        """Очищает весь кэш"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
