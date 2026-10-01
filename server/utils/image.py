"""
Утилиты для работы с изображениями
"""
import base64
import io
from typing import Optional

from PIL import Image

# Локальный сервер, единственный клиент — CEP-панель, которая шлёт кадры из
# AE. Реальные кадры даже в 4K/8K укладываются в десятки МБ; лимит — просто
# защита от случайного/вредоносного decompression-bomb payload'а, а не
# ограничение на легитимный размер кадра.
MAX_BASE64_LENGTH = 100 * 1024 * 1024  # ~100MB base64 (~75MB decoded)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Конвертирует PIL Image в base64 строку"""
    buffer = io.BytesIO()
    # compress_level=1: на 4K кодирование с уровнем по умолчанию (6) занимало
    # ~1 с, а файл всё равно временный — живёт до импорта в AE
    image.save(buffer, format=format, compress_level=1)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """Конвертирует base64 строку в PIL Image"""
    # Убираем data:image/png;base64, если есть
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]

    if len(b64_string) > MAX_BASE64_LENGTH:
        raise ValueError(
            f"Image payload too large: {len(b64_string)} base64 chars "
            f"(limit {MAX_BASE64_LENGTH})"
        )

    image_data = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_data))
    # Force load to catch truncation errors early
    image.load()
    if image.mode in ("I", "I;16", "I;16B", "I;16L"):
        # 16-битный grayscale: convert("L") обрезал бы значения до 255
        # вместо масштабирования, и кадр/маска становились белыми
        import numpy as np
        image = Image.fromarray((np.array(image, dtype=np.uint32) >> 8).clip(0, 255).astype(np.uint8), "L")
    return image


def resize_for_model(image: Image.Image, max_size: Optional[int] = 1024) -> Image.Image:
    """
    Уменьшает изображение с сохранением пропорций, если большая сторона
    больше max_size. max_size=None — без ограничения. Кратность размеров
    обеспечивает pad_to_multiple — раньше здесь размеры округлялись вниз
    ресайзом, и кадр 1080px для FLUX (/32) растягивался в 1056px.
    """
    w, h = image.size
    if max_size is None or (w <= max_size and h <= max_size):
        return image

    ratio = min(max_size / w, max_size / h)
    new_size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def pad_to_multiple(image: Image.Image, divisor: int) -> Image.Image:
    """
    Дополняет изображение справа и снизу до размеров, кратных divisor.
    RGB дополняется повтором краевых пикселей (модель не видит резкой
    чёрной полосы), маска — нулями (дополнение не перерисовывается).
    Исходная область — image.crop((0, 0, *original_size)).
    """
    w, h = image.size
    pad_w = (-w) % divisor
    pad_h = (-h) % divisor
    if pad_w == 0 and pad_h == 0:
        return image

    import numpy as np

    arr = np.array(image)
    pad = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (arr.ndim - 2)
    if image.mode == "L":
        arr = np.pad(arr, pad, mode="constant")
    else:
        arr = np.pad(arr, pad, mode="edge")
    return Image.fromarray(arr, image.mode)


def get_mask_bbox(
    mask: Image.Image,
    padding: int = 64,
    divisor: int = 8,
    threshold: int = 128,
    context_ratio: float = 0.0,
) -> tuple:
    """
    Находит bounding box белой области маски с отступом.

    Args:
        mask: Маска в режиме L (grayscale)
        padding: Минимальный отступ вокруг маски в пикселях
        divisor: Размеры округляются до кратных этому числу (8 для SD, 32 для FLUX)
        threshold: Пиксели маски > threshold считаются частью маски
        context_ratio: Отступ не меньше этой доли от большей стороны маски —
            большой маске нужно больше контекста, чем фиксированные 128px

    Returns:
        tuple: (x1, y1, x2, y2) или None если маска пустая
    """
    import numpy as np

    mask_array = np.array(mask)

    white_pixels = np.where(mask_array > threshold)

    if len(white_pixels[0]) == 0:
        return None

    # Bounding box
    y_min, y_max = white_pixels[0].min(), white_pixels[0].max()
    x_min, x_max = white_pixels[1].min(), white_pixels[1].max()

    padding = max(padding, int(context_ratio * max(x_max - x_min + 1, y_max - y_min + 1)))

    # Добавляем padding
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(mask.width, x_max + padding)
    y2 = min(mask.height, y_max + padding)

    # Делаем размеры кратными divisor
    width = x2 - x1
    height = y2 - y1

    # Округляем вверх до кратного divisor
    new_width = ((width + divisor - 1) // divisor) * divisor
    new_height = ((height + divisor - 1) // divisor) * divisor

    # Расширяем bbox если нужно
    extra_w = new_width - width
    extra_h = new_height - height

    x1 = max(0, x1 - extra_w // 2)
    y1 = max(0, y1 - extra_h // 2)
    x2 = min(mask.width, x1 + new_width)
    y2 = min(mask.height, y1 + new_height)

    # Корректируем если вышли за границы
    if x2 - x1 < new_width:
        x1 = max(0, x2 - new_width)
    if y2 - y1 < new_height:
        y1 = max(0, y2 - new_height)

    return (x1, y1, x2, y2)
