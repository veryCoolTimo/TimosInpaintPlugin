"""
Утилиты для работы с изображениями
"""
import base64
import io
from PIL import Image


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Конвертирует PIL Image в base64 строку"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """Конвертирует base64 строку в PIL Image"""
    # Убираем data:image/png;base64, если есть
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]

    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Убеждаемся что изображение в RGB"""
    if image.mode == "RGBA":
        # Создаём белый фон
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode != "RGB":
        return image.convert("RGB")
    return image


def ensure_mask_format(mask: Image.Image) -> Image.Image:
    """
    Приводит маску к правильному формату:
    - Grayscale (L mode)
    - Белый = область инпейнта
    - Чёрный = сохранить оригинал
    """
    if mask.mode == "RGBA":
        # Берём альфа-канал или конвертируем в grayscale
        mask = mask.convert("L")
    elif mask.mode != "L":
        mask = mask.convert("L")

    return mask


def resize_for_model(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Ресайз изображения для модели.
    SDXL работает лучше с размерами кратными 8.
    """
    w, h = image.size

    # Если уже в пределах — не трогаем
    if w <= max_size and h <= max_size:
        # Делаем размеры кратными 8
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image

    # Ресайз с сохранением пропорций
    ratio = min(max_size / w, max_size / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    # Делаем кратным 8
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def apply_mask_feather(mask: Image.Image, feather_px: int) -> Image.Image:
    """Применяет размытие к краям маски"""
    if feather_px <= 0:
        return mask

    from PIL import ImageFilter
    return mask.filter(ImageFilter.GaussianBlur(radius=feather_px))


def expand_mask(mask: Image.Image, expand_px: int) -> Image.Image:
    """Расширяет маску на указанное количество пикселей"""
    if expand_px <= 0:
        return mask

    from PIL import ImageFilter
    # Дилатация через maximum filter
    for _ in range(expand_px):
        mask = mask.filter(ImageFilter.MaxFilter(3))

    return mask
