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
    image = Image.open(io.BytesIO(image_data))
    # Force load to catch truncation errors early
    image.load()
    return image


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


def ensure_mask_format(mask: Image.Image, invert: bool = False) -> Image.Image:
    """
    Приводит маску к правильному формату:
    - Grayscale (L mode)
    - Белый = область инпейнта
    - Чёрный = сохранить оригинал

    Args:
        mask: Входная маска
        invert: Инвертировать маску (для AE где маска = что сохранить)
    """
    if mask.mode == "RGBA":
        mask = mask.convert("L")
    elif mask.mode != "L":
        mask = mask.convert("L")

    if invert:
        # Инвертируем: черный <-> белый
        import PIL.ImageOps
        mask = PIL.ImageOps.invert(mask)

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


def blank_masked_area(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Заменяет содержимое под маской средним цветом фона.
    Это заставляет модель генерировать на основе контекста снаружи маски,
    а не воспроизводить содержимое внутри.
    """
    import numpy as np

    img = np.array(image).copy()
    m = np.array(mask)

    # Unmasked pixels (background)
    bg_mask = m < 128
    if np.any(bg_mask):
        # Average color of background
        avg_color = img[bg_mask].mean(axis=0).astype(np.uint8)
    else:
        avg_color = np.array([128, 128, 128], dtype=np.uint8)

    # Fill masked area with average background color
    img[m >= 128] = avg_color

    return Image.fromarray(img)


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


def get_mask_bbox(mask: Image.Image, padding: int = 64) -> tuple:
    """
    Находит bounding box белой области маски с отступом.

    Args:
        mask: Маска в режиме L (grayscale)
        padding: Отступ вокруг маски в пикселях

    Returns:
        tuple: (x1, y1, x2, y2) или None если маска пустая
    """
    import numpy as np

    mask_array = np.array(mask)

    # Находим все белые пиксели (значение > 128)
    white_pixels = np.where(mask_array > 128)

    if len(white_pixels[0]) == 0:
        return None

    # Bounding box
    y_min, y_max = white_pixels[0].min(), white_pixels[0].max()
    x_min, x_max = white_pixels[1].min(), white_pixels[1].max()

    # Добавляем padding
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(mask.width, x_max + padding)
    y2 = min(mask.height, y_max + padding)

    # Делаем размеры кратными 8 для SD
    width = x2 - x1
    height = y2 - y1

    # Округляем вверх до кратного 8
    new_width = ((width + 7) // 8) * 8
    new_height = ((height + 7) // 8) * 8

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


def crop_to_mask(image: Image.Image, mask: Image.Image, padding: int = 64) -> tuple:
    """
    Обрезает изображение и маску по области маски.

    Args:
        image: Исходное изображение
        mask: Маска
        padding: Отступ вокруг маски

    Returns:
        tuple: (cropped_image, cropped_mask, bbox) или (image, mask, None) если маска слишком большая
    """
    bbox = get_mask_bbox(mask, padding)

    if bbox is None:
        return image, mask, None

    x1, y1, x2, y2 = bbox
    crop_width = x2 - x1
    crop_height = y2 - y1

    # Если crop область больше 70% от оригинала - не обрезаем
    original_area = image.width * image.height
    crop_area = crop_width * crop_height

    if crop_area > original_area * 0.7:
        return image, mask, None

    cropped_image = image.crop(bbox)
    cropped_mask = mask.crop(bbox)

    return cropped_image, cropped_mask, bbox


def paste_back(original: Image.Image, result: Image.Image, bbox: tuple, mask: Image.Image = None) -> Image.Image:
    """
    Вставляет результат обратно в оригинальное изображение.

    Args:
        original: Оригинальное изображение
        result: Результат инпейнта (обрезанный)
        bbox: Координаты обрезки (x1, y1, x2, y2)
        mask: Опциональная маска для плавного смешивания

    Returns:
        Image: Финальное изображение
    """
    if bbox is None:
        return result

    x1, y1, x2, y2 = bbox

    # Ресайзим результат если размеры не совпадают
    expected_size = (x2 - x1, y2 - y1)
    if result.size != expected_size:
        result = result.resize(expected_size, Image.Resampling.LANCZOS)

    # Создаём копию оригинала
    final = original.copy()

    if mask is not None:
        # Плавное смешивание по маске
        cropped_mask = mask.crop(bbox)
        if cropped_mask.size != result.size:
            cropped_mask = cropped_mask.resize(result.size, Image.Resampling.LANCZOS)
        final.paste(result, (x1, y1), cropped_mask)
    else:
        final.paste(result, (x1, y1))

    return final
