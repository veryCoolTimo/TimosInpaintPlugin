"""
Подготовка кадра и маски для движка и склейка результата с оригиналом.

Инварианты:
- модель получает кроп вокруг маски, уменьшенный только если он больше
  нативного размера движка, и дополненный (не растянутый) до кратности;
- результат вклеивается в полном разрешении по мягкой маске: пиксели, где
  маска 0, остаются байт-в-байт исходными;
- прозрачность слоя обрабатывается честным альфа-композитингом, а не
  подменой альфы порогом.
"""
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.image import get_mask_bbox, pad_to_multiple, resize_for_model

# Пиксели мягкой маски ярче этого порога модель перерисовывает целиком
MODEL_MASK_THRESHOLD = 16
# Минимальный контекст вокруг маски — доля от её большей стороны
CROP_CONTEXT_RATIO = 0.25
# Добавка зерна меньше этого σ не видна — не добавляем
REGRAIN_MIN_ADD = 0.3
# Выше этого σ "зерно" скорее всего текстура, а не шум плёнки/сенсора
REGRAIN_MAX_SIGMA = 12.0
# Корреляция остатка через несколько пикселей: у шума ~0, у штриховки/ткани
# заметно больше — тогда это текстура, и шум не добавляем
REGRAIN_MAX_CORR = 0.25


class InputError(ValueError):
    """Ошибка во входных данных — сервер отвечает 400"""


@dataclass
class Prepared:
    model_image: Image.Image          # RGB, кратен size_divisor
    model_mask: Image.Image           # L, 0/255, того же размера
    model_size: Tuple[int, int]       # размер до дополнения
    cropped_size: Tuple[int, int]     # размер кропа в исходном разрешении
    bbox: Optional[Tuple[int, int, int, int]]
    original_rgb: np.ndarray          # HxWx3 uint8, без заливки прозрачного
    original_alpha: Optional[np.ndarray]  # HxW uint8 или None
    comp: np.ndarray                  # HxW uint8, мягкая маска склейки
    expand_mode: bool                 # маска построена из прозрачности слоя
    fill_transparent: bool


def _grow_and_feather(mask: np.ndarray, expand_px: int, feather_px: int) -> np.ndarray:
    """
    Расширяет маску на expand_px и добавляет мягкий край наружу шириной
    feather_px. Внутренность маски не трогается (раньше feather размывал
    маску в обе стороны, и у тонкой маски центр заменялся не полностью).
    Через distance transform — один проход за ~десятки мс даже на 4K.
    """
    if expand_px <= 0 and feather_px <= 0:
        return mask

    outside = (mask <= 127).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5)

    out = mask.copy()
    if expand_px > 0:
        out[dist <= expand_px] = 255
        dist = np.maximum(dist - expand_px, 0)
    if feather_px > 0:
        t = np.clip(1.0 - dist / feather_px, 0.0, 1.0)
        soft = (t * t * (3 - 2 * t) * 255).astype(np.uint8)  # smoothstep
        out = np.maximum(out, soft)
    return out


def prepare(
    image: Image.Image,
    mask: Optional[Image.Image],
    *,
    max_resolution: Optional[int],
    size_divisor: int,
    invert_mask: bool = False,
    expand_px: int = 0,
    feather_px: int = 0,
    crop: bool = True,
    crop_padding: int = 128,
    fill_transparent: bool = False,
) -> Prepared:
    rgba = image.convert("RGBA") if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info else None
    original_rgb = np.array(image.convert("RGB"))
    original_alpha = np.array(rgba.getchannel("A")) if rgba is not None else None
    h, w = original_rgb.shape[:2]

    expand_mode = mask is None
    if expand_mode:
        # Режим expand: дорисовываем всё, что не полностью непрозрачно.
        # Полупрозрачные края контента потом ложатся поверх сгенерированного
        # (см. finish), поэтому по границе не остаётся прозрачного ободка.
        if original_alpha is None:
            raise InputError("Layer has no mask and no transparency — draw a mask first")
        # Порог 250, а не 255: слой с Opacity 99% или лёгким свечением
        # иначе целиком считался бы "дырой"
        mask_arr = np.where(original_alpha < 250, 255, 0).astype(np.uint8)
        fill_transparent = True
        crop = False  # дорисовке нужен весь кадр как контекст
    else:
        mask_arr = np.array(mask.convert("L"))
        if invert_mask:
            mask_arr = 255 - mask_arr
        if mask_arr.shape != (h, w):
            raise InputError(f"Mask size {mask_arr.shape[::-1]} doesn't match image size {(w, h)}")

    if not np.any(mask_arr > MODEL_MASK_THRESHOLD):
        raise InputError("Mask is empty — nothing to inpaint")

    comp = _grow_and_feather(mask_arr, expand_px, feather_px)
    model_mask_arr = np.where(comp > MODEL_MASK_THRESHOLD, 255, 0).astype(np.uint8)

    # Модель не должна видеть мусорный RGB полностью прозрачных пикселей
    # (AE пишет туда чёрный) — заливаем средним цветом видимого
    model_rgb = original_rgb.copy()
    if original_alpha is not None:
        hidden = original_alpha < 128
        visible = ~hidden
        fill = model_rgb[visible].mean(axis=0) if visible.any() else np.full(3, 128.0)
        model_rgb[hidden] = fill.astype(np.uint8)

    bbox = None
    if crop:
        bbox = get_mask_bbox(
            Image.fromarray(comp, "L"),
            padding=crop_padding,
            divisor=size_divisor,
            threshold=0,
            context_ratio=CROP_CONTEXT_RATIO,
        )
        if bbox == (0, 0, w, h):
            bbox = None

    model_image = Image.fromarray(model_rgb, "RGB")
    model_mask = Image.fromarray(model_mask_arr, "L")
    if bbox:
        model_image = model_image.crop(bbox)
        model_mask = model_mask.crop(bbox)

    cropped_size = model_image.size
    model_image = resize_for_model(model_image, max_size=max_resolution)
    if model_image.size != cropped_size:
        # Порог ниже середины: при уменьшении тонкие штрихи (провода,
        # волоски) размываются ниже 128 и иначе пропадали бы из маски
        resized = np.array(model_mask.resize(model_image.size, Image.Resampling.BILINEAR))
        model_mask = Image.fromarray(np.where(resized > MODEL_MASK_THRESHOLD, 255, 0).astype(np.uint8), "L")

    model_size = model_image.size
    model_image = pad_to_multiple(model_image, size_divisor)
    model_mask = pad_to_multiple(model_mask, size_divisor)

    return Prepared(
        model_image=model_image,
        model_mask=model_mask,
        model_size=model_size,
        cropped_size=cropped_size,
        bbox=bbox,
        original_rgb=original_rgb,
        original_alpha=original_alpha,
        comp=comp,
        expand_mode=expand_mode,
        fill_transparent=fill_transparent,
    )


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


# Остаток зерна меряется относительно локального среднего на этом масштабе
GRAIN_RESIDUAL_SIGMA = 3.0
# Статистика считается по участку не больше этого (стороны в px) — хватает
# для оценки, а на 4K-масках экономит ~секунду
GRAIN_STATS_MAX_SIDE = 1024


def _stats_window(sel: np.ndarray, margin: int = 0) -> Tuple[slice, slice]:
    """Окно ≤ GRAIN_STATS_MAX_SIDE вокруг центра выбранных пикселей"""
    ys, xs = np.nonzero(sel)
    h, w = sel.shape
    y1, y2 = max(0, ys.min() - margin), min(h, ys.max() + 1 + margin)
    x1, x2 = max(0, xs.min() - margin), min(w, xs.max() + 1 + margin)
    half = GRAIN_STATS_MAX_SIDE // 2
    if y2 - y1 > GRAIN_STATS_MAX_SIDE:
        cy = (y1 + y2) // 2
        y1, y2 = cy - half, cy + half
    if x2 - x1 > GRAIN_STATS_MAX_SIDE:
        cx = (x1 + x2) // 2
        x1, x2 = cx - half, cx + half
    return slice(y1, y2), slice(x1, x2)


def _grain_stats(rgb: np.ndarray, sel: np.ndarray, check_texture: bool = True) -> Optional[Tuple[np.ndarray, float]]:
    """
    Статистика зерна по выбранным пикселям: ковариация шума по RGB (3×3 —
    покрывает и яркостное плёночное зерно, и цветной шум сенсора) и
    автокорреляция на 1px (размер зерна).

    Берутся только плоские участки (мало деталей после размытия), выбросы
    отсекаются. Если остаток коррелирован через несколько пикселей — это
    текстура (штриховка, ткань, листва), а не зерно: None. Ровная заливка
    без шума — нулевая ковариация.
    """
    if sel.sum() < 500:
        return None
    luma = _luma(rgb)
    smooth = cv2.GaussianBlur(luma, (0, 0), 2.0)
    grad = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    flat = sel & (grad <= np.percentile(grad[sel], 30))
    if flat.sum() < 200:
        return None

    # Остаток относительно локального среднего на масштабе 3px: линейные
    # градиенты вычитаются точно, а крупное плёночное зерно ещё остаётся
    res = rgb - cv2.GaussianBlur(rgb, (0, 0), GRAIN_RESIDUAL_SIGMA)
    lres = _luma(res)
    r = lres[flat]
    sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    if sigma < 0.5:
        return np.zeros((3, 3), np.float32), 0.0
    if check_texture and sigma > REGRAIN_MAX_SIGMA:
        return None

    fine = luma - cv2.GaussianBlur(luma, (0, 0), 1.0)
    pairs = flat[:, :-3] & flat[:, 3:]
    if check_texture and pairs.sum() > 100:
        corr = np.corrcoef(fine[:, :-3][pairs], fine[:, 3:][pairs])[0, 1]
        if abs(corr) > REGRAIN_MAX_CORR:
            return None

    x = res[flat]
    med = np.median(x, axis=0)
    lim = 4 * 1.4826 * np.median(np.abs(x - med), axis=0) + 1e-3
    x = x[(np.abs(x - med) <= lim).all(axis=1)]
    if len(x) < 100:
        return None
    cov = np.cov(x.T).astype(np.float32)
    if not np.isfinite(cov).all():
        return None

    pairs = flat[:, :-1] & flat[:, 1:]
    rho1 = float(np.corrcoef(lres[:, :-1][pairs], lres[:, 1:][pairs])[0, 1]) if pairs.sum() > 100 else 0.0
    return cov, (rho1 if np.isfinite(rho1) else 0.0)


def _grain_field(shape: Tuple[int, int], cov: np.ndarray, rho1: float, seed: int) -> np.ndarray:
    """
    Шум HxWx3, у которого после того же фильтра, что и при измерении
    (вычитание локального среднего), ковариация равна cov, а размер зерна
    соответствует автокорреляции rho1.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((*shape, 3), dtype=np.float32)
    if 0.05 < rho1 < 0.95:
        # У белого шума после Гаусса σ_g автокорреляция на 1px = exp(-1/(4σ_g²))
        sigma_g = float(np.sqrt(-1.0 / (4.0 * np.log(rho1))))
        z = cv2.GaussianBlur(z, (0, 0), sigma_g)
    # Калибровка через фильтр измерения: крупное зерно им частично
    # срезается, и без этого доливалось на ~40% слабее, чем вокруг
    tile = z[:256, :256]
    measured = (tile - cv2.GaussianBlur(tile, (0, 0), GRAIN_RESIDUAL_SIGMA)).reshape(-1, 3).std(axis=0)
    z /= measured + 1e-6
    vals, vecs = np.linalg.eigh(cov)
    transform = vecs * np.sqrt(np.clip(vals, 0, None))
    return z @ transform.T


def _regrain(gen: np.ndarray, orig: np.ndarray, comp: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    """
    Доливает зерно туда, где модель выдала гладкую заливку (LaMa сглаживает,
    растянутый после уменьшения кроп — тем более). Добавляется ровно
    недостающее (ковариация окружения минус ковариация заливки) — без
    порогов-выключателей, чтобы на секвенции зерно не мигало от кадра к
    кадру. На плоской манхве и гладких градиентах окружение без шума — ничего
    не меняется. Всё считается в пределах кропа; gen меняется на месте.
    """
    region = comp > 0
    ring = comp == 0
    if alpha is not None:
        ring &= alpha == 255
    if not region.any():
        return gen

    win = _stats_window(region, margin=64)
    out_stats = _grain_stats(orig[win], ring[win])
    if out_stats is None:
        return gen
    cov_out, rho1 = out_stats
    if not cov_out.any():
        return gen
    inner = comp == 255
    in_stats = _grain_stats(gen[_stats_window(inner)], inner[_stats_window(inner)], check_texture=False) if inner.any() else None
    cov_in = in_stats[0] if in_stats is not None else np.zeros((3, 3), np.float32)

    vals, vecs = np.linalg.eigh(cov_out - cov_in)
    vals = np.clip(vals, 0, None)
    if float(np.sqrt(vals.max())) < REGRAIN_MIN_ADD:
        return gen
    cov_add = (vecs * vals) @ vecs.T

    # Seed от содержимого кадра: на секвенции рисунок шума меняется от
    # кадра к кадру, как настоящее зерно, а не стоит на месте
    seed = zlib.crc32(np.ascontiguousarray(orig[::7, ::7]).tobytes())
    noise = _grain_field(gen.shape[:2], cov_add, rho1, seed)

    # В зоне растушёвки смешиваются два независимых шума и дисперсия падает
    # в c²+(1−c)² раз — добавку там усиливаем, чтобы зерно не проседало
    w = np.array([0.299, 0.587, 0.114], np.float32)
    var_out, var_in, var_add = (float(w @ m @ w) for m in (cov_out, cov_in, cov_add))
    c = comp.astype(np.float32) / 255.0
    k = np.ones_like(c)
    soft = region & (comp < 255)
    if var_add > 0 and soft.any():
        cs = c[soft]
        k2 = ((2.0 / cs - 1.0) * var_out - var_in) / var_add
        k[soft] = np.sqrt(np.clip(k2, 1.0, 9.0))

    gen += noise * (k * region)[..., None]
    np.clip(gen, 0, 255, out=gen)
    return gen


def finish(p: Prepared, result: Image.Image) -> Image.Image:
    """Склеивает выход модели с оригиналом в полном разрешении"""
    result = result.convert("RGB")
    if result.size != p.model_image.size:
        result = result.resize(p.model_image.size, Image.Resampling.LANCZOS)
    result = result.crop((0, 0, *p.model_size))
    if result.size != p.cropped_size:
        result = result.resize(p.cropped_size, Image.Resampling.LANCZOS)

    # Всё считаем только в пределах кропа: за ним comp == 0 и пиксели
    # остаются исходными (на 4K с маленькой маской это сотни МБ разницы)
    h, w = p.comp.shape
    x1, y1, x2, y2 = p.bbox or (0, 0, w, h)
    win = (slice(y1, y2), slice(x1, x2))
    o = p.original_rgb[win].astype(np.float32)
    comp = p.comp[win]
    alpha = p.original_alpha[win] if p.original_alpha is not None else None

    gen = np.array(result, dtype=np.float32)
    gen = _regrain(gen, o, comp, alpha)
    c = (comp.astype(np.float32) / 255.0)[..., None]

    if alpha is None:
        out = p.original_rgb.copy()
        out[win] = np.rint(c * gen + (1 - c) * o).astype(np.uint8)
        return Image.fromarray(out, "RGB")

    a = (alpha.astype(np.float32) / 255.0)[..., None]
    if p.expand_mode:
        # Сгенерированный фон ПОД исходным слоем: полупрозрачные края
        # контента ложатся поверх дорисовки, итог полностью непрозрачный
        out_a = np.ones_like(a)
        rgb = a * o + (1 - a) * gen
    elif p.fill_transparent:
        # Сгенерированное ПОВЕРХ исходного (удаляем объект и заодно
        # закрываем прозрачное под маской)
        out_a = c + a * (1 - c)
        rgb = (c * gen + a * (1 - c) * o) / np.where(out_a > 0, out_a, 1.0)
    else:
        # Прозрачность слоя не меняем
        out_a = a
        rgb = c * gen + (1 - c) * o

    # Где итог полностью прозрачен, оставляем исходный RGB — так пиксели вне
    # маски совпадают с оригиналом байт-в-байт
    rgb = np.where(out_a > 0, rgb, o)
    out = np.dstack([p.original_rgb, p.original_alpha])
    out[win] = np.dstack([
        np.rint(np.clip(rgb, 0, 255)).astype(np.uint8),
        np.rint(np.clip(out_a[..., 0] * 255, 0, 255)).astype(np.uint8),
    ])
    return Image.fromarray(out, "RGBA")
