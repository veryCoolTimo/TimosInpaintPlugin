"""
Тесты пайплайна /inpaint без скачивания моделей: AI-движки подменяются
заглушкой, OpenCV работает по-настоящему. Тест с настоящей LaMa помечен
slow и по умолчанию не запускается: python -m pytest tests -m slow

Запуск: cd server && python -m pytest tests -q
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

import main
import pipeline
from engines.base import BaseEngine
from utils import base64_to_image, image_to_base64


class StubEngine(BaseEngine):
    """
    Запоминает вход и возвращает:
      fill="red"    — сплошной красный (видно, что результат вклеен);
      fill="invert" — инвертированный вход: инверсия попиксельная, так что
                      по ней видно, что кроп вклеен ровно на своё место;
      fill="masked" — красный только под маской модели, иначе вход как есть;
      fill="flat"   — средний цвет входа (гладкая заливка без зерна).
    """

    def __init__(self, max_resolution=None, size_divisor=8, fill="red"):
        self.max_resolution = max_resolution
        self.size_divisor = size_divisor
        self.fill = fill
        self.calls = []
        self.masks = []

    name = "stub"
    supports_controlnet = False
    device = "cpu"

    def load(self):
        pass

    def unload(self):
        pass

    def is_loaded(self):
        return True

    def inpaint(self, image, mask, **kwargs):
        self.calls.append(image.size)
        self.masks.append(mask)
        if self.fill == "invert":
            return ImageOps.invert(image)
        if self.fill == "masked":
            red = Image.new("RGB", image.size, (255, 0, 0))
            return Image.composite(red, image, mask)
        if self.fill == "flat":
            mean = tuple(int(v) for v in np.array(image).reshape(-1, 3).mean(axis=0))
            return Image.new("RGB", image.size, mean)
        return Image.new("RGB", image.size, (255, 0, 0))


def noise_image(w, h, seed=0, mode="RGB"):
    rng = np.random.default_rng(seed)
    channels = 4 if mode == "RGBA" else 3
    arr = rng.integers(0, 256, (h, w, channels), dtype=np.uint8)
    if mode == "RGBA":
        arr[:, :, 3] = 255
    return Image.fromarray(arr, mode)


def rect_mask(w, h, box):
    arr = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = 255
    return Image.fromarray(arr, "L")


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def bboxes(monkeypatch):
    """Записывает bbox кропа, который выбрал пайплайн"""
    seen = []
    real = pipeline.prepare

    def spy(*args, **kwargs):
        p = real(*args, **kwargs)
        seen.append(p.bbox)
        return p

    monkeypatch.setattr(pipeline, "prepare", spy)
    return seen


def post(client, image, mask=None, **params):
    body = {"image": image_to_base64(image), "mask": image_to_base64(mask) if mask else ""}
    body.update(params)
    return client.post("/inpaint", json=body)


def result_of(r):
    assert r.status_code == 200, r.text
    return base64_to_image(r.json()["result"])


def assert_outside_untouched(original, result, mask):
    orig = np.array(original.convert("RGB")).astype(int)
    res = np.array(result.convert("RGB")).astype(int)
    outside = np.array(mask) == 0
    assert np.array_equal(orig[outside], res[outside]), "pixels outside the mask changed"


# --- разрешение и кроп ---------------------------------------------------

def test_small_mask_ai_runs_at_native_resolution(client, monkeypatch):
    stub = StubEngine(max_resolution=1024, size_divisor=32)
    monkeypatch.setattr(main, "ai_engine", stub)
    img = noise_image(1920, 1080)
    mask = rect_mask(1920, 1080, (900, 500, 1000, 600))

    result = result_of(post(client, img, mask, mode="ai"))

    assert result.size == (1920, 1080)
    w, h = stub.calls[0]
    assert w < 1024 and h < 1024  # кроп ~100px + контекст, без уменьшения
    assert_outside_untouched(img, result, mask)
    assert np.array(result)[550, 950].tolist() == [255, 0, 0]


def test_large_mask_still_cropped_and_outside_untouched(client, monkeypatch, bboxes):
    """Кроп на >70% кадра раньше отключался, и весь кадр мылился."""
    stub = StubEngine(max_resolution=1024, size_divisor=32)
    monkeypatch.setattr(main, "ai_engine", stub)
    img = noise_image(3840, 2160, seed=1)
    mask = rect_mask(3840, 2160, (900, 500, 2900, 1650))

    result = result_of(post(client, img, mask, mode="ai", crop_padding=16))

    assert bboxes[0] is not None and bboxes[0] != (0, 0, 3840, 2160)
    assert max(stub.calls[0]) <= 1024
    assert result.size == (3840, 2160)
    assert_outside_untouched(img, result, mask)


def test_crop_is_pasted_back_in_place(client, monkeypatch, bboxes):
    stub = StubEngine(max_resolution=None, fill="invert")
    monkeypatch.setattr(main, "lama_engine", stub)
    img = noise_image(1000, 700, seed=7)
    box = (300, 200, 700, 500)

    result = np.array(result_of(post(client, img, rect_mask(1000, 700, box), mode="remove", crop_padding=16)))

    assert bboxes[0] is not None
    x1, y1, x2, y2 = box
    assert np.array_equal(result[y1:y2, x1:x2], 255 - np.array(img)[y1:y2, x1:x2])
    assert_outside_untouched(img, Image.fromarray(result), rect_mask(1000, 700, box))


def test_non_divisible_frame_is_padded_not_stretched(client, monkeypatch):
    """1080 не делится на 32 — раньше кадр растягивался в 1056 и обратно."""
    stub = StubEngine(max_resolution=None, size_divisor=32, fill="invert")
    monkeypatch.setattr(main, "ai_engine", stub)
    img = noise_image(1000, 1080, seed=11)
    mask = rect_mask(1000, 1080, (50, 50, 950, 1030))

    result = np.array(result_of(post(client, img, mask, mode="ai")))

    w, h = stub.calls[0]
    assert w % 32 == 0 and h % 32 == 0 and w >= 1000 and h >= 1080
    assert np.array_equal(result[50:1030, 50:950], 255 - np.array(img)[50:1030, 50:950])


def test_lama_mode_does_not_downscale_to_512(client, monkeypatch):
    stub = StubEngine(max_resolution=2048)
    monkeypatch.setattr(main, "lama_engine", stub)
    img = noise_image(1600, 1200, seed=2)
    mask = rect_mask(1600, 1200, (400, 300, 1200, 900))

    result = result_of(post(client, img, mask, mode="remove"))

    w, h = stub.calls[0]
    assert w >= 800 and h >= 600
    assert_outside_untouched(img, result, mask)


def test_no_auto_esrgan(client, monkeypatch):
    monkeypatch.setattr(main, "ai_engine", StubEngine(max_resolution=256))

    def boom(*a, **k):
        raise AssertionError("ESRGAN must not run during inpaint")

    monkeypatch.setattr(main.upscale_engine, "load", boom)
    monkeypatch.setattr(main.upscale_engine, "upscale", boom)
    img = noise_image(1200, 800, seed=3)
    mask = rect_mask(1200, 800, (100, 100, 1100, 700))

    assert result_of(post(client, img, mask, mode="ai")).size == (1200, 800)


def test_tiny_image_on_flux_sized_engine(client, monkeypatch):
    monkeypatch.setattr(main, "ai_engine", StubEngine(max_resolution=1024, size_divisor=32))

    result = result_of(post(client, noise_image(20, 20, seed=10), rect_mask(20, 20, (5, 5, 15, 15)), mode="ai"))

    assert result.size == (20, 20)


def test_opencv_mode_real_engine(client):
    img = noise_image(640, 480, seed=4)
    mask = rect_mask(640, 480, (300, 200, 340, 240))

    result = result_of(post(client, img, mask, mode="clean"))

    assert result.size == (640, 480)
    assert_outside_untouched(img, result, mask)


# --- маска: тонкие штрихи и feather --------------------------------------

def test_thin_mask_survives_downscale(client, monkeypatch):
    """Провод в 3px на 4K при уменьшении до 512 раньше почти пропадал из маски."""
    stub = StubEngine(max_resolution=512, fill="masked")
    monkeypatch.setattr(main, "ai_engine", stub)
    img = Image.new("RGB", (3840, 2160), (100, 100, 100))
    mask = rect_mask(3840, 2160, (200, 1000, 3640, 1003))

    result = np.array(result_of(post(client, img, mask, mode="ai")))

    wire = result[1000:1003, 300:3500]
    assert wire[..., 0].min() > 200 and wire[..., 1].max() < 60


def test_feather_blends_without_seam_at_crop_edge(client, monkeypatch):
    """Хвост растушёвки не должен обрезаться краем кропа."""
    monkeypatch.setattr(main, "lama_engine", StubEngine(max_resolution=None, fill="invert"))
    img = noise_image(800, 600, seed=8)
    mask = rect_mask(800, 600, (300, 250, 500, 350))

    result = result_of(post(client, img, mask, mode="remove", feather=30, crop_padding=16))

    c = pipeline._grow_and_feather(np.array(mask), 0, 30).astype(np.float32)[..., None] / 255
    o = np.array(img).astype(np.float32)
    expected = np.rint(c * (255 - o) + (1 - c) * o).astype(np.uint8)
    assert np.array_equal(np.array(result), expected)


def test_feather_fully_replaces_mask_interior(client, monkeypatch):
    """Тонкая маска (10px) с feather 4: раньше центр получал только ~73%
    замены и удаляемый провод просвечивал."""
    monkeypatch.setattr(main, "lama_engine", StubEngine(max_resolution=None))
    img = noise_image(600, 400, seed=12)

    result = result_of(post(client, img, rect_mask(600, 400, (0, 195, 600, 205)), mode="remove", feather=4))

    assert (np.array(result)[195:205] == [255, 0, 0]).all()


def test_feather_tail_is_about_feather_px():
    """Растушёвка заканчивается примерно на feather px от маски, а не на 2×."""
    mask = np.array(rect_mask(400, 400, (150, 150, 250, 250)))
    comp = pipeline._grow_and_feather(mask, 0, 50)
    row = comp[200]
    assert row[249] == 255 and row[249 + 25] > 0 and row[249 + 51] == 0


def test_expand_grows_mask(client, monkeypatch):
    monkeypatch.setattr(main, "lama_engine", StubEngine(max_resolution=None))
    img = noise_image(300, 300, seed=13)

    result = np.array(result_of(post(client, img, rect_mask(300, 300, (100, 100, 200, 200)), mode="remove", expand=10)))

    assert result[150, 92].tolist() == [255, 0, 0]
    assert result[150, 80].tolist() == np.array(img)[150, 80].tolist()


# --- прозрачность --------------------------------------------------------

def rgba_with_band(w=800, h=600, seed=5):
    arr = np.array(noise_image(w, h, seed=seed, mode="RGBA"))
    arr[:, 600:, 3] = 0       # прозрачная полоса справа
    arr[:, 598:600, 3] = 128  # полупрозрачный сглаженный край контента
    return arr


def test_expand_from_alpha_is_opaque_and_keeps_soft_edge(client, monkeypatch, bboxes):
    monkeypatch.setattr(main, "ai_engine", StubEngine(max_resolution=None))
    arr = rgba_with_band()

    result = np.array(result_of(post(client, Image.fromarray(arr, "RGBA"), None, mode="ai")))

    assert bboxes[0] is None  # expand берёт весь кадр как контекст
    assert (result[..., 3] == 255).all()  # нет прозрачного ободка
    assert result[300, 700, :3].tolist() == [255, 0, 0]
    # полупрозрачный край: контент поверх дорисовки, a=128/255
    a = 128 / 255
    expected = np.rint(a * arr[300, 598, :3] + (1 - a) * np.array([255, 0, 0]))
    assert np.abs(result[300, 598, :3].astype(int) - expected).max() <= 1
    assert np.array_equal(result[:, :598], arr[:, :598])


def test_mask_with_alpha_no_fill_keeps_alpha_exactly(client, monkeypatch):
    monkeypatch.setattr(main, "ai_engine", StubEngine(max_resolution=None))
    arr = rgba_with_band()
    arr[:, :50, 3] = 77  # полупрозрачная область вне маски
    mask = rect_mask(800, 600, (100, 100, 300, 300))

    result = np.array(result_of(post(client, Image.fromarray(arr, "RGBA"), mask, mode="ai")))

    assert np.array_equal(result[..., 3], arr[..., 3])
    outside = np.array(mask) == 0
    # вне маски RGBA байт-в-байт, в т.ч. у полупрозрачных пикселей (раньше
    # они получали средний цвет заливки)
    assert np.array_equal(result[outside], arr[outside])
    assert result[200, 200, :3].tolist() == [255, 0, 0]


def test_fill_transparent_puts_generated_over(client, monkeypatch):
    monkeypatch.setattr(main, "ai_engine", StubEngine(max_resolution=None))
    arr = rgba_with_band()
    mask = rect_mask(800, 600, (550, 200, 750, 400))

    result = np.array(result_of(post(client, Image.fromarray(arr, "RGBA"), mask, mode="ai", fill_transparent=True, feather=10)))

    assert (result[200:400, 550:750, 3] == 255).all()
    assert result[300, 700].tolist() == [255, 0, 0, 255]
    # мягкий край над прозрачным: цвет — чистая генерация, без примеси
    # среднего цвета заливки (раньше был ореол)
    soft = result[300, 755]
    assert 0 < soft[3] < 255 and soft[:3].tolist() == [255, 0, 0]


# --- зерно ---------------------------------------------------------------

def grain_std(arr):
    """σ яркостного зерна (шум добавляется в яркость, как у настоящего зерна)"""
    import cv2
    f = arr.astype(np.float32)
    luma = f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114
    return float(np.std(luma - cv2.GaussianBlur(luma, (0, 0), 1.5)))


def test_regrain_after_downscale(client, monkeypatch):
    """4K, большая маска, движок 512: после растяжения область была гладкой
    (зерно 1.7 против 14 вокруг)."""
    monkeypatch.setattr(main, "ai_engine", StubEngine(max_resolution=512, fill="flat"))
    rng = np.random.default_rng(14)
    arr = np.clip(120 + rng.normal(0, 8, (2160, 3840, 3)), 0, 255).astype(np.uint8)
    mask = rect_mask(3840, 2160, (1000, 600, 2800, 1600))

    result = np.array(result_of(post(client, Image.fromarray(arr), mask, mode="ai")))

    inside = grain_std(result[800:1400, 1200:2600])
    outside = grain_std(arr[100:500, 100:900])
    assert 0.6 * outside < inside < 1.4 * outside


def test_no_regrain_on_flat_art(client, monkeypatch):
    """Плоская заливка (манхва) — шум добавлять нечего."""
    monkeypatch.setattr(main, "ai_engine", StubEngine(max_resolution=512, fill="flat"))
    img = Image.new("RGB", (3840, 2160), (200, 180, 160))
    mask = rect_mask(3840, 2160, (1000, 600, 2800, 1600))

    result = np.array(result_of(post(client, img, mask, mode="ai")))

    assert (result == [200, 180, 160]).all()


def test_no_regrain_next_to_hatching(client, monkeypatch):
    """Штриховка манхвы (период 9px) вокруг маски раньше оценивалась как
    зерно σ≈80, и внутрь маски ложилось цветное «конфетти»."""
    monkeypatch.setattr(main, "ai_engine", StubEngine(max_resolution=512, fill="flat"))
    x = np.arange(3840)
    stripes = np.where((x % 9) < 2, 30, 235).astype(np.uint8)
    arr = np.repeat(np.repeat(stripes[None, :, None], 2160, 0), 3, 2)
    mask = rect_mask(3840, 2160, (1000, 600, 2800, 1600))

    result = np.array(result_of(post(client, Image.fromarray(arr), mask, mode="ai")))

    assert grain_std(result[800:1400, 1200:2600]) < 0.5


def test_regrain_without_downscale_when_model_output_is_smooth(client, monkeypatch):
    """LaMa в родном разрешении тоже сглаживает — зерно доливается и без уменьшения."""
    monkeypatch.setattr(main, "lama_engine", StubEngine(max_resolution=None, fill="flat"))
    rng = np.random.default_rng(15)
    arr = np.clip(120 + rng.normal(0, 6, (1080, 1920, 3)), 0, 255).astype(np.uint8)
    mask = rect_mask(1920, 1080, (700, 300, 1200, 800))

    result = np.array(result_of(post(client, Image.fromarray(arr), mask, mode="remove")))

    inside = grain_std(result[400:700, 800:1100])
    outside = grain_std(arr[50:250, 50:600])
    assert 0.6 * outside < inside < 1.4 * outside
    # шум сенсора здесь независимый по каналам — цветной шум тоже совпадает
    chroma_in = np.std(np.diff(result[400:700, 800:1100].astype(int), axis=2))
    chroma_out = np.std(np.diff(arr[50:250, 50:600].astype(int), axis=2))
    assert 0.7 * chroma_out < chroma_in < 1.3 * chroma_out


def test_mono_grain_adds_no_color_noise(client, monkeypatch):
    """Плёночное (яркостное) зерно: внутри маски не должно быть цветного шума."""
    monkeypatch.setattr(main, "lama_engine", StubEngine(max_resolution=None, fill="flat"))
    rng = np.random.default_rng(16)
    mono = rng.normal(0, 6, (1080, 1920, 1))
    arr = np.clip(120 + np.repeat(mono, 3, axis=2), 0, 255).astype(np.uint8)
    mask = rect_mask(1920, 1080, (700, 300, 1200, 800))

    result = np.array(result_of(post(client, Image.fromarray(arr), mask, mode="remove")))

    patch = result[400:700, 800:1100].astype(int)
    assert grain_std(patch) > 0.6 * grain_std(arr[50:250, 50:600])
    assert np.std(patch[..., 0] - patch[..., 1]) < 1.0


def test_coarse_grain_size_matches(client, monkeypatch):
    """Крупное зерно (размытый шум) — доливаем такое же крупное, а не «песок»."""
    import cv2
    monkeypatch.setattr(main, "lama_engine", StubEngine(max_resolution=None, fill="flat"))
    rng = np.random.default_rng(17)
    g = cv2.GaussianBlur(rng.normal(0, 1, (1080, 1920)).astype(np.float32), (0, 0), 1.2)
    g *= 6 / g.std()
    arr = np.clip(120 + np.repeat(g[..., None], 3, axis=2), 0, 255).astype(np.uint8)
    mask = rect_mask(1920, 1080, (700, 300, 1200, 800))

    result = np.array(result_of(post(client, Image.fromarray(arr), mask, mode="remove")))

    def rho1(a):
        l = a[..., 1].astype(np.float32)
        l = l - cv2.GaussianBlur(l, (0, 0), 3.0)
        return np.corrcoef(l[:, :-1].ravel(), l[:, 1:].ravel())[0, 1]

    inside, outside = rho1(result[400:700, 800:1100]), rho1(arr[50:250, 50:600])
    assert abs(inside - outside) < 0.15
    # и по силе: крупное зерно раньше доливалось на ~40% слабее
    assert 0.8 < grain_std(result[400:700, 800:1100]) / grain_std(arr[50:250, 50:600]) < 1.25


# --- ошибки --------------------------------------------------------------

def test_expand_without_alpha_or_mask_is_400(client):
    assert post(client, noise_image(64, 64), None, mode="clean").status_code == 400


def test_empty_mask_is_400(client):
    r = post(client, noise_image(64, 64), rect_mask(64, 64, (0, 0, 0, 0)), mode="clean")
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_small_feathered_mask_is_not_rejected(client, monkeypatch):
    monkeypatch.setattr(main, "lama_engine", StubEngine())
    r = post(client, noise_image(400, 400, seed=9), rect_mask(400, 400, (190, 190, 210, 210)), mode="remove", feather=10)
    assert r.status_code == 200, r.text


def test_garbage_image_is_400(client):
    r = client.post("/inpaint", json={"image": "bm90IGFuIGltYWdl", "mask": "", "mode": "clean"})
    assert r.status_code == 400


def test_mask_size_mismatch_is_400(client):
    r = post(client, noise_image(64, 64), rect_mask(32, 32, (0, 0, 10, 10)), mode="clean")
    assert r.status_code == 400


# --- апскейл -------------------------------------------------------------

def test_upscale_keeps_alpha():
    from engines.upscale_engine import UpscaleEngine

    class FakeUpsampler:
        def enhance(self, img, outscale):
            import cv2
            h, w = img.shape[:2]
            return cv2.resize(img, (w * outscale, h * outscale)), None

    eng = UpscaleEngine(device="cpu")
    eng.upsampler = FakeUpsampler()
    eng.current_model = "anime"
    arr = np.array(noise_image(32, 16, mode="RGBA"))
    arr[:, :8, 3] = 0
    arr[:, :8, :3] = 0  # AE пишет прозрачное как чёрное
    out = eng.upscale(Image.fromarray(arr, "RGBA"), scale=4, model_type="anime")

    assert out.mode == "RGBA" and out.size == (128, 64)
    a = np.array(out)[:, :, 3]
    # LANCZOS даёт лёгкий ореол у края — проверяем вдали от перехода (x=32)
    assert (a[:, :16] == 0).all() and (a[:, 48:] == 255).all()
    # прозрачное залито средним цветом, а не чёрным — нет тёмной каймы
    assert np.array(out)[:, 28:36, :3].mean() > 60


# --- настоящая модель ----------------------------------------------------

@pytest.mark.slow
def test_lama_real_model_removes_object(client):
    """Настоящая LaMa на CPU (~10 с). Скачивает ~200MB при первом запуске."""
    pytest.importorskip("simple_lama_inpainting")
    from PIL import ImageDraw

    y, x = np.mgrid[0:768, 0:1024]
    arr = np.stack([x / 1024 * 200 + 30, y / 768 * 150 + 50, 120 + 40 * np.sin(x / 60)], -1).astype(np.uint8)
    img = Image.fromarray(arr)
    ImageDraw.Draw(img).ellipse((400, 300, 600, 450), fill=(250, 240, 20))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse((390, 290, 610, 460), fill=255)

    r = post(client, img, mask, mode="remove")
    if r.status_code == 500 and "load" in r.text.lower():
        pytest.skip(f"LaMa unavailable: {r.text}")
    result = result_of(r)

    assert_outside_untouched(img, result, mask)
    inside = np.array(result)[350:400, 450:550].astype(int)
    background = arr[350:400, 450:550].astype(int)
    assert np.abs(inside - background).mean() < 20  # жёлтого круга нет
