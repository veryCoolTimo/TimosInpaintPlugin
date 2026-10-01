#!/usr/bin/env python
"""
Проверка AI Gen движка без After Effects: загрузка, инпейнт, время, память.

    .venv/bin/python scripts/smoke_ai_engine.py                    # klein, тестовая картинка
    .venv/bin/python scripts/smoke_ai_engine.py --image frame.png --mask mask.png --prompt "brick wall"
    .venv/bin/python scripts/smoke_ai_engine.py --engine flux      # сравнить с FLUX.1 Fill

Результат и полная склейка сохраняются в ./smoke_out/. Картинка идёт через
тот же пайплайн, что и в сервере (кроп вокруг маски, склейка в полном
разрешении), так что время близко к реальному /inpaint.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import config  # noqa: E402
import pipeline  # noqa: E402


def make_engine(name):
    if name == "klein":
        from engines import Flux2KleinEngine
        return Flux2KleinEngine(model_id=config.KLEIN_MODEL)
    if name == "flux":
        from engines import FluxFillEngine
        return FluxFillEngine(
            gguf_repo=config.FLUX_GGUF_REPO,
            gguf_filename=config.FLUX_GGUF_FILENAME,
            base_model=config.FLUX_BASE_MODEL,
        )
    raise SystemExit(f"unknown engine {name}")


def test_scene():
    """1920×1080: градиент + текстура, жёлтый круг под маской"""
    y, x = np.mgrid[0:1080, 0:1920]
    arr = np.stack([x / 1920 * 180 + 40, y / 1080 * 120 + 60, 120 + 40 * np.sin(x / 50) * np.cos(y / 40)], -1)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    ImageDraw.Draw(img).ellipse((800, 380, 1120, 700), fill=(250, 230, 20))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse((780, 360, 1140, 720), fill=255)
    return img, mask


def memory_gb():
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1e9
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="klein", choices=["klein", "flux"])
    ap.add_argument("--image")
    ap.add_argument("--mask", help="белое = перерисовать")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=2, help="первый прогон включает прогрев")
    args = ap.parse_args()

    if args.image:
        img = Image.open(args.image)
        mask = Image.open(args.mask) if args.mask else None
    else:
        img, mask = test_scene()

    if args.mask and not args.image:
        raise SystemExit("--mask needs --image")

    engine = make_engine(args.engine)
    print(f"engine={engine.name} device={engine.device} torch={torch.__version__}")

    # Маску проверяем до загрузки модели (16 ГБ, минуты)
    try:
        p = pipeline.prepare(img, mask, max_resolution=engine.max_resolution, size_divisor=engine.size_divisor)
    except pipeline.InputError as e:
        raise SystemExit(f"bad input: {e}")
    print(f"crop {p.bbox} {p.cropped_size} -> model {p.model_image.size}")

    t = time.time()
    engine.load()
    print(f"load: {time.time() - t:.1f}s, memory {memory_gb():.1f} GB")

    out_dir = Path("smoke_out")
    out_dir.mkdir(exist_ok=True)
    for run in range(args.runs):
        steps = []
        t = time.time()
        result = engine.inpaint(
            p.model_image, p.model_mask, prompt=args.prompt, seed=args.seed,
            num_inference_steps=args.steps, step_callback=lambda s, n: steps.append(time.time()),
        )
        dt = time.time() - t
        per_step = np.diff([t] + steps).mean() if steps else float("nan")
        print(f"run {run + 1}: {dt:.1f}s total, {len(steps)} steps, ~{per_step:.1f}s/step, memory {memory_gb():.1f} GB")

    result.save(out_dir / f"{engine.name}_model_output.png")
    pipeline.finish(p, result).save(out_dir / f"{engine.name}_final.png")
    print(f"saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
