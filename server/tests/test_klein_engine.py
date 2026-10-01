"""
FLUX.2 klein без скачивания 16 ГБ весов: пайплайн собирается из крошечных
случайных компонентов (как в тестах самого diffusers). Проверяется
интеграция — размеры, шаги, отмена, склейка через /inpaint — а не качество.

Нужен diffusers>=0.38 (Flux2KleinInpaintPipeline); иначе тесты пропускаются.
"""
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "Flux2KleinInpaintPipeline"):
    pytest.skip("diffusers without Flux2KleinInpaintPipeline", allow_module_level=True)

import main  # noqa: E402
from engines.klein_engine import Flux2KleinEngine  # noqa: E402
from utils import base64_to_image, image_to_base64  # noqa: E402


class Cancelled(Exception):
    pass


def tiny_pipeline():
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinInpaintPipeline,
        Flux2Transformer2DModel,
    )
    from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    transformer = Flux2Transformer2DModel(
        patch_size=1, in_channels=4, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=16,
        timestep_guidance_channels=256, axes_dims_rope=[4, 4, 4, 4], guidance_embeds=False,
    )
    text_encoder = Qwen3ForCausalLM(Qwen3Config(
        intermediate_size=16, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, vocab_size=151936, max_position_embeddings=512,
    ))
    try:
        tokenizer = Qwen2TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
    except Exception as e:  # нет сети
        pytest.skip(f"tiny tokenizer unavailable: {e}")
    # 4 уровня VAE → scale factor 8, как у настоящей модели (кратность 16)
    vae = AutoencoderKLFlux2(
        sample_size=32, in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(4, 4, 4, 4), layers_per_block=1, latent_channels=1,
        norm_num_groups=1, use_quant_conv=False, use_post_quant_conv=False,
    )
    return Flux2KleinInpaintPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(), text_encoder=text_encoder,
        tokenizer=tokenizer, transformer=transformer, vae=vae,
    )


@pytest.fixture(scope="module")
def engine():
    eng = Flux2KleinEngine(device="cpu")
    eng.pipe = tiny_pipeline()
    eng.pipe_kwargs = {"text_encoder_out_layers": (1,), "max_sequence_length": 64}
    return eng


def noise(w, h, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


def test_output_matches_input_size(engine):
    for w, h in [(64, 48), (320, 176)]:
        out = engine.inpaint(noise(w, h), Image.new("L", (w, h), 255), seed=1)
        assert out.size == (w, h)


def test_rejects_sizes_the_pipeline_would_silently_resize(engine):
    with pytest.raises(ValueError):
        engine.inpaint(noise(70, 48), Image.new("L", (70, 48), 255))
    with pytest.raises(ValueError):
        engine.inpaint(noise(1280, 1024), Image.new("L", (1280, 1024), 255))


def test_default_four_steps_and_progress(engine):
    steps = []
    engine.inpaint(noise(64, 64), Image.new("L", (64, 64), 255), step_callback=lambda s, t: steps.append((s, t)))
    assert steps == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_cancel_between_steps(engine):
    def cb(step, total):
        if step == 2:
            raise Cancelled()

    with pytest.raises(Cancelled):
        engine.inpaint(noise(64, 64), Image.new("L", (64, 64), 255), step_callback=cb)


def test_progress_total_follows_strength(engine):
    steps = []
    engine.inpaint(noise(64, 64), Image.new("L", (64, 64), 255), strength=0.6,
                   step_callback=lambda s, t: steps.append((s, t)))
    assert steps and all(t == len(steps) for _, t in steps)


def test_thin_mask_is_grid_dilated():
    """Штрих в 10 px пропадал при сжатии маски до латентов (1/16)."""
    from engines.klein_engine import grid_dilate_mask

    arr = np.zeros((64, 96), np.uint8)
    arr[:, 10:20] = 255
    out = np.array(grid_dilate_mask(Image.fromarray(arr, "L")))
    assert (out[:, 0:32] == 255).all() and (out[:, 32:] == 0).all()


def test_seed_is_reproducible(engine):
    a = engine.inpaint(noise(64, 64), Image.new("L", (64, 64), 255), seed=7)
    b = engine.inpaint(noise(64, 64), Image.new("L", (64, 64), 255), seed=7)
    assert np.array_equal(np.array(a), np.array(b))


def test_full_inpaint_request_through_server(engine, monkeypatch):
    """Кадр не кратен 16 и больше 1 МП: пайплайн сервера кропает, дополняет
    до 16 и склеивает — вне маски пиксели исходные."""
    img = noise(1500, 1100, seed=3)
    mask = np.zeros((1100, 1500), np.uint8)
    mask[400:700, 600:900] = 255
    with TestClient(main.app) as client:
        # после старта: lifespan создаёт движки заново
        monkeypatch.setattr(main, "ai_engine", engine)
        r = client.post("/inpaint", json={
            "image": image_to_base64(img),
            "mask": image_to_base64(Image.fromarray(mask, "L")),
            "mode": "ai",
        })
    assert r.status_code == 200, r.text
    result = np.array(base64_to_image(r.json()["result"]))
    assert result.shape == (1100, 1500, 3)
    outside = mask == 0
    assert np.array_equal(result[outside], np.array(img)[outside])
