"""
FLUX.1 Fill dev — dedicated inpainting model via GGUF Q5_K_M quantization.
Runs on Apple Silicon M4 Pro 24GB with cpu_offload.
"""
import gc
import logging
import os
from typing import Optional

import torch
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)


def _patch_rope_for_mps():
    """
    Monkey-patch FLUX rope embedding to use float32 instead of float64.
    MPS does not support float64 tensors — this prevents RuntimeError.

    Idempotent: load()/unload()/load() cycles used to re-wrap
    FluxPosEmbed.forward on every load(), growing the closure chain each
    time (unload() never undid the patch). Guarded by an attribute on the
    function itself so repeated calls are a no-op.
    """
    try:
        from diffusers.models.embeddings import FluxPosEmbed
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not import FluxPosEmbed: {e}")
        return

    if getattr(FluxPosEmbed.forward, "_ae_mps_patched", False):
        return

    _orig_forward = FluxPosEmbed.forward

    def _patched_forward(self, ids):
        # Call original but ensure no float64 on MPS
        result = _orig_forward(self, ids)
        if result.dtype == torch.float64:
            result = result.to(torch.float32)
        return result

    _patched_forward._ae_mps_patched = True
    FluxPosEmbed.forward = _patched_forward
    logger.info("Patched FluxPosEmbed.forward for MPS float32 compatibility")


class FluxFillEngine(BaseEngine):
    """
    FLUX.1 Fill dev inpainting engine.
    Uses GGUF Q5_K_M quantization (~8GB) for Apple Silicon compatibility.

    Native resolution: 1024x1024 (multiples of 32).
    No negative prompts, no task tokens.
    """

    def __init__(
        self,
        gguf_repo: str = "YarvixPA/FLUX.1-Fill-dev-gguf",
        gguf_filename: str = "flux1-fill-dev-Q5_K_M.gguf",
        base_model: str = "black-forest-labs/FLUX.1-Fill-dev",
        default_guidance_scale: float = 30.0,
        default_num_inference_steps: int = 28,
        device: Optional[str] = None,
    ):
        self.gguf_repo = gguf_repo
        self.gguf_filename = gguf_filename
        self.base_model = base_model
        self.default_guidance_scale = default_guidance_scale
        self.default_num_inference_steps = default_num_inference_steps

        if device:
            self._device = device
        elif torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self.pipe = None
        logger.info(f"FluxFillEngine initialized, device: {self._device}")

    @property
    def name(self) -> str:
        return "flux-fill"

    @property
    def device(self) -> str:
        return self._device

    @property
    def supports_controlnet(self) -> bool:
        return False

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def load(self) -> None:
        if self.pipe is not None:
            logger.info("FLUX pipeline already loaded")
            return

        logger.info("Loading FLUX.1 Fill dev (GGUF Q5_K_M)...")

        # MPS requires fallback for unsupported ops
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        if self._device == "mps":
            _patch_rope_for_mps()

        # Step 1: Download GGUF file
        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading GGUF: {self.gguf_repo}/{self.gguf_filename}")
        gguf_path = hf_hub_download(
            repo_id=self.gguf_repo,
            filename=self.gguf_filename,
        )
        logger.info(f"GGUF path: {gguf_path}")

        # Step 2: Load quantized transformer
        from diffusers import FluxFillPipeline, FluxTransformer2DModel
        from diffusers.utils import export_to_video  # noqa: ensures gguf support is importable

        try:
            from diffusers import GGUFQuantizationConfig
        except ImportError:
            from diffusers.quantizers import GGUFQuantizationConfig

        logger.info("Loading GGUF transformer...")
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
            torch_dtype=torch.float16,
        )
        logger.info("Transformer loaded")

        # Step 3: Load pipeline with pre-loaded transformer
        logger.info(f"Loading FluxFillPipeline from {self.base_model}...")
        self.pipe = FluxFillPipeline.from_pretrained(
            self.base_model,
            transformer=transformer,
            torch_dtype=torch.float16,
        )
        logger.info("Pipeline loaded")

        # Step 4: Force VAE to float32 on MPS for stability
        if self._device == "mps":
            self.pipe.vae = self.pipe.vae.to(torch.float32)
            logger.info("VAE forced to float32 for MPS stability")

        # Step 5: Enable memory-efficient loading
        self.pipe.enable_model_cpu_offload()
        logger.info("CPU offload enabled")

        # Step 6: Attention slicing for MPS memory
        if self._device == "mps":
            self.pipe.enable_attention_slicing()
            logger.info("Attention slicing enabled for MPS")

        logger.info("FLUX.1 Fill engine ready")

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        logger.info("FLUX engine unloaded")

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 1.0,
        guidance_scale: float = 0.0,
        num_inference_steps: int = 0,
        controlnet_scale: float = 0.5,
        seed: Optional[int] = None,
        step_callback=None,
        task: str = "",
        **kwargs,
    ) -> Image.Image:
        """
        Run FLUX.1 Fill inpainting.

        FLUX ignores: negative_prompt, strength, controlnet_scale, task.
        guidance_scale and num_inference_steps use engine defaults if not overridden.
        """
        if self.pipe is None:
            raise RuntimeError("FLUX engine not loaded. Call load() first.")

        # Use engine defaults if caller passed zeros/defaults
        if guidance_scale <= 0.1:
            guidance_scale = self.default_guidance_scale
        if num_inference_steps <= 0:
            num_inference_steps = self.default_num_inference_steps

        # Ensure dimensions are multiples of 32
        w, h = image.size
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.info(f"Snapped to 32-multiples: ({w},{h}) -> ({new_w},{new_h})")

        # CPU generator (MPS doesn't support torch.Generator("mps") for seeding)
        generator = torch.Generator("cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        # Ensure image is RGB and mask is L
        if image.mode != "RGB":
            image = image.convert("RGB")
        if mask.mode != "L":
            mask = mask.convert("L")

        logger.info(
            f"FLUX inpaint: {new_w}x{new_h}, guidance={guidance_scale}, "
            f"steps={num_inference_steps}, seed={seed}, prompt='{prompt[:80]}'"
        )

        # Step callback wrapper for diffusers callback format
        callback_fn = None
        if step_callback is not None:
            def callback_fn(pipe, step_index, timestep, callback_kwargs):
                step_callback(step_index + 1, num_inference_steps)
                return callback_kwargs

        result = self.pipe(
            prompt=prompt if prompt else "background",
            image=image,
            mask_image=mask,
            height=new_h,
            width=new_w,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=generator,
            callback_on_step_end=callback_fn,
        ).images[0]

        logger.info(f"FLUX inpaint complete: {result.size}")
        return result
