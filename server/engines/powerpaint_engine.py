"""
Движок инпейнтинга на основе PowerPaint v2 (BrushNet + task prompts).
73% human preference для object removal vs 12% SDXL.
"""
import logging
import os
from typing import Optional

import torch
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)

# PowerPaint task prompts
TASK_PROMPTS = {
    "object_removal": "P_ctxt",
    "context_aware": "P_ctxt",
    "text_guided": "P_obj",
    "shape_guided": "P_shape",
    "image_outpainting": "P_ctxt",
}


class PowerPaintEngine(BaseEngine):
    """
    PowerPaint v2 — специализированная модель для инпейнтинга.
    Использует task prompts для переключения между режимами:
    - object_removal: удаление объектов (подавляет генерацию новых)
    - context_aware: заполнение фоном
    - text_guided: генерация по промпту
    """

    def __init__(
        self,
        model_id: str = "JunhaoZhuang/PowerPaint_v2",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.is_sdxl = False  # PowerPaint основан на SD 1.5

        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.pipe = None
        logger.info(f"PowerPaintEngine initialized, device: {self.device}")

    @property
    def name(self) -> str:
        return "powerpaint"

    @property
    def supports_controlnet(self) -> bool:
        return False

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def _ensure_files(self):
        """Download only the specific files we need from HuggingFace."""
        from huggingface_hub import hf_hub_download

        needed_files = [
            # Base model (RealisticVision)
            "realisticVisionV60B1_v51VAE/model_index.json",
            "realisticVisionV60B1_v51VAE/scheduler/scheduler_config.json",
            "realisticVisionV60B1_v51VAE/tokenizer/merges.txt",
            "realisticVisionV60B1_v51VAE/tokenizer/special_tokens_map.json",
            "realisticVisionV60B1_v51VAE/tokenizer/tokenizer_config.json",
            "realisticVisionV60B1_v51VAE/tokenizer/vocab.json",
            "realisticVisionV60B1_v51VAE/text_encoder/config.json",
            "realisticVisionV60B1_v51VAE/text_encoder/model.safetensors",
            "realisticVisionV60B1_v51VAE/unet/config.json",
            "realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model-002.safetensors",
            "realisticVisionV60B1_v51VAE/vae/config.json",
            "realisticVisionV60B1_v51VAE/vae/diffusion_pytorch_model.safetensors",
            # PowerPaint BrushNet weights
            "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors",
            "PowerPaint_Brushnet/pytorch_model.bin",
        ]

        for i, filename in enumerate(needed_files):
            logger.info(f"Ensuring file [{i+1}/{len(needed_files)}]: {filename}")
            hf_hub_download(self.model_id, filename)

    def load(self) -> None:
        """Загружает PowerPaint v2 pipeline.

        Sequence (from official app.py):
        1. Download needed files individually (NOT snapshot_download)
        2. Create BrushNet architecture from RealisticVision UNet
        3. Load text_encoder_brushnet from RealisticVision
        4. Create pipeline from realisticVisionV60B1_v51VAE base
        5. Replace UNet from realisticVision (custom UNet2DConditionModel)
        6. Add task tokens (P_ctxt, P_shape, P_obj) via TokenizerWrapper
        7. Load BrushNet weights from safetensors
        8. Load text encoder weights from pytorch_model.bin
        9. Set UniPC scheduler
        """
        if self.is_loaded():
            logger.info("PowerPaint already loaded")
            return

        logger.info(f"Loading PowerPaint model: {self.model_id}")

        from engines.powerpaint.pipeline_powerpaint_brushnet import (
            StableDiffusionPowerPaintBrushNetPipeline,
        )
        from engines.powerpaint.brushnet import BrushNetModel
        from engines.powerpaint.unet_2d_condition import UNet2DConditionModel
        from engines.powerpaint.utils import TokenizerWrapper, add_tokens
        from diffusers import DPMSolverMultistepScheduler
        from transformers import CLIPTextModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_model

        # MPS requires float32
        dtype = torch.float32 if self.device in ["mps", "cpu"] else torch.float16

        # Step 1: Download only needed files (not full snapshot)
        logger.info("Ensuring PowerPaint files are downloaded...")
        self._ensure_files()

        # Resolve base model path from HF cache
        # hf_hub_download returns: .../snapshots/<hash>/realisticVisionV60B1_v51VAE/model_index.json
        # We need: .../snapshots/<hash>/realisticVisionV60B1_v51VAE/
        base_model_path = os.path.dirname(
            hf_hub_download(self.model_id, "realisticVisionV60B1_v51VAE/model_index.json")
        )
        logger.info(f"Base model path: {base_model_path}")

        # Resolve PowerPaint weights paths
        brushnet_weights_path = hf_hub_download(
            self.model_id, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"
        )
        text_enc_weights_path = hf_hub_download(
            self.model_id, "PowerPaint_Brushnet/pytorch_model.bin"
        )

        # Step 2: Create BrushNet architecture from UNet
        logger.info("Creating BrushNet from UNet architecture...")
        unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            torch_dtype=dtype,
        )
        brushnet = BrushNetModel.from_unet(unet)

        # Step 3: Load text encoder for BrushNet
        logger.info("Loading text encoder for BrushNet...")
        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )

        # Step 4: Create pipeline from RealisticVision base
        # model_index.json doesn't list brushnet/text_encoder_brushnet,
        # so from_pretrained may not register them. We pass them as kwargs
        # and also manually assign after creation.
        logger.info("Loading pipeline from RealisticVision base...")
        self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
            base_model_path,
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
        )

        # Ensure PowerPaint-specific components are registered
        if not hasattr(self.pipe, 'text_encoder_brushnet') or self.pipe.text_encoder_brushnet is None:
            self.pipe.text_encoder_brushnet = text_encoder_brushnet
            logger.info("Manually registered text_encoder_brushnet")
        if not hasattr(self.pipe, 'brushnet') or self.pipe.brushnet is None:
            self.pipe.brushnet = brushnet
            logger.info("Manually registered brushnet")

        # Step 5: Replace UNet (use custom UNet2DConditionModel from PowerPaint)
        logger.info("Loading RealisticVision UNet...")
        self.pipe.unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            torch_dtype=dtype,
        )

        # Step 6: Setup TokenizerWrapper and add task tokens
        logger.info("Setting up task tokens...")
        self.pipe.tokenizer = TokenizerWrapper(
            from_pretrained=base_model_path,
            subfolder="tokenizer",
        )
        add_tokens(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )

        # Step 7: Load BrushNet weights from safetensors
        logger.info("Loading BrushNet weights...")
        load_model(self.pipe.brushnet, brushnet_weights_path)

        # Step 8: Load text encoder BrushNet weights.
        # weights_only=True (safe unpickling) is preferred — this is a
        # third-party .bin file from HF Hub, and torch.load() without it
        # will happily execute arbitrary pickled objects. Some older
        # checkpoints don't unpickle cleanly under weights_only=True (they
        # were saved with e.g. numpy scalars or other non-tensor objects),
        # so we fall back to the unsafe path with a loud warning rather than
        # hard-failing — but the safe path is what actually runs by default.
        logger.info("Loading text encoder BrushNet weights...")
        try:
            state_dict = torch.load(text_enc_weights_path, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.warning(
                f"torch.load(weights_only=True) failed ({e}); falling back to "
                f"weights_only=False for {text_enc_weights_path}. This trusts "
                f"the pickled content of that file — only acceptable because "
                f"it comes from the pinned PowerPaint HF repo."
            )
            state_dict = torch.load(text_enc_weights_path, map_location="cpu", weights_only=False)
        self.pipe.text_encoder_brushnet.load_state_dict(state_dict, strict=False)

        # Step 9: DPM++ 2M Karras — MPS-compatible, good quality at 20 steps
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

        self.pipe.to(self.device)

        # Explicitly move components that from_pretrained didn't register
        # (model_index.json only lists standard SD components, not brushnet/text_encoder_brushnet)
        self.pipe.brushnet.to(self.device)
        self.pipe.text_encoder_brushnet.to(self.device)

        if self.device == "mps":
            self.pipe.enable_attention_slicing()
            logger.info("MPS optimizations enabled")

        logger.info("PowerPaint loaded successfully")

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("PowerPaint unloaded")

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 1.0,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        controlnet_scale: float = 0.5,
        seed: Optional[int] = None,
        step_callback=None,
        task: str = "object_removal",
        **kwargs,
    ) -> Image.Image:
        """
        Инпейнтинг через PowerPaint.

        task: "object_removal" | "context_aware" | "text_guided"
        """
        if not self.is_loaded():
            raise RuntimeError("PowerPaint not loaded. Call load() first.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if image.size != mask.size:
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)

        w, h = image.size

        # Task token from the task parameter (NOT overridden by prompt presence)
        task_token = TASK_PROMPTS.get(task, "P_ctxt")

        # Build promptA/B with task tokens (PowerPaint dual-prompt format)
        # For object_removal/context_aware: use ONLY the task token, no user text.
        # PowerPaint's P_ctxt token alone tells the model to fill with surrounding context.
        # Adding user text with P_ctxt breaks the removal behavior.
        if task in ("object_removal", "context_aware", "image_outpainting"):
            promptA = task_token
            promptB = task_token
            unet_prompt = ""
        else:
            # text_guided: include user prompt with task token
            user_prompt = prompt or ""
            promptA = f"{user_prompt} {task_token}".strip()
            promptB = f"{user_prompt} {task_token}".strip()
            unet_prompt = user_prompt

        # Negative prompts always use P_obj to suppress unwanted object generation
        neg_token = TASK_PROMPTS["text_guided"]  # P_obj
        neg_promptA = f"{negative_prompt} {neg_token}".strip()
        neg_promptB = f"{negative_prompt} {neg_token}".strip()

        logger.info(
            f"Running PowerPaint: size={image.size}, task={task}, token={task_token}, "
            f"steps={num_inference_steps}, guidance={guidance_scale}"
        )
        logger.info(f"promptA: {promptA}")
        logger.info(f"negative_promptA: {neg_promptA}")

        # Callback for progress
        def on_step(pipe, step, timestep, callback_kwargs):
            if step_callback:
                step_callback(step + 1, num_inference_steps)
            return callback_kwargs

        pipe_kwargs = dict(
            promptA=promptA,
            promptB=promptB,
            prompt=unet_prompt,
            negative_promptA=neg_promptA,
            negative_promptB=neg_promptB,
            negative_prompt=negative_prompt,
            image=image,
            mask=mask,
            height=h,
            width=w,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            brushnet_conditioning_scale=1.0,
            generator=generator,
        )

        if step_callback:
            pipe_kwargs["callback_on_step_end"] = on_step

        result = self.pipe(**pipe_kwargs).images[0]

        logger.info("PowerPaint inpaint completed")
        return result
