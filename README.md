# AE Inpaint Plugin

AI inpainting for After Effects. Remove objects and clean backgrounds.

## Requirements

- macOS Apple Silicon (M1/M2/M3/M4) — this is the only platform tested; the
  server will run elsewhere but engines fall back to CPU and haven't been verified.
- macOS 14 (Sonoma) or newer for AI Gen (FLUX.2 klein runs in bfloat16 on
  the Apple GPU); 24 GB+ unified memory recommended (~16 GB of weights)
- After Effects 2024+
- Python 3.10 or 3.11 (`brew install python@3.11`; 3.12+ can't install
  the Pillow version LaMa needs)

## Installation

```bash
./scripts/install.sh
./scripts/install_extension.sh
```

Restart After Effects.

**Updating** an existing install: quit After Effects, `git pull`, then run
`./scripts/install.sh` again — it reuses `.venv` and installs the versions
from `server/requirements.lock.txt`. The panel starts the local server itself on first use —
you don't need to run `start_server.sh` manually (that script is only useful
for watching server logs directly while developing).

## How to Use

1. Open panel: `Window > Extensions > AE Inpaint`

2. Select layer with your image

3. Pick a mode:
   - **Remove** — LaMa. Fast, no prompt, good for plain object removal.
   - **AI Gen** — FLUX.2 [klein] 4B by default (4 steps; other engines in
     `server/config.py`). Slower than Remove, supports a text prompt, best
     for filling with something specific rather than just erasing, and for
     expanding a layer into its transparent area.
   - **Classic** — OpenCV Telea. Instant, no AI, best for small
     defects/simple textures; no hallucinations but no real understanding
     of content either.

4. Draw a mask on the selected layer (standard AE mask, `G` for the Pen
   tool). White/inside area = what gets inpainted. If the layer has several
   masks, the one currently selected in the AE timeline is used (Mask 1 if
   none is selected).

   No mask on the layer at all? The layer's alpha channel is used instead
   (transparent pixels = area to fill) — useful for extending/cleaning up
   footage that already has transparency. A layer with neither a mask nor
   an alpha channel returns an error.

5. Click **Inpaint**. **Stop** cancels the in-flight request and asks the
   server to abort generation between steps — the server may still finish
   the step it was already on, but no layer gets imported after a cancel.

Result appears as a new layer above the source, rendered at the
composition's resolution with the source layer's position/anchor/scale/
rotation baked in. It is **not** a live-linked layer — effects, track
mattes, 3D, parenting and time-remapping on the source layer are not
carried over, and the new layer won't follow further edits to the source.

## Files

- **Results** are saved to `AE Inpaint Results/` next to your `.aep`
  (or `~/Documents/AE Inpaint Results/` if the project was never saved) and
  imported into an `AE Inpaint Results` folder in the Project panel. Keep
  these files — the layers in your project reference them.
- **Temporary renders** (frame + mask sent to the server) go to the system
  temp folder (`$TMPDIR/ae-inpaint/`) and are deleted after every run, even
  on error or Stop. Leftovers from a crash are removed the next time the
  panel opens (if older than a day).
- **Server log**: `$TMPDIR/ae-inpaint/server.log` (+ `server.log.1`,
  rotated at 2 MB).
- The local server is started by the panel. Every open panel registers with
  it; when all of them are closed (or After Effects quits) the server cancels
  any running job and shuts down within a few seconds. A server started by
  hand with `scripts/start_server.sh` is never shut down automatically, even
  if panels connected to it.
- Older versions wrote `_AI_CACHE/` and `_AI_OUT/` next to the project.
  `_AI_CACHE` can be deleted; `_AI_OUT` holds results that older projects may
  still reference.

## Settings

Click **Settings** to adjust (AI Gen only, except where noted):
- Strength (0.3-1.0)
- Guidance (1-15)
- Steps (10-50)
- Seed (-1 = random)
- Invert mask, Crop to mask, Fill transparent *(all modes)*
- Mask feather/expand *(all modes)*

## Notes

- First run downloads the model for each mode: LaMa is small; AI Gen
  (FLUX.2 klein 4B) is about 16 GB. Expect a long wait on first use.
  klein is Apache 2.0 and not gated — no HuggingFace login needed.
- The previous engine, FLUX.1 Fill (`ENGINE_TYPE = "flux"` in
  `server/config.py`), still works but is slower (28 steps) and needs a
  one-time HuggingFace login: accept the license at
  https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev, create a token at
  https://huggingface.co/settings/tokens and run `hf auth login`
  from the project's `.venv` (or set `HF_TOKEN`).
- To check AI Gen speed/memory on your Mac without After Effects:
  `.venv/bin/python scripts/smoke_ai_engine.py` (see `--help`).
- Typical inpaint time varies with mode, image size and engine; AI Gen with
  a diffusion model is meaningfully slower than Remove/Classic.
- Add a prompt for better results — only used in **AI Gen** mode; Remove and
  Classic ignore it.
- This plugin processes one frame at a time. Running it across a range of
  frames for video does not have any temporal-consistency handling — expect
  flicker if you inpaint the same object across many consecutive frames.

## License

MIT

