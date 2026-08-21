# AE Inpaint Plugin

AI inpainting for After Effects. Remove objects and clean backgrounds.

## Requirements

- macOS Apple Silicon (M1/M2/M3/M4) — this is the only platform tested; the
  server will run elsewhere but engines fall back to CPU and haven't been verified.
- After Effects 2024+
- Python 3.10+

## Installation

```bash
./scripts/install.sh
./scripts/install_extension.sh
```

Restart After Effects. The panel starts the local server itself on first use —
you don't need to run `start_server.sh` manually (that script is only useful
for watching server logs directly while developing).

## How to Use

1. **Save your AE project first.** The panel needs a project path to know
   where to write cache/output files, and fails immediately if the project
   was never saved.

2. Open panel: `Window > Extensions > AE Inpaint`

3. Select layer with your image

4. Pick a mode:
   - **Remove** — LaMa. Fast, no prompt, good for plain object removal.
   - **AI Gen** — the configured diffusion engine (FLUX.1 Fill by default;
     see `server/config.py`). Slower, supports a text prompt, best for
     filling with something specific rather than just erasing.
   - **Classic** — OpenCV Telea. Instant, no AI, best for small
     defects/simple textures; no hallucinations but no real understanding
     of content either.

5. Draw a mask on the selected layer (standard AE mask, `G` for the Pen
   tool). White/inside area = what gets inpainted. If the layer has several
   masks, the one currently selected in the AE timeline is used (Mask 1 if
   none is selected).

   No mask on the layer at all? The layer's alpha channel is used instead
   (transparent pixels = area to fill) — useful for extending/cleaning up
   footage that already has transparency. A layer with neither a mask nor
   an alpha channel returns an error.

6. Click **Inpaint**. **Stop** cancels the in-flight request and asks the
   server to abort generation between steps — the server may still finish
   the step it was already on, but no layer gets imported after a cancel.

Result appears as a new layer above the source, rendered at the
composition's resolution with the source layer's position/anchor/scale/
rotation baked in. It is **not** a live-linked layer — effects, track
mattes, 3D, parenting and time-remapping on the source layer are not
carried over, and the new layer won't follow further edits to the source.

## Settings

Click **Settings** to adjust (AI Gen only, except where noted):
- Strength (0.3-1.0)
- Guidance (1-15)
- Steps (10-50)
- Seed (-1 = random)
- Invert mask, Crop to mask, Fill transparent *(all modes)*
- Mask feather/expand *(all modes)*

## Notes

- First run downloads the configured model — size varies a lot by engine
  (LaMa is small and fast to fetch; FLUX.1 Fill and PowerPaint are several
  GB each). Expect a long wait on first use of a given mode.
- Typical inpaint time varies with mode, image size and engine; AI Gen with
  a diffusion model is meaningfully slower than Remove/Classic.
- Add a prompt for better results — only used in **AI Gen** mode; Remove and
  Classic ignore it.
- This plugin processes one frame at a time. Running it across a range of
  frames for video does not have any temporal-consistency handling — expect
  flicker if you inpaint the same object across many consecutive frames.

## License

MIT

Note: `server/engines/powerpaint/` vendors adapted code from
[open-mmlab/PowerPaint](https://github.com/open-mmlab/PowerPaint) and
[TencentARC/BrushNet](https://github.com/TencentARC/BrushNet) (Apache 2.0) —
see `server/engines/powerpaint/NOTICE`.
