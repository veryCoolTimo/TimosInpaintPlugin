# AE Inpaint Plugin

AI inpainting for After Effects. Remove objects and clean backgrounds.

## Requirements

- macOS Apple Silicon (M1/M2/M3/M4)
- After Effects 2024+
- Python 3.10+

## Installation

```bash
./scripts/install.sh
./scripts/install_extension.sh
```

Restart After Effects.

## How to Use

1. Open panel: `Window > Extensions > AE Inpaint`

2. Select layer with your image

3. Draw mask on that layer:
   - Press `G` (Pen tool)
   - Draw shape around area to inpaint
   - White area = AI will fill this

4. Click **Inpaint**

Result appears as new layer above, with matching transforms.

## Mask Tips

- Mask is drawn directly on source layer (standard AE mask)
- If multiple masks: select one, or Mask 1 is used
- No mask = error message

## Settings

Click **Settings** to adjust:
- Strength (0.5-1.0)
- Guidance (1-15)
- Steps (10-50)
- Mask feather/expand

## Notes

- First run downloads model (~5GB)
- Each inpaint: 20-40 seconds
- Add prompt for better results

## License

MIT
