# AE Inpaint Plugin

AI inpainting for After Effects. Remove objects and clean backgrounds from manhwa/manga artwork.

## Requirements

- macOS Apple Silicon (M1/M2/M3/M4)
- After Effects 2024+
- Python 3.10+

## Installation

```bash
# Install Python dependencies
./scripts/install.sh

# Install CEP extension
./scripts/install_extension.sh
```

## Usage

1. Open After Effects
2. Go to `Window > Extensions > AE Inpaint`
3. Server starts automatically
4. Select source layer
5. Create mask layer above (Shape/Solid, white = inpaint area)
6. Click Inpaint

## Project Structure

```
extension/     CEP panel (HTML/JS)
server/        Python backend (FastAPI + SDXL)
scripts/       Installation scripts
```

## Settings

| Parameter | Range | Description |
|-----------|-------|-------------|
| Strength | 0.5-1.0 | Denoising strength |
| Guidance | 1-15 | Prompt influence |
| Steps | 10-50 | Inference steps |
| Feather | 0-20px | Mask edge blur |
| Expand | 0-20px | Mask expansion |

## API

Server runs on `http://127.0.0.1:7860`

- `GET /health` - Server status
- `POST /inpaint` - Run inpainting
- `POST /load` - Load model
- `POST /clear-cache` - Clear cache

## Troubleshooting

Server not starting:
```bash
ls .venv/bin/python  # Check venv exists
./scripts/install.sh  # Reinstall if missing
```

Extension not visible:
```bash
defaults write com.adobe.CSXS.11 PlayerDebugMode 1
# Restart After Effects
```

## License

MIT
