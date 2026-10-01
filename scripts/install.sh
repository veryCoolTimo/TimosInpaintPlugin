#!/bin/bash
# Установка Python зависимостей для AE Inpaint Server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$PROJECT_DIR/server"

echo "=== AE Inpaint Server - Install ==="
echo "Project: $PROJECT_DIR"
echo ""

# This whole stack (MPS-specific code in the engines, `defaults write` in
# install_extension.sh) only targets macOS Apple Silicon. Previously this
# script installed the same way on any OS/arch with no warning — Linux/
# Windows/Intel Mac would either silently fall back to CPU inference or fail
# deep inside pip/torch with no clear message.
if [ "$(uname -s)" != "Darwin" ]; then
    echo "Warning: this project targets macOS (Apple Silicon). Detected OS: $(uname -s)."
    echo "The server may still run, but engines will fall back to CPU and this hasn't been tested."
elif [ "$(uname -m)" != "arm64" ]; then
    echo "Warning: this project targets Apple Silicon (arm64). Detected: $(uname -m)."
    echo "It may still run under Rosetta or on Intel Macs, but this hasn't been tested."
fi

# Проверяем Python. Нужен 3.10 или 3.11: pillow 9.5 (его требует LaMa) есть
# готовыми колёсами только до 3.11 — на 3.12+ (Homebrew ставит такой по
# умолчанию) pip пытается собрать его из исходников и обычно падает.
PYTHON=""
for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" &> /dev/null && \
        "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 11) else 1)'; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    FOUND=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "none")
    echo "Error: Python 3.10 or 3.11 required (found: $FOUND)."
    echo "Install it with:  brew install python@3.11"
    exit 1
fi
echo "Python: $PYTHON ($("$PYTHON" --version))"

# Создаём виртуальное окружение
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Активируем venv
source "$VENV_DIR/bin/activate"

# Обновляем pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Устанавливаем зависимости.
# requirements.lock.txt (если есть) — точные версии, verified working на
# Apple Silicon; requirements.txt — диапазоны на случай если lock-файл
# устарел или у тебя другая архитектура/Python. См. server/requirements.lock.txt.
echo ""
if [ -f "$SERVER_DIR/requirements.lock.txt" ]; then
    echo "Installing dependencies (from requirements.lock.txt, verified versions)..."
    pip install -r "$SERVER_DIR/requirements.lock.txt"
else
    echo "Installing dependencies (from requirements.txt, ranged — no lock file found)..."
    pip install -r "$SERVER_DIR/requirements.txt"
fi

# Создаём папку для моделей — server/models, а не корневой models/. Раньше
# здесь создавался PROJECT_DIR/models, но UpscaleEngine на самом деле
# сохраняет веса в server/models (Path(__file__).parent.parent / "models"
# из server/engines/upscale_engine.py) — два разных пустых/непустых
# каталога models/ путали план и факт.
mkdir -p "$SERVER_DIR/models"

echo ""
echo "=== Installation complete ==="
echo ""
echo "Installed:"
echo "  - venv: $VENV_DIR"
echo "  - Python deps: $SERVER_DIR/requirements.txt"
echo "  - Model weights will download on first use into:"
echo "      ~/.cache/huggingface/hub (diffusion models)"
echo "      $SERVER_DIR/models (Real-ESRGAN upscaler)"
echo ""
echo "Next step — install the CEP extension (this also starts the server"
echo "automatically the first time you click Inpaint, you don't need to run"
echo "start_server.sh yourself):"
echo "  ./scripts/install_extension.sh"
