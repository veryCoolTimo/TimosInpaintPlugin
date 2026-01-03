#!/bin/bash
# Установка Python зависимостей для AE Inpaint Server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$PROJECT_DIR/server"

echo "=== AE Inpaint Server - Install ==="
echo "Project: $PROJECT_DIR"
echo ""

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Создаём виртуальное окружение
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Активируем venv
source "$VENV_DIR/bin/activate"

# Обновляем pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Устанавливаем зависимости
echo ""
echo "Installing dependencies..."
pip install -r "$SERVER_DIR/requirements.txt"

# Создаём папку для моделей
mkdir -p "$PROJECT_DIR/models"

echo ""
echo "=== Installation complete ==="
echo ""
echo "To start the server:"
echo "  ./scripts/start_server.sh"
echo ""
echo "To install the CEP extension:"
echo "  ./scripts/install_extension.sh"
