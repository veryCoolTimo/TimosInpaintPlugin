#!/bin/bash
# Запуск сервера инпейнтинга

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$PROJECT_DIR/server"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== AE Inpaint Server ==="

# Проверяем venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found."
    echo "Run ./scripts/install.sh first"
    exit 1
fi

# Активируем venv
source "$VENV_DIR/bin/activate"

# Переходим в папку сервера
cd "$SERVER_DIR"

echo "Starting server on http://127.0.0.1:7860"
echo "Press Ctrl+C to stop"
echo ""

# Запускаем сервер
python -m uvicorn main:app --host 127.0.0.1 --port 7860 --reload
