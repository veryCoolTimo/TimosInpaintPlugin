#!/bin/bash
# Установка CEP расширения для After Effects

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXTENSION_DIR="$PROJECT_DIR/extension"

# Имя расширения
EXTENSION_NAME="com.timo.aeinpaint"

echo "=== AE Inpaint Extension - Install ==="
echo ""

# CEP extensions path below is macOS-specific. Previously this script just
# ran the same way on any OS and silently did the wrong thing (or nothing
# useful) on Windows — After Effects on Windows uses
# %APPDATA%\Adobe\CEP\extensions instead, which this script doesn't handle.
if [ "$(uname -s)" != "Darwin" ]; then
    echo "Error: this installer only supports macOS."
    echo "On Windows, the CEP extensions folder is %APPDATA%\\Adobe\\CEP\\extensions —"
    echo "this script would need a Windows-specific port to install there; it hasn't been written."
    exit 1
fi

# Папка CEP расширений для macOS
CEP_DIR="$HOME/Library/Application Support/Adobe/CEP/extensions"

# Создаём папку CEP если не существует
mkdir -p "$CEP_DIR"

# Путь к симлинку
LINK_PATH="$CEP_DIR/$EXTENSION_NAME"

# Удаляем старый симлинк если есть
if [ -L "$LINK_PATH" ]; then
    echo "Removing old symlink..."
    rm "$LINK_PATH"
elif [ -d "$LINK_PATH" ]; then
    echo "Removing old directory..."
    rm -rf "$LINK_PATH"
fi

# Создаём симлинк
echo "Creating symlink..."
ln -s "$EXTENSION_DIR" "$LINK_PATH"

echo ""
echo "Extension installed to:"
echo "  $LINK_PATH"
echo ""

# Включаем режим разработки для CEP (отключает проверку подписи)
echo "Enabling CEP debug mode..."

# Флаг ставится для каждой версии CEP отдельно: AE 2024 — CEP 11,
# AE 2025/2026 — CEP 12. Без флага своей версии AE молча не показывает
# неподписанную панель в Window > Extensions.
for v in 9 10 11 12 13; do
    defaults write "com.adobe.CSXS.$v" PlayerDebugMode 1 2>/dev/null || true
done

echo ""
echo "=== Installation complete ==="
echo ""
echo "Restart After Effects to see the extension:"
echo "  Window > Extensions > AE Inpaint"
echo ""
echo "The panel starts the local server itself on first use — no separate"
echo "step needed. (./scripts/start_server.sh is only for watching server"
echo "logs directly while developing.)"
