#!/bin/bash
# Установка CEP расширения для After Effects

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXTENSION_DIR="$PROJECT_DIR/extension"

# Папка CEP расширений для macOS
CEP_DIR="$HOME/Library/Application Support/Adobe/CEP/extensions"

# Имя расширения
EXTENSION_NAME="com.timo.aeinpaint"

echo "=== AE Inpaint Extension - Install ==="
echo ""

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

# Для разных версий macOS
defaults write com.adobe.CSXS.11 PlayerDebugMode 1 2>/dev/null || true
defaults write com.adobe.CSXS.10 PlayerDebugMode 1 2>/dev/null || true
defaults write com.adobe.CSXS.9 PlayerDebugMode 1 2>/dev/null || true

echo ""
echo "=== Installation complete ==="
echo ""
echo "Restart After Effects to see the extension:"
echo "  Window > Extensions > AE Inpaint"
echo ""
echo "Don't forget to start the server:"
echo "  ./scripts/start_server.sh"
