/**
 * AE Inpaint Panel - Main Logic
 */

// Node.js modules (доступны в CEP)
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Глобальные переменные
let csInterface;
let serverOnline = false;
let isProcessing = false;
let serverProcess = null;
let extensionPath = null;

// DOM элементы
const elements = {
    btnInpaint: null,
    btnDebug: null,
    btnClearCache: null,
    serverStatus: null,
    prompt: null,
    log: null,
    progressOverlay: null,
    progressText: null,
    // Sliders
    strength: null,
    guidance: null,
    steps: null,
    feather: null,
    expand: null
};

/**
 * Инициализация
 */
function init() {
    // CSInterface
    csInterface = new CSInterface();

    // Получаем путь к расширению
    extensionPath = csInterface.getSystemPath('extension');
    // Путь к проекту (на уровень выше extension/)
    const projectPath = path.dirname(extensionPath);

    // Кэшируем элементы
    elements.btnInpaint = document.getElementById('btn-inpaint');
    elements.btnDebug = document.getElementById('btn-debug');
    elements.btnClearCache = document.getElementById('btn-clear-cache');
    elements.serverStatus = document.getElementById('server-status');
    elements.prompt = document.getElementById('prompt');
    elements.log = document.getElementById('log');
    elements.progressOverlay = document.getElementById('progress-overlay');
    elements.progressText = document.getElementById('progress-text');

    elements.strength = document.getElementById('strength');
    elements.guidance = document.getElementById('guidance');
    elements.steps = document.getElementById('steps');
    elements.feather = document.getElementById('feather');
    elements.expand = document.getElementById('expand');

    // Привязываем обработчики
    elements.btnInpaint.addEventListener('click', handleInpaint);
    elements.btnDebug.addEventListener('click', handleDebugExport);
    elements.btnClearCache.addEventListener('click', handleClearCache);

    // Слайдеры
    setupSlider('strength');
    setupSlider('guidance');
    setupSlider('steps');
    setupSlider('feather');
    setupSlider('expand');

    // Проверяем сервер и автозапуск
    initServer();
    setInterval(checkServerStatus, 10000); // каждые 10 сек

    log('Panel initialized', 'info');
    log(`Extension: ${extensionPath}`, 'info');
}

/**
 * Получить путь к проекту (корень репозитория)
 */
function getProjectPath() {
    // extensionPath указывает на папку extension/
    // проект находится на уровень выше
    return path.dirname(extensionPath);
}

/**
 * Инициализация сервера (проверка + автозапуск)
 */
async function initServer() {
    updateServerStatus(false, 'Checking...');

    try {
        await API.healthCheck();
        updateServerStatus(true, 'Ready');
        log('Server already running', 'success');
    } catch (error) {
        log('Server offline, starting...', 'warning');
        startServer();
    }
}

/**
 * Запуск Python сервера
 */
function startServer() {
    const projectPath = getProjectPath();
    const venvPython = path.join(projectPath, '.venv', 'bin', 'python');
    const serverMain = path.join(projectPath, 'server', 'main.py');

    // Проверяем существование venv
    if (!fs.existsSync(venvPython)) {
        log('Error: venv not found. Run install.sh first', 'error');
        updateServerStatus(false, 'No venv');
        return;
    }

    log('Starting server...', 'info');
    updateServerStatus(false, 'Starting...');

    // Запускаем сервер
    serverProcess = spawn(venvPython, ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '7860'], {
        cwd: path.join(projectPath, 'server'),
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe']
    });

    // Логируем stdout
    serverProcess.stdout.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg) {
            console.log('[Server]', msg);
            // Проверяем готовность
            if (msg.includes('Uvicorn running') || msg.includes('Application startup complete')) {
                log('Server started', 'success');
                checkServerStatus();
            }
        }
    });

    // Логируем stderr
    serverProcess.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg) {
            console.error('[Server Error]', msg);
            // Uvicorn пишет в stderr даже при успехе
            if (msg.includes('Uvicorn running') || msg.includes('Started')) {
                checkServerStatus();
            }
        }
    });

    serverProcess.on('error', (err) => {
        log(`Server error: ${err.message}`, 'error');
        updateServerStatus(false, 'Error');
    });

    serverProcess.on('exit', (code) => {
        log(`Server exited with code ${code}`, code === 0 ? 'info' : 'error');
        serverProcess = null;
        updateServerStatus(false, 'Stopped');
    });

    // Не ждём завершения
    serverProcess.unref();

    // Проверяем через 3 секунды
    setTimeout(checkServerStatus, 3000);
}

/**
 * Остановка сервера
 */
function stopServer() {
    if (serverProcess) {
        log('Stopping server...', 'info');
        serverProcess.kill('SIGTERM');
        serverProcess = null;
    }
}

/**
 * Настройка слайдера
 */
function setupSlider(id) {
    const slider = document.getElementById(id);
    const valueSpan = document.getElementById(`${id}-value`);

    slider.addEventListener('input', () => {
        valueSpan.textContent = slider.value;
    });
}

/**
 * Переключение секции
 */
function toggleSection(header) {
    const section = header.parentElement;
    section.classList.toggle('collapsed');
}

/**
 * Логирование
 */
function log(message, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;

    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    entry.textContent = `[${time}] ${message}`;

    elements.log.appendChild(entry);
    elements.log.scrollTop = elements.log.scrollHeight;

    // Ограничиваем количество записей
    while (elements.log.children.length > 50) {
        elements.log.removeChild(elements.log.firstChild);
    }
}

/**
 * Показать/скрыть прогресс
 */
function showProgress(text) {
    elements.progressText.textContent = text;
    elements.progressOverlay.classList.remove('hidden');
    isProcessing = true;
}

function hideProgress() {
    elements.progressOverlay.classList.add('hidden');
    isProcessing = false;
}

/**
 * Обновление статуса сервера
 */
function updateServerStatus(online, text) {
    serverOnline = online;
    elements.serverStatus.className = `status-indicator ${online ? 'online' : 'offline'}`;
    elements.serverStatus.querySelector('.status-text').textContent = text;
    elements.btnInpaint.disabled = !online || isProcessing;
}

/**
 * Проверка статуса сервера
 */
async function checkServerStatus() {
    try {
        const health = await API.healthCheck();
        updateServerStatus(true, health.engine_loaded ? 'Ready' : 'Model not loaded');
    } catch (error) {
        updateServerStatus(false, 'Offline');
    }
}

/**
 * Вызов ExtendScript функции
 */
function evalScript(script) {
    return new Promise((resolve, reject) => {
        csInterface.evalScript(script, (result) => {
            if (result === 'EvalScript error.' || result === 'undefined') {
                reject(new Error('ExtendScript error'));
                return;
            }
            try {
                resolve(JSON.parse(result));
            } catch (e) {
                resolve(result);
            }
        });
    });
}

/**
 * Получить настройки
 */
function getSettings() {
    return {
        strength: parseFloat(elements.strength.value),
        guidance: parseFloat(elements.guidance.value),
        steps: parseInt(elements.steps.value),
        feather: parseInt(elements.feather.value),
        expand: parseInt(elements.expand.value)
    };
}

/**
 * Главный обработчик инпейнтинга
 */
async function handleInpaint() {
    if (isProcessing || !serverOnline) return;

    try {
        showProgress('Preparing...');
        log('Starting inpaint process...', 'info');

        // 1. Получаем информацию о проекте
        const projectInfo = await evalScript('getProjectInfo()');
        if (projectInfo.error) {
            throw new Error(projectInfo.error);
        }
        log(`Comp: ${projectInfo.compName}, Frame: ${projectInfo.currentFrame}`, 'info');

        // 2. Находим маску
        const maskInfo = await evalScript('findMaskLayer()');
        if (!maskInfo.found) {
            throw new Error(maskInfo.error || 'Mask layer not found');
        }
        log(`Found mask: ${maskInfo.name}`, 'info');

        // 3. Получаем выбранный слой (источник)
        const selectedLayer = await evalScript('getSelectedLayer()');
        if (selectedLayer.error) {
            throw new Error(selectedLayer.error);
        }
        log(`Source layer: ${selectedLayer.name}`, 'info');

        // 4. Экспортируем кадр и маску
        showProgress('Exporting frames...');

        const cacheDir = projectInfo.projectPath + '/_AI_CACHE';
        const exportResult = await evalScript(
            `exportForInpaint(${selectedLayer.index}, ${maskInfo.index}, "${cacheDir.replace(/\\/g, '/')}")`
        );

        if (exportResult.error) {
            throw new Error(exportResult.error);
        }
        log(`Exported: ${exportResult.imagePath}`, 'info');

        // 5. Читаем файлы в base64
        showProgress('Loading images...');

        const imageBase64 = await fileToBase64(exportResult.imagePath);
        const maskBase64 = await fileToBase64(exportResult.maskPath);

        // 6. Отправляем на сервер
        showProgress('Running AI inference...');
        log('Sending to server...', 'info');

        const settings = getSettings();
        const prompt = elements.prompt.value.trim();

        const result = await API.inpaint({
            imageBase64,
            maskBase64,
            prompt,
            settings,
            cacheDir: projectInfo.projectPath
        });

        if (result.cached) {
            log('Result from cache!', 'success');
        } else {
            log('Inference completed', 'success');
        }

        // 7. Сохраняем результат
        showProgress('Importing result...');

        const outputDir = projectInfo.projectPath + '/_AI_OUT';
        const resultPath = `${outputDir}/${projectInfo.compName}_frame${projectInfo.currentFrame}_result.png`;

        await base64ToFile(result.result, resultPath);
        log(`Saved: ${resultPath}`, 'info');

        // 8. Импортируем в AE
        const importResult = await evalScript(
            `importResultAsLayer("${resultPath.replace(/\\/g, '/')}", ${selectedLayer.index}, "Inpaint Result")`
        );

        if (importResult.error) {
            throw new Error(importResult.error);
        }

        log(`Created layer: ${importResult.layerName}`, 'success');
        log('Done!', 'success');

    } catch (error) {
        log(`Error: ${error.message}`, 'error');
        console.error(error);
    } finally {
        hideProgress();
    }
}

/**
 * Debug экспорт (без inference)
 */
async function handleDebugExport() {
    try {
        log('Debug export...', 'info');

        const projectInfo = await evalScript('getProjectInfo()');
        if (projectInfo.error) {
            throw new Error(projectInfo.error);
        }

        const maskInfo = await evalScript('findMaskLayer()');
        if (!maskInfo.found) {
            throw new Error(maskInfo.error || 'Mask layer not found');
        }

        const selectedLayer = await evalScript('getSelectedLayer()');
        if (selectedLayer.error) {
            throw new Error(selectedLayer.error);
        }

        const cacheDir = projectInfo.projectPath + '/_AI_CACHE';
        const exportResult = await evalScript(
            `exportForInpaint(${selectedLayer.index}, ${maskInfo.index}, "${cacheDir.replace(/\\/g, '/')}")`
        );

        if (exportResult.error) {
            throw new Error(exportResult.error);
        }

        log(`Image: ${exportResult.imagePath}`, 'success');
        log(`Mask: ${exportResult.maskPath}`, 'success');

    } catch (error) {
        log(`Debug error: ${error.message}`, 'error');
    }
}

/**
 * Очистка кэша
 */
async function handleClearCache() {
    try {
        const projectInfo = await evalScript('getProjectInfo()');
        if (projectInfo.error) {
            throw new Error(projectInfo.error);
        }

        await API.clearCache(projectInfo.projectPath);
        log('Cache cleared', 'success');

    } catch (error) {
        log(`Clear cache error: ${error.message}`, 'error');
    }
}

// Инициализация при загрузке
document.addEventListener('DOMContentLoaded', init);

// Остановка сервера при закрытии панели
window.addEventListener('beforeunload', () => {
    // Не останавливаем сервер — пусть работает в фоне
    // Если хочешь останавливать: stopServer();
});

// Экспорт для глобального доступа
window.toggleSection = toggleSection;
