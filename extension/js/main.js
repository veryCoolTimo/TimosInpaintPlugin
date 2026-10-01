/**
 * AE Inpaint Panel — связка AE (ExtendScript) ↔ локальный сервер.
 * Всё отображение — в ui.js (UI.*), файлы — в files.js (Files.*).
 */

const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let csInterface;
let isProcessing = false;
// Stop во время подготовки (запуск сервера, рендер кадра): запроса к
// серверу ещё нет, прерывать нечего — обработчик проверяет флаг сам
let cancelRequested = false;
let extensionPath = null;
let serverProcess = null;
let verboseLog = false; // true — подробности (ответы ExtendScript, вывод сервера) в журнал

// Модель AI Gen уже скачивалась — не спрашивать про загрузку
const FIRST_RUN_KEY = 'ae_inpaint_model_downloaded';

function loadJSX() {
    let extPath = extensionPath;
    if (extPath.startsWith('file://')) {
        extPath = decodeURIComponent(extPath.replace('file://', ''));
    }

    let jsxPath = path.join(extPath, 'jsx', 'host.jsx');
    try {
        jsxPath = fs.realpathSync(jsxPath);
    } catch (e) {
        console.error('Symlink resolve failed:', e);
    }

    try {
        let jsxContent = fs.readFileSync(jsxPath, 'utf8');
        if (jsxContent.charCodeAt(0) === 0xFEFF) {
            jsxContent = jsxContent.slice(1);
        }
        csInterface.evalScript(jsxContent, (result) => {
            console.log('JSX eval result:', result);
        });
    } catch (e) {
        console.error('JSX read error:', e.message);
    }
}

function init() {
    csInterface = new CSInterface();
    extensionPath = csInterface.getSystemPath('extension');

    UI.init({
        onInpaint: handleInpaint,
        onStop: handleStop,
        onUpscale: handleUpscale,
        onDebugToggle: handleToggleDebugMode,
    });

    loadJSX();

    csInterface.evalScript('app.version', (version) => log(`After Effects ${version}`));
    setTimeout(() => {
        csInterface.evalScript('typeof getProjectInfo', (result) => {
            if (result !== 'function') log('Panel scripts failed to load — reopen the panel', 'error');
        });
    }, 1000);

    // В консоли CEP: showDevTools()
    window.showDevTools = () => {
        UI.showDevTools();
        log('Dev tools enabled');
    };

    Files.cleanStaleTemp();
    isServerOnline().then(online => {
        UI.serverState(online ? 'online' : 'off');
        if (online) attachToServer();
    });
}

// ---------- helpers ----------

function log(message, kind = 'info') {
    UI.log(message, kind);
}

function logVerbose(message) {
    console.log(message);
    if (verboseLog) UI.log(message);
}

// Safe wrapper for building ExtendScript string literals from JS values:
// a `"` or backslash in a comp/layer name must not break out of the literal.
function jsxStr(value) {
    return JSON.stringify(String(value));
}

// Paths go into ExtendScript as forward-slash strings
function jsxPath(p) {
    return jsxStr(p.replace(/\\/g, '/'));
}

function getProjectPath() {
    let extPath = extensionPath;
    if (extPath.startsWith('file://')) {
        extPath = decodeURIComponent(extPath.replace('file://', ''));
    }
    let realPath = extPath;
    try {
        realPath = fs.realpathSync(extPath);
    } catch (e) {
        console.error('realpathSync failed:', e);
    }
    return path.dirname(realPath);
}

function evalScript(script) {
    return new Promise((resolve, reject) => {
        csInterface.evalScript(script, (result) => {
            logVerbose('evalScript result: ' + result?.substring?.(0, 100));
            if (result === 'EvalScript error.') {
                reject(new Error('After Effects script error — reopen the panel'));
                return;
            }
            if (result === 'undefined' || result === undefined || result === null) {
                reject(new Error('After Effects returned nothing — reopen the panel'));
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

function seconds(since) {
    return `${((Date.now() - since) / 1000).toFixed(1)} s`;
}

// ---------- server lifecycle ----------

function killOrphanServer() {
    // Убивает только наш uvicorn, оставшийся на 7860, — чужие процессы не трогает
    try {
        const { execSync } = require('child_process');
        const pids = execSync("lsof -ti:7860", { timeout: 3000 }).toString().trim();
        if (!pids) return;

        pids.split('\n').forEach(pid => {
            pid = pid.trim();
            if (!pid) return;
            let cmd = '';
            try {
                cmd = execSync(`ps -p ${pid} -o command=`, { timeout: 3000 }).toString();
            } catch (e) {
                return; // process already gone
            }
            if (cmd.indexOf('uvicorn') !== -1 && cmd.indexOf('main:app') !== -1) {
                execSync(`kill -9 ${pid}`, { timeout: 3000 });
                logVerbose('Killed orphan AE Inpaint server process on port 7860: ' + pid);
            } else {
                logVerbose(`Process ${pid} holds port 7860 but isn't our server (${cmd.trim()}) — leaving it alone.`);
            }
        });
    } catch (e) {
        // No process on port — that's fine
    }
}

function startServer() {
    return new Promise((resolve, reject) => {
        const projectPath = getProjectPath();
        const venvPython = path.join(projectPath, '.venv', 'bin', 'python');

        if (!fs.existsSync(venvPython)) {
            reject(new Error('Python environment not found — run scripts/install.sh first'));
            return;
        }

        killOrphanServer();
        UI.serverState('starting');
        log('Starting local server…');

        // Обработчики ниже относятся только к ЭТОМУ процессу: таймер или
        // exit старого, упавшего при старте процесса не должны трогать новый,
        // запущенный повторным кликом
        const proc = spawn(venvPython, ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '7860'], {
            cwd: path.join(projectPath, 'server'),
            env: { ...process.env, PYTHONUNBUFFERED: '1', AE_INPAINT_PARENT_PID: String(process.pid) }
        });
        serverProcess = proc;

        let started = false;
        let portBusy = false;
        // первый старт импортирует torch — на холодном диске это не 30 с
        const timer = setTimeout(() => {
            if (!started) {
                // Иначе зависший процесс остаётся сиротой, а следующий клик
                // запускает ещё один поверх
                proc.kill('SIGTERM');
                reject(new Error('Server did not start in 90 s'));
            }
        }, 90000);

        proc.stderr.on('data', (data) => {
            const msg = data.toString();
            logVerbose('[server] ' + msg.trim());
            if (/address already in use/i.test(msg)) {  // macOS: "Address already in use"
                portBusy = true;
            }
            // Только "Uvicorn running on": "Application startup complete"
            // печатается ДО попытки занять порт, и при занятом 7860 (это порт
            // Gradio по умолчанию) панель считала чужое приложение своим
            if (!started && msg.includes('Uvicorn running on')) {
                started = true;
                clearTimeout(timer);
                UI.serverState('online');
                log('Server ready', 'success');
                resolve();
            }
        });

        proc.stdout.on('data', (data) => {
            logVerbose('[server] ' + data.toString().trim());
        });

        proc.on('error', (err) => {
            clearTimeout(timer);
            UI.serverState('off');
            reject(new Error(`Server error: ${err.message}`));
        });

        proc.on('exit', (code) => {
            clearTimeout(timer);
            if (!started) {
                // Упал при старте (сломанный venv, занятый порт) — сразу ошибка
                reject(new Error(portBusy
                    ? 'Port 7860 is used by another application — close it and try again'
                    : `Server exited during startup (code ${code}). See ${Files.TEMP_ROOT}/server.log`));
            }
            if (serverProcess === proc) {
                serverProcess = null;
                UI.serverState('off');
            }
            logVerbose('Server stopped');
        });
    });
}

// Сервер может быть общим для нескольких панелей (второй экземпляр AE,
// переоткрытая панель): каждая сообщает свой PID, и сервер выключается сам,
// только когда закрылись все. Поэтому при закрытии панели сервер не
// убиваем — им может пользоваться другая.
async function attachToServer() {
    try {
        return await API.attach(process.pid);
    } catch (e) {
        logVerbose('Attach failed: ' + e.message);
        return null;
    }
}

// Запускает сервер, если он не отвечает, и регистрирует панель
async function ensureServer(tab) {
    if (await isServerOnline()) {
        const attached = await attachToServer();
        if (attached && attached.status !== 'shutting_down') return;
        // attach упал: старый сервер без /attach (ок, работаем с ним) или
        // сервер умер между запросами — тогда запускаем новый
        if (!attached && await isServerOnline()) return;
        // Сервер как раз выключается (закрылась последняя панель) — ждём,
        // пока он освободит порт, и запускаем новый
        for (let i = 0; i < 50 && await isServerOnline(); i++) {
            await new Promise(r => setTimeout(r, 200));
        }
    }
    UI.status(tab, 'Starting the local server…');
    await startServer();
    await attachToServer();
}

async function isServerOnline() {
    try {
        await API.healthCheck();
        return true;
    } catch {
        return false;
    }
}

// ---------- first run ----------

async function confirmModelDownload() {
    try {
        if (localStorage.getItem(FIRST_RUN_KEY) === 'true') return true;
    } catch (e) {
        // нет localStorage — спросим
    }
    try {
        const health = await API.healthCheck();
        if (health.model_cached) return true;
    } catch (e) {
        // сервер ещё не запущен — спросим
    }
    return UI.confirmFirstRun();
}

function markModelDownloaded() {
    try {
        localStorage.setItem(FIRST_RUN_KEY, 'true');
    } catch (e) {
        console.error('Failed to save first run flag:', e);
    }
}

// ---------- Stop ----------

let activeTab = null;

async function handleStop() {
    if (!isProcessing) return;
    cancelRequested = true;  // подготовка и цикл апскейла проверяют этот флаг
    UI.stopping(activeTab);

    // Прерываем запрос в панели и просим сервер остановить генерацию между
    // шагами. Состояние "занят" снимает finally прерванного обработчика —
    // иначе новую задачу можно было бы запустить поверх незавершённой.
    // /cancel — только если сервер сейчас выполняет НАШ запрос: сервер общий
    // для панелей, и Stop во время подготовки в одной отменял бы генерацию
    // в другой
    const ownJobRunning = API.hasActiveRequest();
    API.abortCurrent();
    if (ownJobRunning) {
        try {
            await API.cancelJob();
        } catch (e) {
            // сервер уже закончил или недоступен
        }
    }
}

function begin(tab) {
    isProcessing = true;
    cancelRequested = false;
    activeTab = tab;
    UI.setBusy(tab, true);
}

function checkCancelled() {
    if (cancelRequested) throw new Error('Cancelled');
}

// Ждёт promise, но сразу прерывается по Stop: холодный старт сервера идёт
// до 90 с, и кнопка Stop всё это время ничего не делала. Сам запуск
// продолжается в фоне — к следующему клику сервер будет готов.
function unlessCancelled(promise) {
    return new Promise((resolve, reject) => {
        const timer = setInterval(() => {
            if (cancelRequested) {
                clearInterval(timer);
                reject(new Error('Cancelled'));
            }
        }, 150);
        promise.then(
            (value) => { clearInterval(timer); resolve(value); },
            (error) => { clearInterval(timer); reject(error); }
        );
    });
}

function end(tab) {
    UI.setBusy(tab, false);
    isProcessing = false;
    activeTab = null;
}

// ---------- Fill ----------

async function handleInpaint() {
    if (isProcessing) return;

    const mode = UI.mode();
    const settings = UI.fillSettings();
    const started = Date.now();
    let jobDir = null;
    let progressTimer = null;
    let polling = false;

    // Панель занята сразу, ещё до окна про загрузку модели: иначе второй
    // клик за время его проверки запускал вторую задачу
    begin('fill');
    try {
        if (mode === 'ai' && !(await confirmModelDownload())) {
            UI.status('fill', 'Model download skipped');
            return;
        }

        await unlessCancelled(ensureServer('fill'));

        UI.status('fill', 'Reading the composition…');
        let projectInfo;
        try {
            projectInfo = await evalScript('getProjectInfo()');
        } catch (e) {
            throw new Error('After Effects script error — reopen the panel');
        }
        if (projectInfo.error) throw new Error(projectInfo.error);
        if (!projectInfo.projectPath) log(`Project not saved — results go to ~/Documents/${Files.RESULTS_FOLDER_NAME}`);
        // Папку результатов создаём до генерации: на read-only томе ошибка
        // должна прийти сразу, а не после минуты работы модели
        const outputDir = Files.getResultsDir(projectInfo.projectPath);

        const layerInfo = await evalScript('getSelectedLayerWithMask()');
        logVerbose('Layer info: ' + JSON.stringify(layerInfo));
        if (layerInfo.error) throw new Error(layerInfo.error);
        log(`${projectInfo.compName} · frame ${projectInfo.currentFrame} · ${layerInfo.name}` +
            (layerInfo.noMask ? ' (no mask: filling transparent area)' : ` · ${layerInfo.selectedMaskName}`));

        jobDir = Files.makeJobDir();
        let imageBase64, maskBase64;
        UI.status('fill', 'Rendering the frame…');
        if (layerInfo.noMask) {
            // Без маски: сервер строит маску из прозрачности слоя (expand)
            const exportResult = await evalScript(`exportLayerFrame(${layerInfo.index}, ${jsxPath(jobDir)})`);
            if (exportResult.error) throw new Error(exportResult.error);
            imageBase64 = await Files.readRenderedPng(exportResult.imagePath);
            maskBase64 = '';
        } else {
            const exportResult = await evalScript(
                `exportForInpaint(${layerInfo.index}, ${layerInfo.selectedMaskIndex}, ${jsxPath(jobDir)})`
            );
            if (exportResult.error) throw new Error(exportResult.error);
            imageBase64 = await Files.readRenderedPng(exportResult.imagePath);
            maskBase64 = await Files.readRenderedPng(exportResult.maskPath);
        }
        checkCancelled();

        const working = { remove: 'Removing…', ai: 'Generating…', clean: 'Filling…' }[mode] || 'Working…';
        UI.status('fill', working);
        if (mode === 'ai') {
            polling = true;
            progressTimer = setInterval(async () => {
                const prog = await API.getProgress();
                // Ответ мог прийти уже после конца генерации — не затирать
                // "Importing…"/"Stopping…"
                if (!prog || !polling || cancelRequested) return;
                if (prog.stage === 'loading_model') {
                    UI.progress('fill', null);
                    UI.status('fill', 'Loading the model… (the first time includes the download)');
                } else if (prog.stage === 'inpainting' && prog.total_steps > 0) {
                    UI.progress('fill', prog.step / prog.total_steps);
                    UI.status('fill', `Generating · step ${prog.step} of ${prog.total_steps}`);
                }
            }, 500);
        }

        let result;
        try {
            result = await API.inpaint({
                imageBase64,
                maskBase64,
                mode,
                prompt: settings.prompt,
                settings,
            });
        } finally {
            polling = false;
            if (progressTimer) clearInterval(progressTimer);
        }
        if (mode === 'ai') markModelDownloaded();

        UI.progress('fill', 1);
        UI.status('fill', 'Importing the result…');
        // Имя композиции — в путь файла: в нём бывают "/" и прочее
        const safeCompName = projectInfo.compName.replace(/[^a-zA-Z0-9]/g, '_');
        const resultPath = path.join(outputDir, `${safeCompName}_f${projectInfo.currentFrame}_${Date.now()}.png`);
        await Files.base64ToFile(result.result, resultPath);

        const timestamp = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const importResult = await evalScript(
            `importResultAsLayer(${jsxPath(resultPath)}, ${layerInfo.index}, ${jsxStr(`Inpaint ${timestamp}`)})`
        );
        if (importResult.error) throw new Error(importResult.error);

        UI.status('fill', `Done in ${seconds(started)} · “${importResult.layerName}”`, 'success');
        log(`Created “${importResult.layerName}” (${seconds(started)})`, 'success');
    } catch (error) {
        const message = error.message === 'Cancelled' ? 'Stopped' : error.message;
        UI.status('fill', message, message === 'Stopped' ? 'info' : 'error');
        log(message, message === 'Stopped' ? 'info' : 'error');
    } finally {
        if (jobDir) Files.removeDir(jobDir);
        end('fill');
    }
}

// ---------- Upscale ----------

async function handleUpscale() {
    if (isProcessing) return;

    const settings = UI.upscaleSettings();
    const started = Date.now();
    let jobDir = null;
    begin('upscale');
    try {
        await unlessCancelled(ensureServer('upscale'));

        UI.status('upscale', 'Reading the composition…');
        const projectInfo = await evalScript('getProjectInfo()');
        if (projectInfo.error) throw new Error(projectInfo.error);
        if (!projectInfo.projectPath) log(`Project not saved — results go to ~/Documents/${Files.RESULTS_FOLDER_NAME}`);

        jobDir = Files.makeJobDir();
        const outputDir = Files.getResultsDir(projectInfo.projectPath);

        const layersInfo = await evalScript('getSelectedLayers()');
        if (layersInfo.error) throw new Error(layersInfo.error);

        // Снизу вверх: индексы ещё не обработанных слоёв не сдвигаются от
        // вставки результатов над уже обработанными
        const layers = layersInfo.layers.sort((a, b) => b.index - a.index);
        log(`Upscaling ${layers.length} layer(s) ×${settings.scale}`);

        let done = 0;
        let failed = 0;
        for (let i = 0; i < layers.length && !cancelRequested; i++) {
            const layer = layers[i];
            const label = layers.length > 1 ? `${i + 1}/${layers.length} · ${layer.name}` : layer.name;
            UI.progress('upscale', i / layers.length);
            UI.status('upscale', `Rendering ${label}…`);

            try {
                const exportResult = await evalScript(`exportLayerFrame(${layer.index}, ${jsxPath(jobDir)})`);
                if (!exportResult || exportResult.error || !exportResult.imagePath) {
                    throw new Error(exportResult?.error || 'export failed');
                }
                const imageBase64 = await Files.readRenderedPng(exportResult.imagePath);
                if (cancelRequested) break;

                UI.status('upscale', `Upscaling ${label}…`);
                const result = await API.upscale({ imageBase64, scale: settings.scale, modelType: settings.modelType });
                if (cancelRequested) break;

                const safeName = layer.name.replace(/[^a-zA-Z0-9]/g, '_');
                const resultPath = path.join(outputDir, `${safeName}_x${settings.scale}_${Date.now()}.png`);
                await Files.base64ToFile(result.result, resultPath);

                // Результат в N раз больше композиции — импортируем в 100/N %,
                // чтобы кадрирование совпало, а пикселей стало больше
                const importResult = await evalScript(
                    `importResultAsLayer(${jsxPath(resultPath)}, ${exportResult.layerIndex}, ` +
                    `${jsxStr(`${layer.name} ×${settings.scale}`)}, ${100 / settings.scale})`
                );
                if (importResult.error) throw new Error(importResult.error);
                done++;
            } catch (e) {
                if (e.message === 'Cancelled') break;
                failed++;
                log(`${layer.name}: ${e.message}`, 'error');
            }
        }

        UI.progress('upscale', 1);
        if (cancelRequested) {
            UI.status('upscale', `Stopped · ${done} of ${layers.length} done`);
        } else if (failed) {
            UI.status('upscale', `${done} of ${layers.length} done, ${failed} failed — see Activity`, 'error');
        } else {
            UI.status('upscale', `Done in ${seconds(started)} · ${done} layer${done === 1 ? '' : 's'}`, 'success');
        }
        log(`Upscale: ${done}/${layers.length} done (${seconds(started)})`,
            failed ? 'error' : (cancelRequested ? 'info' : 'success'));
    } catch (error) {
        const message = error.message === 'Cancelled' ? 'Stopped' : error.message;
        UI.status('upscale', message, message === 'Stopped' ? 'info' : 'error');
        log(message, message === 'Stopped' ? 'info' : 'error');
    } finally {
        if (jobDir) Files.removeDir(jobDir);
        end('upscale');
    }
}

// ---------- dev tools ----------

function handleToggleDebugMode() {
    const { exec } = require('child_process');
    exec('defaults read com.adobe.CSXS.11 PlayerDebugMode 2>/dev/null || echo "0"', (err, stdout) => {
        const newVal = stdout.trim() === '1' ? '0' : '1';
        const cmds = [
            `defaults write com.adobe.CSXS.11 PlayerDebugMode ${newVal}`,
            `defaults write com.adobe.CSXS.10 PlayerDebugMode ${newVal}`,
            `defaults write com.adobe.CSXS.9 PlayerDebugMode ${newVal}`
        ].join(' && ');
        exec(cmds, () => {
            UI.setDebugLabel(newVal === '1');
            log(`CEP Debug ${newVal === '1' ? 'enabled' : 'disabled'}. Restart AE.`, 'success');
        });
    });
}

document.addEventListener('DOMContentLoaded', init);
