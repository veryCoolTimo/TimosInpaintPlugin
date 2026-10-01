/**
 * AE Inpaint Panel - Main Logic
 */

const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');


let csInterface;
let isProcessing = false;
let upscaleCancelled = false;
let extensionPath = null;
let serverProcess = null;
let isFirstRun = true;
let verboseLog = false; // Set to true for debug logging

const elements = {};

// Local storage key for first run check
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

    console.log('Loading JSX from:', jsxPath);

    try {
        let jsxContent = fs.readFileSync(jsxPath, 'utf8');
        if (jsxContent.charCodeAt(0) === 0xFEFF) {
            jsxContent = jsxContent.slice(1);
        }
        console.log('JSX content length:', jsxContent.length);
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

    // Cache DOM elements - Fill tab
    elements.btnInpaint = document.getElementById('btn-inpaint');
    elements.btnStop = document.getElementById('btn-stop');
    elements.btnToggleSettings = document.getElementById('btn-toggle-settings');
    elements.btnDebugMode = document.getElementById('btn-debug-mode');
    elements.settingsPanel = document.getElementById('settings-panel');
    elements.devTools = document.getElementById('dev-tools');
    elements.prompt = document.getElementById('prompt');
    elements.negativePrompt = document.getElementById('negative-prompt');
    elements.log = document.getElementById('log');
    elements.strength = document.getElementById('strength');
    elements.guidance = document.getElementById('guidance');
    elements.steps = document.getElementById('steps');
    elements.seed = document.getElementById('seed');
    elements.feather = document.getElementById('feather');
    elements.expand = document.getElementById('expand');
    elements.invertMask = document.getElementById('invert-mask');
    elements.cropToMask = document.getElementById('crop-to-mask');
    elements.fillTransparent = document.getElementById('fill-transparent');
    elements.statusIndicator = document.getElementById('status-indicator');
    elements.statusText = document.getElementById('status-text');
    elements.progressOverlay = document.getElementById('progress-overlay');
    elements.progressText = document.getElementById('progress-text');
    elements.progressDetail = document.getElementById('progress-detail');
    elements.firstRunModal = document.getElementById('first-run-modal');
    elements.btnFirstRunCancel = document.getElementById('btn-first-run-cancel');
    elements.btnFirstRunContinue = document.getElementById('btn-first-run-continue');
    elements.btnStopOverlay = document.getElementById('btn-stop-overlay');

    // Cache DOM elements - Upscale tab
    elements.btnUpscale = document.getElementById('btn-upscale');
    elements.tabFill = document.getElementById('tab-fill');
    elements.tabUpscale = document.getElementById('tab-upscale');

    // Check if model was already downloaded
    try {
        isFirstRun = localStorage.getItem(FIRST_RUN_KEY) !== 'true';
    } catch (e) {
        isFirstRun = true;
    }

    // Load JSX
    loadJSX();

    // Test ExtendScript
    csInterface.evalScript('app.version', (result) => {
        console.log('AE version:', result);
        log('AE ' + result);
    });

    // Verify JSX loaded
    setTimeout(() => {
        csInterface.evalScript('typeof getProjectInfo', (result) => {
            console.log('getProjectInfo type:', result);
            if (result === 'function') {
                log('Ready');
            } else {
                log('JSX load failed', 'error');
            }
        });
    }, 1000);

    // Event handlers - Fill tab
    elements.btnInpaint.addEventListener('click', handleInpaint);
    elements.btnStop.addEventListener('click', handleStop);
    elements.btnToggleSettings.addEventListener('click', handleToggleSettings);
    elements.btnDebugMode.addEventListener('click', handleToggleDebugMode);
    elements.btnFirstRunCancel.addEventListener('click', hideFirstRunModal);
    elements.btnFirstRunContinue.addEventListener('click', handleFirstRunContinue);
    elements.btnStopOverlay.addEventListener('click', handleStop);

    // Event handlers - Upscale tab
    elements.btnUpscale.addEventListener('click', handleUpscale);

    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => handleTabSwitch(tab.dataset.tab));
    });

    // Setup radio groups with visual feedback
    setupRadioGroup('mode');
    setupRadioGroup('scale');
    setupRadioGroup('upscale-model');

    // Mode change handler (for hiding AI-only settings)
    document.querySelectorAll('input[name="mode"]').forEach(radio => {
        radio.addEventListener('change', handleModeChange);
    });

    // Setup sliders
    setupSlider('strength');
    setupSlider('guidance');
    setupSlider('steps');
    setupSlider('feather');
    setupSlider('expand');

    // Initial mode setup
    handleModeChange();

    // Developer console command to show dev tools
    window.showDevTools = () => {
        elements.devTools.classList.remove('hidden');
        log('Dev tools enabled');
    };

    Files.cleanStaleTemp();
    // Check initial server status — was previously just defaulting to
    // "Offline" on every panel open even when the server was already
    // running from a prior session, since no actual /health call happened.
    isServerOnline().then(online => {
        updateServerStatus(online);
        if (online) attachToServer();
    });
}

// Paths go into ExtendScript as forward-slash strings
function jsxPath(p) {
    return jsxStr(p.replace(/\\/g, '/'));
}

// Safe wrapper for building ExtendScript string literals from JS values.
// Previously dynamic values (paths, comp/layer names) were interpolated
// directly into the evalScript() source string — a `"` or backslash in a
// composition name would break the generated JSX or, worse, let arbitrary
// characters escape the string literal into executable ExtendScript.
// JSON.stringify produces a valid double-quoted JS/ExtendScript string
// literal with proper escaping.
function jsxStr(value) {
    return JSON.stringify(String(value));
}

// Setup radio button group with active class management
function setupRadioGroup(name) {
    const radios = document.querySelectorAll(`input[name="${name}"]`);

    // Update active class based on checked state
    function updateActiveClass() {
        radios.forEach(radio => {
            const label = radio.closest('label');
            if (label) {
                label.classList.toggle('active', radio.checked);
            }
        });
    }

    // Add change listeners
    radios.forEach(radio => {
        radio.addEventListener('change', updateActiveClass);
    });

    // Set initial state
    updateActiveClass();
}

function handleTabSwitch(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tabName}`);
    });
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
    const projectPath = path.dirname(realPath);
    return projectPath;
}

function setupSlider(id) {
    const slider = document.getElementById(id);
    const span = document.getElementById(`${id}-value`);
    if (slider && span) {
        slider.addEventListener('input', () => {
            // Слайдер с data-auto показывает "Auto", пока его не тронули:
            // тогда сервер берёт значение движка (у klein 4 шага, у FLUX.1
            // Fill 28/30) вместо одинакового для всех 20/7.5
            delete slider.dataset.auto;
            span.textContent = slider.value;
        });
    }
}

// Simplified logging for users
function log(msg, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
    entry.textContent = `${time} ${msg}`;
    elements.log.appendChild(entry);
    elements.log.scrollTop = elements.log.scrollHeight;
    while (elements.log.children.length > 30) {
        elements.log.removeChild(elements.log.firstChild);
    }
}

// Verbose logging (only shown when verboseLog is true)
function logVerbose(msg) {
    if (verboseLog) {
        console.log('[verbose]', msg);
        log(msg, 'info');
    } else {
        console.log(msg);
    }
}

function showProgress(text, detail = '') {
    elements.progressOverlay.classList.remove('hidden');
    elements.progressText.textContent = text;
    elements.progressDetail.textContent = detail;
    elements.btnInpaint.disabled = true;
    if (elements.btnUpscale) elements.btnUpscale.disabled = true;
    isProcessing = true;
}

function updateProgress(text, detail = '') {
    elements.progressText.textContent = text;
    elements.progressDetail.textContent = detail;
}

function hideProgress() {
    elements.progressOverlay.classList.add('hidden');
    elements.btnInpaint.disabled = false;
    if (elements.btnUpscale) elements.btnUpscale.disabled = false;
    elements.btnStop.classList.add('hidden');
    elements.btnStopOverlay.classList.add('hidden');
    isProcessing = false;
}

function showStopButton() {
    elements.btnStop.classList.remove('hidden');
    elements.btnStopOverlay.classList.remove('hidden');
}

function updateServerStatus(online = false, loading = false) {
    elements.statusIndicator.classList.remove('online', 'loading');
    if (loading) {
        elements.statusIndicator.classList.add('loading');
        elements.statusText.textContent = 'Loading...';
    } else if (online) {
        elements.statusIndicator.classList.add('online');
        elements.statusText.textContent = 'Online';
    } else {
        elements.statusText.textContent = 'Offline';
    }
}

function handleModeChange() {
    const mode = getMode();
    const aiOnlyElements = document.querySelectorAll('.ai-only');

    aiOnlyElements.forEach(el => {
        if (mode === 'ai') {
            el.classList.remove('mode-hidden');
        } else {
            el.classList.add('mode-hidden');
        }
    });
}

function handleToggleSettings() {
    elements.settingsPanel.classList.toggle('hidden');
    const isHidden = elements.settingsPanel.classList.contains('hidden');
    elements.btnToggleSettings.textContent = isHidden ? 'Settings' : 'Hide Settings';
}

function showFirstRunModal() {
    elements.firstRunModal.classList.remove('hidden');
}

function hideFirstRunModal() {
    elements.firstRunModal.classList.add('hidden');
}

let firstRunResolve = null;

function handleFirstRunContinue() {
    hideFirstRunModal();
    if (firstRunResolve) {
        firstRunResolve(true);
        firstRunResolve = null;
    }
}

async function checkFirstRun() {
    if (getMode() !== 'ai') {
        return true;
    }

    // Check if model is cached on server
    try {
        const health = await API.healthCheck();
        if (health.model_cached) {
            isFirstRun = false;
            return true;
        }
    } catch (e) {
        // Server not running yet, will check later
    }

    // Check localStorage as fallback
    if (!isFirstRun) {
        return true;
    }

    return new Promise((resolve) => {
        firstRunResolve = resolve;
        showFirstRunModal();

        const cancelHandler = () => {
            hideFirstRunModal();
            resolve(false);
            elements.btnFirstRunCancel.removeEventListener('click', cancelHandler);
        };
        elements.btnFirstRunCancel.addEventListener('click', cancelHandler);
    });
}

function markModelDownloaded() {
    try {
        localStorage.setItem(FIRST_RUN_KEY, 'true');
        isFirstRun = false;
    } catch (e) {
        console.error('Failed to save first run flag:', e);
    }
}

async function handleStop() {
    if (!isProcessing) return;
    log('Stopping...');
    upscaleCancelled = true;  // Signal to stop the batch upscale loop between layers

    // Previously this only hid the progress overlay and set a flag that the
    // upscale loop checked between layers — the actual /inpaint fetch (and
    // the server-side generation behind it) kept running to completion, and
    // could still save a file and import a layer *after* the user had
    // already been told "Stopped". Now we actually abort the in-flight
    // request client-side and ask the server to stop generating too.
    API.abortCurrent();
    try {
        await API.cancelJob();
    } catch (e) {
        // Best-effort — server may already be done, or unreachable.
    }

    // Прогресс прячет finally прерванного обработчика: если спрятать здесь,
    // isProcessing сбросится раньше, чем тот завершится, и новую задачу можно
    // запустить поверх ещё не закончившейся старой.
    updateProgress('Stopping...', '');
    // Server process itself stays running (model stays loaded) for the next request.
}

function killOrphanServer() {
    // Kill any leftover process on port 7860 — but only if it's actually
    // our own uvicorn server. Previously this killed whatever process held
    // the port unconditionally, which could be an unrelated app that
    // happened to be using 7860.
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

        logVerbose('Project path: ' + projectPath);

        if (!fs.existsSync(venvPython)) {
            reject(new Error('Python venv not found. Run install.sh first.'));
            return;
        }

        // Kill any orphaned server process holding the port
        killOrphanServer();

        log('Starting server...');
        updateServerStatus(false, true);

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
                reject(new Error('Server start timeout'));
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
                updateServerStatus(true);
                log('Server ready', 'success');
                resolve();
            }
        });

        proc.stdout.on('data', (data) => {
            logVerbose('[server] ' + data.toString().trim());
        });

        proc.on('error', (err) => {
            clearTimeout(timer);
            updateServerStatus(false);
            reject(new Error(`Server error: ${err.message}`));
        });

        proc.on('exit', (code) => {
            clearTimeout(timer);
            if (!started) {
                // Упал при старте (сломанный venv, занятый порт) — сразу
                // ошибка, а не 90 с оверлея без кнопки Stop
                reject(new Error(portBusy
                    ? 'Port 7860 is used by another application — close it and try again'
                    : `Server exited during startup (code ${code}). See ${Files.TEMP_ROOT}/server.log`));
            }
            if (serverProcess === proc) {
                serverProcess = null;
                updateServerStatus(false);
            }
            logVerbose('Server stopped');
        });
    });
}

function stopServer() {
    if (serverProcess) {
        logVerbose('Stopping server...');
        serverProcess.kill('SIGTERM');
        serverProcess = null;
        updateServerStatus(false);
    }
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
async function ensureServer() {
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
    showProgress('Starting server...', 'This may take a moment');
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

function evalScript(script) {
    return new Promise((resolve, reject) => {
        csInterface.evalScript(script, (result) => {
            logVerbose('evalScript result: ' + result?.substring?.(0, 100));

            if (result === 'EvalScript error.') {
                reject(new Error('EvalScript error'));
                return;
            }
            if (result === 'undefined' || result === undefined || result === null) {
                reject(new Error('Result is undefined'));
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

function getMode() {
    const checked = document.querySelector('input[name="mode"]:checked');
    return checked ? checked.value : 'remove';
}

function getSettings() {
    const seed = parseInt(elements.seed.value);
    return {
        strength: parseFloat(elements.strength.value),
        guidance: elements.guidance.dataset.auto ? null : parseFloat(elements.guidance.value),
        steps: elements.steps.dataset.auto ? null : parseInt(elements.steps.value),
        seed: seed === -1 ? null : seed,
        feather: parseInt(elements.feather.value),
        expand: parseInt(elements.expand.value),
        invertMask: elements.invertMask.checked,
        cropToMask: elements.cropToMask.checked,
        fillTransparent: elements.fillTransparent.checked,
        negativePrompt: elements.negativePrompt.value.trim()
    };
}

async function handleInpaint() {
    if (isProcessing) return;
    let jobDir = null;

    const mode = getMode();

    // Check first run for AI Gen mode only (LaMa downloads automatically and is small)
    if (mode === 'ai') {
        const proceed = await checkFirstRun();
        if (!proceed) {
            log('Cancelled');
            return;
        }
    }

    try {
        showProgress('Preparing...', 'Checking server status');

        // Start server if needed
        await ensureServer();

        const modeLabels = { remove: 'Remove', ai: 'AI Generate', clean: 'Classic' };
        log('Starting ' + (modeLabels[mode] || mode) + '...');

        // 1. Project info
        showProgress('Preparing...', 'Getting project info');
        let projectInfo;
        try {
            projectInfo = await evalScript('getProjectInfo()');
        } catch (e) {
            throw new Error('ExtendScript error. Reload panel.');
        }
        if (projectInfo.error) throw new Error(projectInfo.error);
        if (!projectInfo.projectPath) log(`Project not saved — results go to ~/Documents/${Files.RESULTS_FOLDER_NAME}`);
        log(`${projectInfo.compName}, frame ${projectInfo.currentFrame}`);
        // Папку результатов создаём до генерации: на read-only томе ошибка
        // должна прийти сразу, а не после минуты работы модели
        const outputDir = Files.getResultsDir(projectInfo.projectPath);

        // 2. Selected layer with mask
        showProgress('Preparing...', 'Checking layer and mask');
        const layerInfo = await evalScript('getSelectedLayerWithMask()');
        logVerbose('Layer info: ' + JSON.stringify(layerInfo));
        if (layerInfo.error) throw new Error(layerInfo.error);
        log(`Layer: ${layerInfo.name}${layerInfo.noMask ? ' (expand mode)' : ''}`);

        // 3. Export
        jobDir = Files.makeJobDir();
        let imageBase64, maskBase64;

        if (layerInfo.noMask) {
            // No mask mode: export layer frame, server will generate mask from alpha
            showProgress('Exporting...', 'Rendering layer (expand mode)');
            const exportResult = await evalScript(
                `exportLayerFrame(${layerInfo.index}, ${jsxPath(jobDir)})`
            );
            let parsed = typeof exportResult === 'string' ? JSON.parse(exportResult) : exportResult;
            if (parsed.error) throw new Error(parsed.error);

            showProgress('Loading...', 'Reading exported file');
            imageBase64 = await Files.readRenderedPng(parsed.imagePath);
            maskBase64 = '';  // Empty - server generates from alpha
        } else {
            // Has mask: export both image and mask
            showProgress('Exporting...', 'Rendering layer and mask');
            const exportResult = await evalScript(
                `exportForInpaint(${layerInfo.index}, ${layerInfo.selectedMaskIndex}, ${jsxPath(jobDir)})`
            );
            logVerbose('Export result: ' + JSON.stringify(exportResult));
            if (exportResult.error) throw new Error(exportResult.error);

            showProgress('Loading...', 'Reading exported files');
            imageBase64 = await Files.readRenderedPng(exportResult.imagePath);
            maskBase64 = await Files.readRenderedPng(exportResult.maskPath);
            logVerbose(`Image b64: ${imageBase64.length}, Mask b64: ${maskBase64.length}`);
        }

        // 5. Inpaint
        showProgress('Processing...', 'Starting...');
        showStopButton();

        // Start progress polling
        let progressInterval = null;
        if (mode === 'ai') {
            progressInterval = setInterval(async () => {
                try {
                    const prog = await API.getProgress();
                    if (!prog) return;
                    if (prog.stage === 'loading_model') {
                        updateProgress('Loading model...', 'First time takes longer');
                    } else if (prog.stage === 'inpainting') {
                        const step = prog.step || 0;
                        const total = prog.total_steps || 0;
                        if (total > 0) {
                            updateProgress(`Generating... ${step}/${total}`, `Step ${step} of ${total}`);
                        }
                    }
                } catch (e) {}
            }, 500);
        } else {
            const label = mode === 'remove' ? 'Removing...' : 'Processing...';
            updateProgress(label, '');
        }

        const settings = getSettings();
        let result;
        try {
            result = await API.inpaint({
                imageBase64,
                maskBase64,
                mode: mode,
                prompt: elements.prompt.value.trim(),
                settings: settings
            });
        } finally {
            if (progressInterval) clearInterval(progressInterval);
        }

        // Mark model as downloaded after successful AI inference
        if (mode === 'ai') {
            markModelDownloaded();
        }

        log('Done', 'success');

        // 6. Save result
        showProgress('Importing...', 'Saving result file');
        // compName was previously used unsanitized in a filesystem path —
        // AE composition names can contain "/" and other characters that
        // aren't safe there (layer names elsewhere in this file already get
        // sanitized the same way for the same reason, see safeName below).
        const safeCompName = projectInfo.compName.replace(/[^a-zA-Z0-9]/g, '_');
        const resultPath = path.join(outputDir, `${safeCompName}_f${projectInfo.currentFrame}_${Date.now()}.png`);
        await Files.base64ToFile(result.result, resultPath);

        // 7. Import to AE
        showProgress('Importing...', 'Adding layer to composition');
        const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const resultLayerName = `Inpaint ${timestamp}`;
        const importResult = await evalScript(
            `importResultAsLayer(${jsxPath(resultPath)}, ${layerInfo.index}, ${jsxStr(resultLayerName)})`
        );
        if (importResult.error) throw new Error(importResult.error);

        log(`Created: ${importResult.layerName}`, 'success');

    } catch (error) {
        log(error.message, 'error');
    } finally {
        if (jobDir) Files.removeDir(jobDir);
        hideProgress();
    }
}

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
            elements.btnDebugMode.textContent = `CEP Debug: ${newVal === '1' ? 'ON' : 'OFF'}`;
            log(`CEP Debug ${newVal === '1' ? 'enabled' : 'disabled'}. Restart AE.`, 'success');
        });
    });
}

function getUpscaleSettings() {
    const scale = document.querySelector('input[name="scale"]:checked');
    const modelType = document.querySelector('input[name="upscale-model"]:checked');
    return {
        scale: scale ? parseInt(scale.value) : 4,
        modelType: modelType ? modelType.value : 'anime'
    };
}

async function handleUpscale() {
    if (isProcessing) return;
    let jobDir = null;

    upscaleCancelled = false;

    try {
        showProgress('Preparing...', 'Checking server');

        // Start server once
        await ensureServer();

        const settings = getUpscaleSettings();

        // Get project info
        showProgress('Preparing...', 'Getting project info');
        const projectInfo = await evalScript('getProjectInfo()');
        if (projectInfo.error) throw new Error(projectInfo.error);
        if (!projectInfo.projectPath) log(`Project not saved — results go to ~/Documents/${Files.RESULTS_FOLDER_NAME}`);

        jobDir = Files.makeJobDir();
        const outputDir = Files.getResultsDir(projectInfo.projectPath);

        // Get all selected layers
        showProgress('Preparing...', 'Getting layers');
        const layersInfo = await evalScript('getSelectedLayers()');
        if (layersInfo.error) throw new Error(layersInfo.error);

        // Sort by descending index - process bottom layers first
        // This way indices of unprocessed layers don't shift
        const layers = layersInfo.layers.sort((a, b) => b.index - a.index);

        log(`Upscaling ${layers.length} layer(s) x${settings.scale}`);

        let done = 0;

        for (let i = 0; i < layers.length; i++) {
            if (upscaleCancelled) {
                log('Cancelled');
                break;
            }

            const layer = layers[i];
            const num = i + 1;

            log(`[${num}/${layers.length}] ${layer.name}`);
            showProgress(`${num}/${layers.length}`, `Exporting ${layer.name}...`);
            showStopButton();

            // Export using INDEX (captured at start, valid because we go bottom-up)
            let exportResult;
            try {
                let rawResult = await evalScript(
                    `exportLayerFrame(${layer.index}, ${jsxPath(jobDir)})`
                );
                // Handle case where result is still a string
                if (typeof rawResult === 'string') {
                    exportResult = JSON.parse(rawResult);
                } else {
                    exportResult = rawResult;
                }
            } catch (e) {
                log(`Export failed: ${e.message}`, 'error');
                continue;
            }

            if (!exportResult || exportResult.error) {
                log(`Export error: ${exportResult?.error || 'Unknown error'}`, 'error');
                continue;
            }

            if (!exportResult.imagePath) {
                log(`Export error: No image path`, 'error');
                continue;
            }

            if (upscaleCancelled) break;

            // Read file
            const imageBase64 = await Files.readRenderedPng(exportResult.imagePath);
            console.log(`Read ${exportResult.imagePath}: ${imageBase64?.length || 0} bytes`);

            if (upscaleCancelled) break;

            // Upscale
            showProgress(`${num}/${layers.length}`, `Upscaling ${layer.name}...`);
            let result;
            try {
                result = await API.upscale({
                    imageBase64,
                    scale: settings.scale,
                    modelType: settings.modelType
                });
            } catch (e) {
                log(`Upscale error: ${e.message}`, 'error');
                continue;
            }

            if (upscaleCancelled) break;

            // Save result (with timestamp to avoid stale files)
            const safeName = layer.name.replace(/[^a-zA-Z0-9]/g, '_');
            const resultPath = path.join(outputDir, `${safeName}_x${settings.scale}_${Date.now()}.png`);
            await Files.base64ToFile(result.result, resultPath);

            // Import - use the INDEX from export result (current position)
            showProgress(`${num}/${layers.length}`, `Importing...`);
            const importResult = await evalScript(
                `importResultAsLayer(${jsxPath(resultPath)}, ${exportResult.layerIndex}, ${jsxStr(layer.name + ' x' + settings.scale)}, ${100 / settings.scale})`
            );
            if (importResult.error) {
                log(`Import error: ${importResult.error}`, 'error');
                continue;
            }

            done++;

            // Small delay for AE to update
            await new Promise(r => setTimeout(r, 200));
        }

        log(`Done! ${done}/${layers.length} upscaled`, 'success');

    } catch (error) {
        console.error('handleUpscale error:', error);
        log(error.message, 'error');
    } finally {
        if (jobDir) Files.removeDir(jobDir);
        hideProgress();
    }
}

document.addEventListener('DOMContentLoaded', init);
