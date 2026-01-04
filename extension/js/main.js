/**
 * AE Inpaint Panel - Main Logic
 */

const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let csInterface;
let isProcessing = false;
let extensionPath = null;
let serverProcess = null;

const elements = {};

function loadJSX() {
    // Convert file:// URL to regular path
    let extPath = extensionPath;
    if (extPath.startsWith('file://')) {
        extPath = decodeURIComponent(extPath.replace('file://', ''));
    }

    // Resolve real path (for symlinks)
    let jsxPath = path.join(extPath, 'jsx', 'host.jsx');
    try {
        jsxPath = fs.realpathSync(jsxPath);
    } catch (e) {
        console.error('Symlink resolve failed:', e);
    }

    console.log('Loading JSX from:', jsxPath);

    try {
        let jsxContent = fs.readFileSync(jsxPath, 'utf8');

        // Remove BOM if present
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

    // Cache elements first
    elements.btnInpaint = document.getElementById('btn-inpaint');
    elements.btnStop = document.getElementById('btn-stop');
    elements.btnToggleSettings = document.getElementById('btn-toggle-settings');
    elements.btnDebug = document.getElementById('btn-debug');
    elements.btnClearCache = document.getElementById('btn-clear-cache');
    elements.btnDebugMode = document.getElementById('btn-debug-mode');
    elements.settingsPanel = document.getElementById('settings-panel');
    elements.prompt = document.getElementById('prompt');
    elements.log = document.getElementById('log');
    elements.strength = document.getElementById('strength');
    elements.guidance = document.getElementById('guidance');
    elements.steps = document.getElementById('steps');

    // Load jsx manually (symlink fix)
    loadJSX();

    // Test basic ExtendScript first
    csInterface.evalScript('app.version', (result) => {
        console.log('AE version:', result);
        log('AE: ' + result, 'info');
    });

    // Verify JSX loaded after a brief delay
    setTimeout(() => {
        csInterface.evalScript('typeof getProjectInfo', (result) => {
            console.log('getProjectInfo type:', result);
            if (result === 'function') {
                log('JSX loaded', 'success');
            } else {
                log('JSX failed: ' + result, 'error');
            }
        });
    }, 1000);

    // Event handlers
    elements.btnInpaint.addEventListener('click', handleInpaint);
    elements.btnStop.addEventListener('click', handleStop);
    elements.btnToggleSettings.addEventListener('click', handleToggleSettings);
    elements.btnDebug.addEventListener('click', handleDebugExport);
    elements.btnClearCache.addEventListener('click', handleClearCache);
    elements.btnDebugMode.addEventListener('click', handleToggleDebugMode);

    setupSlider('strength');
    setupSlider('guidance');

    log('Ready', 'info');
}

function getProjectPath() {
    // Convert file:// URL to regular path
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
    console.log('extensionPath:', extensionPath);
    console.log('extPath:', extPath);
    console.log('realPath:', realPath);
    console.log('projectPath:', projectPath);
    return projectPath;
}

function setupSlider(id) {
    const slider = document.getElementById(id);
    const span = document.getElementById(`${id}-value`);
    slider.addEventListener('input', () => span.textContent = slider.value);
}

function log(msg, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString('en-US', {hour12: false})}] ${msg}`;
    elements.log.appendChild(entry);
    elements.log.scrollTop = elements.log.scrollHeight;
    while (elements.log.children.length > 50) elements.log.removeChild(elements.log.firstChild);
}

function showProgress(text) {
    // Just log status, don't block UI
    log('>> ' + text, 'info');
    elements.btnInpaint.disabled = true;
    elements.btnInpaint.textContent = text;
    isProcessing = true;
}

function hideProgress() {
    elements.btnInpaint.disabled = false;
    elements.btnInpaint.textContent = 'Inpaint';
    elements.btnStop.classList.add('hidden');
    isProcessing = false;
}

function showStopButton() {
    elements.btnStop.classList.remove('hidden');
}

async function handleStop() {
    if (!isProcessing) return;
    log('Stopping...', 'info');
    // Kill the server to stop inference
    stopServer();
    hideProgress();
    log('Stopped', 'info');
}


// Start server and wait for it to be ready
function startServer() {
    return new Promise((resolve, reject) => {
        const projectPath = getProjectPath();
        const venvPython = path.join(projectPath, '.venv', 'bin', 'python');

        log('Project path: ' + projectPath, 'info');
        log('venv path: ' + venvPython, 'info');

        if (!fs.existsSync(venvPython)) {
            log('venv check failed, exists: ' + fs.existsSync(projectPath), 'error');
            reject(new Error('venv not found. Run install.sh first.'));
            return;
        }

        log('Starting server...', 'info');

        serverProcess = spawn(venvPython, ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '7860'], {
            cwd: path.join(projectPath, 'server'),
            env: { ...process.env, PYTHONUNBUFFERED: '1' }
        });

        let started = false;

        serverProcess.stderr.on('data', (data) => {
            const msg = data.toString();
            if (!started && (msg.includes('Uvicorn running') || msg.includes('Application startup complete'))) {
                started = true;
                log('Server online', 'success');
                resolve();
            }
        });

        serverProcess.on('error', (err) => {
            reject(new Error(`Server error: ${err.message}`));
        });

        serverProcess.on('exit', (code) => {
            serverProcess = null;
            log('Server stopped', 'info');
        });

        // Timeout after 30 seconds
        setTimeout(() => {
            if (!started) {
                reject(new Error('Server start timeout'));
            }
        }, 30000);
    });
}

function stopServer() {
    if (serverProcess) {
        log('Stopping server...', 'info');
        serverProcess.kill('SIGTERM');
        serverProcess = null;
    }
}

async function isServerOnline() {
    try {
        await API.healthCheck();
        return true;
    } catch {
        return false;
    }
}

function handleToggleSettings() {
    elements.settingsPanel.classList.toggle('hidden');
    elements.btnToggleSettings.textContent = elements.settingsPanel.classList.contains('hidden') ? 'Settings' : 'Hide Settings';
}

function evalScript(script) {
    return new Promise((resolve, reject) => {
        csInterface.evalScript(script, (result) => {
            console.log('evalScript [' + script.substring(0, 30) + '...] result:', result);

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
                // Not JSON - return as is
                resolve(result);
            }
        });
    });
}

function getSettings() {
    return {
        strength: parseFloat(elements.strength.value),
        guidance: parseFloat(elements.guidance.value),
        steps: parseInt(elements.steps.value)
    };
}

async function handleInpaint() {
    if (isProcessing) return;

    try {
        showProgress('Preparing...');

        // Start server if needed
        if (!(await isServerOnline())) {
            showProgress('Starting server...');
            await startServer();
            await new Promise(r => setTimeout(r, 1000));
        }

        log('Starting inpaint...', 'info');

        // 1. Project info
        let projectInfo;
        try {
            projectInfo = await evalScript('getProjectInfo()');
        } catch (e) {
            log('getProjectInfo error: ' + e.message, 'error');
            throw new Error('ExtendScript error. Check log.');
        }
        if (projectInfo.error) throw new Error(projectInfo.error);
        if (!projectInfo.projectPath) throw new Error('Save project first.');
        log(`Comp: ${projectInfo.compName}, Frame: ${projectInfo.currentFrame}`, 'info');

        // 2. Selected layer with mask
        const layerInfo = await evalScript('getSelectedLayerWithMask()');
        log('Layer info: ' + JSON.stringify(layerInfo), 'info');
        if (layerInfo.error) throw new Error(layerInfo.error);
        log(`Layer: ${layerInfo.name}, Mask: ${layerInfo.selectedMaskName}`, 'info');

        // 3. Export
        showProgress('Exporting...');
        const cacheDir = projectInfo.projectPath + '/_AI_CACHE';
        log('Cache dir: ' + cacheDir, 'info');
        const exportResult = await evalScript(
            `exportForInpaint(${layerInfo.index}, ${layerInfo.selectedMaskIndex}, "${cacheDir.replace(/\\/g, '/')}")`
        );
        log('Export result: ' + JSON.stringify(exportResult), 'info');
        if (exportResult.error) throw new Error(exportResult.error);

        // 5. Load images
        showProgress('Loading...');
        const imageBase64 = await fileToBase64(exportResult.imagePath);
        const maskBase64 = await fileToBase64(exportResult.maskPath);
        log(`Image b64: ${imageBase64.length}, Mask b64: ${maskBase64.length}`, 'info');

        // 6. Inpaint
        showProgress('AI processing...');
        showStopButton();
        log('Running inference...', 'info');

        const result = await API.inpaint({
            imageBase64,
            maskBase64,
            prompt: elements.prompt.value.trim(),
            settings: getSettings(),
            cacheDir: projectInfo.projectPath
        });

        log(result.cached ? 'From cache' : 'Inference done', 'success');

        // 7. Save result
        showProgress('Importing...');
        const outputDir = projectInfo.projectPath + '/_AI_OUT';
        const resultPath = `${outputDir}/${projectInfo.compName}_frame${projectInfo.currentFrame}_result.png`;
        await base64ToFile(result.result, resultPath);

        // 8. Import to AE
        const importResult = await evalScript(
            `importResultAsLayer("${resultPath.replace(/\\/g, '/')}", ${layerInfo.index}, "Inpaint Result")`
        );
        if (importResult.error) throw new Error(importResult.error);

        log(`Done: ${importResult.layerName}`, 'success');

    } catch (error) {
        log(`Error: ${error.message}`, 'error');
    } finally {
        hideProgress();
        // Stop server after inpaint
        stopServer();
    }
}

async function handleDebugExport() {
    try {
        log('Debug export...', 'info');
        const projectInfo = await evalScript('getProjectInfo()');
        if (projectInfo.error) throw new Error(projectInfo.error);

        const layerInfo = await evalScript('getSelectedLayerWithMask()');
        if (layerInfo.error) throw new Error(layerInfo.error);

        const cacheDir = projectInfo.projectPath + '/_AI_CACHE';
        const exportResult = await evalScript(
            `exportForInpaint(${layerInfo.index}, ${layerInfo.selectedMaskIndex}, "${cacheDir.replace(/\\/g, '/')}")`
        );
        if (exportResult.error) throw new Error(exportResult.error);

        log(`Image: ${exportResult.imagePath}`, 'success');
        log(`Mask: ${exportResult.maskPath}`, 'success');
    } catch (error) {
        log(`Error: ${error.message}`, 'error');
    }
}

async function handleClearCache() {
    try {
        const projectInfo = await evalScript('getProjectInfo()');
        if (projectInfo.error) throw new Error(projectInfo.error);
        await API.clearCache(projectInfo.projectPath);
        log('Cache cleared', 'success');
    } catch (error) {
        log(`Error: ${error.message}`, 'error');
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
            log(`CEP Debug: ${newVal === '1' ? 'ON' : 'OFF'}. Restart AE.`, 'success');
        });
    });
}

document.addEventListener('DOMContentLoaded', init);
