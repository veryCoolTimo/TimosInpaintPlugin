/**
 * Файлы плагина: временные папки задач, папка результатов, чтение PNG,
 * которые рендерит AE.
 *
 * Временные кадры/маски живут в системном temp, в отдельной папке на каждую
 * задачу, и удаляются в finally — даже при ошибке или Stop. Раньше они
 * копились в {проект}/_AI_CACHE и не удалялись при неудаче/expand/апскейле.
 *
 * Обёрнуто в IIFE: в CEP все <script> делят одну глобальную область, и
 * top-level `const fs` здесь конфликтовал бы с main.js. Экспортируется как
 * window.Files в панели и через module.exports — для тестов в Node.
 */
(function (root) {
    const fs = require('fs');
    const os = require('os');
    const path = require('path');

    const TEMP_ROOT = path.join(os.tmpdir(), 'ae-inpaint');
    // Результаты должны жить, пока на них ссылается проект AE, поэтому они
    // лежат рядом с проектом в одной понятной папке (у несохранённого
    // проекта — в Documents)
    const RESULTS_FOLDER_NAME = 'AE Inpaint Results';
    const STALE_AGE_MS = 24 * 3600 * 1000;

    function makeJobDir(tempRoot = TEMP_ROOT) {
        const dir = path.join(tempRoot, `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
        fs.mkdirSync(dir, { recursive: true });
        return dir;
    }

    function removeDir(dir) {
        try {
            fs.rmSync(dir, { recursive: true, force: true });
        } catch (e) {
            console.log('Temp cleanup failed:', dir, e.message);
        }
    }

    // Подчищает папки задач, оставшиеся от прошлых сессий (крэш AE и т.п.).
    // Свежие не трогает — их может использовать вторая открытая панель.
    function cleanStaleTemp(tempRoot = TEMP_ROOT, maxAgeMs = STALE_AGE_MS) {
        let names;
        try {
            names = fs.readdirSync(tempRoot);
        } catch (e) {
            return;  // папки ещё нет — нечего чистить
        }
        const cutoff = Date.now() - maxAgeMs;
        for (const name of names) {
            const dir = path.join(tempRoot, name);
            try {
                const stat = fs.statSync(dir);
                if (stat.isDirectory() && stat.mtimeMs < cutoff) removeDir(dir);
            } catch (e) {
                // удалили параллельно — ок
            }
        }
    }

    function getResultsDir(projectPath, homeDir = os.homedir()) {
        const base = projectPath || path.join(homeDir, 'Documents');
        const dir = path.join(base, RESULTS_FOLDER_NAME);
        fs.mkdirSync(dir, { recursive: true });
        return dir;
    }

    // PNG полный, если начинается с сигнатуры и заканчивается IEND-чанком
    function isPngComplete(buffer) {
        const signature = Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        if (buffer.length < 20 || !buffer.subarray(0, 8).equals(signature)) {
            return false;
        }
        const iend = Buffer.from([0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82]);
        return buffer.subarray(-8).equals(iend);
    }

    /**
     * Читает PNG, который только что отрендерил AE, и возвращает Base64.
     *
     * saveFrameToPng может вернуться раньше, чем файл дописан, поэтому ждём,
     * пока PNG станет полным. Раньше здесь стояли жёсткие sleep 2 с + `sync`
     * + ещё секунда на проверку размера — около 3 с простоя на КАЖДЫЙ файл,
     * даже когда он давно готов; теперь первая попытка сразу, а пауза растёт
     * от 50 мс.
     */
    async function readRenderedPng(filePath, timeoutMs = 60000) {
        const started = Date.now();
        let delay = 50;
        let lastError = null;

        for (;;) {
            try {
                const buffer = fs.readFileSync(filePath);
                if (isPngComplete(buffer)) {
                    return buffer.toString('base64');
                }
            } catch (e) {
                lastError = e;  // файла ещё нет
            }
            if (Date.now() - started >= timeoutMs) break;
            await new Promise(r => setTimeout(r, delay));
            delay = Math.min(delay * 2, 500);
        }

        throw new Error(`Rendered file not ready after ${timeoutMs / 1000}s: ${filePath}` +
            (lastError ? ` (${lastError.message})` : ''));
    }

    async function base64ToFile(base64Data, filePath) {
        await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
        await fs.promises.writeFile(filePath, Buffer.from(base64Data, 'base64'));
        return filePath;
    }

    const Files = {
        TEMP_ROOT,
        RESULTS_FOLDER_NAME,
        makeJobDir,
        removeDir,
        cleanStaleTemp,
        getResultsDir,
        isPngComplete,
        readRenderedPng,
        base64ToFile,
    };

    root.Files = Files;
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = Files;
    }
})(typeof window !== 'undefined' ? window : globalThis);
