/**
 * API клиент для общения с Python сервером инпейнтинга
 */

const API = {
    baseUrl: 'http://127.0.0.1:7860',
    timeout: 1800000, // 30 минут для первого запуска (загрузка модели)

    /**
     * Проверка здоровья сервера
     */
    async healthCheck() {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                timeout: 5000
            });
            if (!response.ok) throw new Error('Server unhealthy');
            return await response.json();
        } catch (error) {
            throw new Error(`Server unavailable: ${error.message}`);
        }
    },

    /**
     * Загрузка модели
     */
    async loadModel() {
        const response = await fetch(`${this.baseUrl}/load`, {
            method: 'POST'
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load model');
        }
        return await response.json();
    },

    /**
     * Выгрузка модели
     */
    async unloadModel() {
        const response = await fetch(`${this.baseUrl}/unload`, {
            method: 'POST'
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to unload model');
        }
        return await response.json();
    },

    /**
     * Инпейнтинг
     * @param {Object} params
     * @param {string} params.imageBase64 - Base64 PNG изображения
     * @param {string} params.maskBase64 - Base64 PNG маски
     * @param {string} params.prompt - Текстовый промпт
     * @param {Object} params.settings - Настройки (strength, guidance, etc.)
     * @param {string} params.cacheDir - Путь к папке кэша
     */
    async inpaint({ imageBase64, maskBase64, prompt, settings, cacheDir }) {
        const body = {
            image: imageBase64,
            mask: maskBase64,
            prompt: prompt || '',
            negative_prompt: settings.negativePrompt || '',
            strength: settings.strength || 0.85,
            guidance_scale: settings.guidance || 7.5,
            num_steps: settings.steps || 30,
            controlnet_scale: settings.controlnetScale || 0.5,
            seed: settings.seed || null,
            cache_dir: cacheDir || null
        };

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const response = await fetch(`${this.baseUrl}/inpaint`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Inpaint failed');
            }

            return await response.json();

        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Request timeout - inference took too long');
            }
            throw error;
        }
    },

    /**
     * Очистка кэша
     */
    async clearCache(cacheDir) {
        const response = await fetch(`${this.baseUrl}/clear-cache?cache_dir=${encodeURIComponent(cacheDir)}`, {
            method: 'POST'
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to clear cache');
        }
        return await response.json();
    }
};

/**
 * Конвертация файла в Base64
 */
async function fileToBase64(filePath) {
    const fs = require('fs');

    // Wait a bit for file to be fully written by ExtendScript
    await new Promise(r => setTimeout(r, 500));

    // Use synchronous reading to avoid race conditions
    let attempts = 0;
    const maxAttempts = 10;

    while (attempts < maxAttempts) {
        attempts++;

        try {
            const stats = fs.statSync(filePath);
            if (stats.size === 0) {
                console.log(`File empty, retry ${attempts}/${maxAttempts}: ${filePath}`);
                await new Promise(r => setTimeout(r, 300));
                continue;
            }

            console.log(`Reading file: ${filePath} (${stats.size} bytes)`);

            // Synchronous read to avoid race conditions
            const buffer = fs.readFileSync(filePath);

            // Verify size matches
            if (buffer.length !== stats.size) {
                console.error(`Read mismatch: got ${buffer.length}, expected ${stats.size}, retrying...`);
                await new Promise(r => setTimeout(r, 300));
                continue;
            }

            const b64 = buffer.toString('base64');
            console.log(`Buffer size: ${buffer.length}, Base64 length: ${b64.length}`);
            return b64;

        } catch (e) {
            console.log(`File error, retry ${attempts}/${maxAttempts}: ${e.message}`);
            await new Promise(r => setTimeout(r, 300));
        }
    }

    throw new Error(`Failed to read file after ${maxAttempts} attempts: ${filePath}`);
}

/**
 * Сохранение Base64 в файл
 */
async function base64ToFile(base64Data, filePath) {
    return new Promise((resolve, reject) => {
        const fs = require('fs');
        const path = require('path');

        // Create directory if it doesn't exist
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }

        const buffer = Buffer.from(base64Data, 'base64');
        fs.writeFile(filePath, buffer, (err) => {
            if (err) {
                reject(err);
                return;
            }
            resolve(filePath);
        });
    });
}
