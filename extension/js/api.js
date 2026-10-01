/**
 * API клиент для общения с Python сервером инпейнтинга
 */

const API = {
    baseUrl: 'http://127.0.0.1:7860',
    timeout: 1800000, // 30 минут для первого запуска (загрузка модели)

    // AbortController for whatever /inpaint or /upscale request is currently
    // in flight, so handleStop() in main.js can actually cancel it. Fetch's
    // own `{ timeout: N }` option (used below to previously "time out"
    // getProgress/healthCheck) doesn't exist in the Fetch API and was
    // silently ignored — real timeouts need an AbortController too.
    _activeController: null,

    /**
     * POST, который при 409 (сервер ещё доделывает прошлую задачу — например,
     * LaMa после Stop дорабатывает пару секунд) немного ждёт и повторяет,
     * вместо ошибки "Server is busy" сразу после Stop.
     */
    async _postWhenFree(path, body, signal) {
        const payload = JSON.stringify(body);
        for (let attempt = 0; ; attempt++) {
            const response = await fetch(`${this.baseUrl}${path}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: payload,
                signal
            });
            if (response.status !== 409 || attempt >= 60) return response;  // до ~30 с
            await new Promise(r => setTimeout(r, 500));
        }
    },

    /** Есть ли у этой панели запрос /inpaint или /upscale в работе */
    hasActiveRequest() {
        return this._activeController !== null;
    },

    /**
     * Прерывает текущий активный запрос /inpaint или /upscale, если есть.
     */
    abortCurrent() {
        if (this._activeController) {
            this._activeController.abort();
        }
    },

    /**
     * Просит сервер остановить текущую job (проверяется между шагами
     * генерации). Best-effort — не гарантирует мгновенную остановку.
     */
    async cancelJob() {
        const response = await fetch(`${this.baseUrl}/cancel`, { method: 'POST' });
        if (!response.ok) throw new Error('Failed to cancel job');
        return await response.json();
    },

    /**
     * Получить прогресс текущей операции
     */
    async getProgress() {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);
        try {
            const response = await fetch(`${this.baseUrl}/progress`, {
                method: 'GET',
                signal: controller.signal
            });
            if (!response.ok) return null;
            return await response.json();
        } catch {
            return null;
        } finally {
            clearTimeout(timeoutId);
        }
    },

    /**
     * Проверка здоровья сервера
     */
    async healthCheck() {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                signal: controller.signal
            });
            if (!response.ok) throw new Error('Server unhealthy');
            const health = await response.json();
            // На 7860 может висеть чужое приложение (Gradio и т.п.)
            if (health.status !== 'ok' || !('engine' in health)) {
                throw new Error('Port 7860 is used by another application');
            }
            return health;
        } catch (error) {
            throw new Error(`Server unavailable: ${error.message}`);
        } finally {
            clearTimeout(timeoutId);
        }
    },

    /**
     * Регистрирует панель на сервере (см. attachToServer в main.js)
     */
    async attach(pid) {
        const response = await fetch(`${this.baseUrl}/attach`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pid })
        });
        if (response.status === 503) return { status: 'shutting_down' };
        if (!response.ok) throw new Error(`attach failed: ${response.status}`);
        return await response.json();
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
     * @param {string} params.mode - Режим: 'ai' или 'clean'
     * @param {string} params.prompt - Текстовый промпт
     * @param {Object} params.settings - Настройки (strength, guidance, etc.)
     */
    async inpaint({ imageBase64, maskBase64, mode, prompt, settings }) {
        const body = {
            image: imageBase64,
            mask: maskBase64,
            mode: mode || 'ai',
            prompt: prompt || '',
            negative_prompt: settings.negativePrompt || '',
            strength: settings.strength || 1.0,
            // null — значения движка по умолчанию
            guidance_scale: Number.isFinite(settings.guidance) ? settings.guidance : null,
            num_steps: Number.isFinite(settings.steps) ? settings.steps : null,
            controlnet_scale: settings.controlnetScale || 0.5,
            // || превращал seed 0 в «случайный»
            seed: Number.isFinite(settings.seed) ? settings.seed : null,
            feather: settings.feather || 0,
            expand: settings.expand || 0,
            crop_to_mask: settings.cropToMask !== false,  // default true
            crop_padding: 128,
            invert_mask: settings.invertMask || false,
            fill_transparent: settings.fillTransparent || false
        };

        const controller = new AbortController();
        this._activeController = controller;
        let timedOut = false;
        const timeoutId = setTimeout(() => { timedOut = true; controller.abort(); }, this.timeout);

        try {
            const response = await this._postWhenFree('/inpaint', body, controller.signal);

            clearTimeout(timeoutId);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Inpaint failed');
            }

            return await response.json();

        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                // Same controller is used for the timeout deadline and for
                // abortCurrent() (Stop button) — distinguish which one fired.
                throw new Error(timedOut ? 'Request timeout - inference took too long' : 'Cancelled');
            }
            throw error;
        } finally {
            if (this._activeController === controller) this._activeController = null;
        }
    },

    /**
     * Апскейл изображения
     * @param {Object} params
     * @param {string} params.imageBase64 - Base64 PNG изображения
     * @param {number} params.scale - Множитель масштаба (2 или 4)
     * @param {string} params.modelType - Тип модели: 'anime' или 'general'
     */
    async upscale({ imageBase64, scale, modelType }) {
        const body = {
            image: imageBase64,
            scale: scale || 4,
            model_type: modelType || 'anime'
        };

        const controller = new AbortController();
        this._activeController = controller;
        let timedOut = false;
        const timeoutId = setTimeout(() => { timedOut = true; controller.abort(); }, this.timeout);

        try {
            const response = await this._postWhenFree('/upscale', body, controller.signal);

            clearTimeout(timeoutId);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upscale failed');
            }

            return await response.json();

        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error(timedOut ? 'Request timeout - upscale took too long' : 'Cancelled');
            }
            throw error;
        } finally {
            if (this._activeController === controller) this._activeController = null;
        }
    }
};
