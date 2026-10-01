/**
 * UI панели: тема AE, вкладки, переключатели, "горячие" числа, статус,
 * журнал. Не знает ни про AE, ни про сервер — main.js вызывает UI.* и
 * подписывается на кнопки через UI.init(handlers). Поэтому панель можно
 * открыть и проверить в обычном браузере (без CEP/Node).
 *
 * В IIFE, чтобы не делить глобальную область с main.js (см. files.js).
 */
(function (root) {
    const STORAGE_KEY = 'ae_inpaint_ui_v1';

    const MODE_HINTS = {
        remove: 'Erases the masked area using the surroundings. Fast, no prompt.',
        ai: 'Paints new content into the mask (FLUX.2 klein). Without a mask, ' +
            'fills the layer\'s transparent area.',
        clean: 'Instant classic fill. Good for small specks and simple textures.',
    };

    const $ = (id) => document.getElementById(id);
    let state = {};

    // ---------- persistence (per-user convenience only) ----------

    function load() {
        try {
            state = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') || {};
        } catch (e) {
            state = {};
        }
    }

    function save() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        } catch (e) {
            // приватный режим / нет доступа — просто не запоминаем
        }
    }

    // ---------- theme ----------

    function hex(c) {
        const h = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
        return `#${h(c.r)}${h(c.g)}${h(c.b)}`;
    }

    function rgb(h) {
        return { r: parseInt(h.slice(1, 3), 16), g: parseInt(h.slice(3, 5), 16), b: parseInt(h.slice(5, 7), 16) };
    }

    function mix(a, b, t) {
        return { r: a.r + (b.r - a.r) * t, g: a.g + (b.g - a.g) * t, b: a.b + (b.b - a.b) * t };
    }

    function shade(c, d) {
        return { r: c.r + d, g: c.g + d, b: c.b + d };
    }

    // Контраст по WCAG
    function contrast(a, b) {
        const lum = (c) => {
            const ch = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
            return 0.2126 * ch(c.r) + 0.7152 * ch(c.g) + 0.0722 * ch(c.b);
        };
        const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
        return (hi + 0.05) / (lo + 0.05);
    }

    // Первый оттенок по порядку предпочтения с контрастом ≥ 4.5 (WCAG AA);
    // нет такого — первый с ≥ 3 (на средних серых AE иначе ошибка
    // становилась белой и переставала выглядеть ошибкой); нет и такого —
    // самый контрастный. "Самый контрастный" сразу делал синий и красный на
    // тёмном фоне почти белыми
    function readable(bg, candidates, min = 4.5) {
        const colors = candidates.map(rgb);
        return colors.find(c => contrast(c, bg) >= min)
            || colors.find(c => contrast(c, bg) >= Math.min(min, 3))
            || [WHITE, BLACK].reduce((best, c) => (contrast(c, bg) > contrast(best, bg) ? c : best));
    }

    const WHITE = { r: 255, g: 255, b: 255 };
    const BLACK = { r: 0, g: 0, b: 0 };

    // Подстраивает палитру под яркость интерфейса AE (Preferences >
    // Appearance): цвета текста смешиваются с фоном, синий и красный
    // выбираются из оттенков по лучшему контрасту — на средних серых
    // (#5d, #8c) фиксированные цвета почти не читались
    function applyTheme(skin) {
        const color = skin && skin.panelBackgroundColor && skin.panelBackgroundColor.color;
        if (!color) return;
        const bg = { r: color.red, g: color.green, b: color.blue };
        const light = contrast(BLACK, bg) > contrast(WHITE, bg);
        const ink = light ? BLACK : WHITE;
        // Доля смешивания с белым/чёрным растёт, пока не наберётся нужный
        // контраст: на средних серых фиксированные доли его не давали
        const tone = (start, min) => {
            let t = start;
            while (t < 1 && contrast(mix(bg, ink, t), bg) < min) t += 0.02;
            return hex(mix(bg, ink, Math.min(1, t)));
        };
        const s = document.documentElement.style;
        s.setProperty('--bg', hex(bg));
        s.setProperty('--surface', hex(shade(bg, light ? -10 : 8)));
        s.setProperty('--surface-hover', hex(shade(bg, light ? -20 : 16)));
        s.setProperty('--field', hex(shade(bg, light ? 18 : -9)));
        s.setProperty('--line', hex(mix(bg, ink, 0.18)));
        s.setProperty('--text', tone(light ? 0.78 : 0.75, 4.5));
        s.setProperty('--text-strong', tone(0.92, 7));
        s.setProperty('--text-dim', tone(light ? 0.58 : 0.52, 3));
        s.setProperty('--hot', hex(readable(bg, ['#4ca2f5', '#6cb4ff', '#8cc4ff', '#0b5cad', '#084a8c', '#063a70', '#03264a', '#cfe5ff'])));
        s.setProperty('--danger', hex(readable(bg, ['#ff6b66', '#ff7b72', '#ff8f8a', '#b3261e', '#8f1a14', '#6b120d', '#4a0b07', '#ffb3b0'])));
        s.setProperty('--ok', hex(readable(bg, ['#4caf6e', '#2e7d4a', '#1f5c35'], 3)));
    }

    function initTheme() {
        if (typeof CSInterface === 'undefined') return;
        try {
            // Поставляемый CSInterface.js — урезанный: нет свойства
            // hostEnvironment и константы THEME_COLOR_CHANGED_EVENT, есть
            // только getHostEnvironment()
            const cs = new CSInterface();
            const skin = () => cs.getHostEnvironment().appSkinInfo;
            applyTheme(skin());
            cs.addEventListener('com.adobe.csxs.events.ThemeColorChanged', () => applyTheme(skin()));
        } catch (e) {
            console.log('Theme sync unavailable:', e.message);
        }
    }

    // ---------- tabs & segmented controls ----------

    function selectTab(name) {
        document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
        state.tab = name;
        save();
    }

    function segmentedValue(id) {
        const active = $(id).querySelector('button.active');
        return active ? active.dataset.value : null;
    }

    function setSegmented(id, value) {
        const buttons = $(id).querySelectorAll('button');
        if (![...buttons].some(b => b.dataset.value === value)) return;
        buttons.forEach(b => {
            b.classList.toggle('active', b.dataset.value === value);
            b.setAttribute('aria-checked', String(b.dataset.value === value));
        });
    }

    function initSegmented(id, onChange) {
        const el = $(id);
        if (state[id] != null) setSegmented(id, state[id]);
        el.querySelectorAll('button').forEach(b => {
            b.setAttribute('role', 'radio');
            b.addEventListener('click', () => {
                if (b.disabled) return;
                setSegmented(id, b.dataset.value);
                state[id] = b.dataset.value;
                save();
                if (onChange) onChange(b.dataset.value);
            });
        });
        setSegmented(id, segmentedValue(id));
    }

    // Кнопка называется действием режима — вкладка "Fill", а кнопка
    // "Inpaint" путали
    const ACTION_LABELS = { remove: 'Remove', ai: 'Generate', clean: 'Fill' };

    function applyMode(mode) {
        const gen = mode === 'ai';
        document.querySelectorAll('.gen-only').forEach(el => { el.hidden = !gen; });
        $('mode-hint').textContent = MODE_HINTS[mode] || '';
        $('btn-inpaint').textContent = ACTION_LABELS[mode] || 'Fill';
    }

    // ---------- hot text numbers ----------
    // Как в Timeline AE: тянуть мышью влево/вправо — менять значение
    // (Shift — ×10), клик — ввести с клавиатуры. Пустое значение для полей с
    // data-empty означает "по умолчанию" (Auto/Random).

    function hotValue(el) {
        const raw = el.dataset.value;
        return raw === '' || raw == null ? null : Number(raw);
    }

    function decimals(step) {
        const s = String(step);
        return s.includes('.') ? s.split('.')[1].length : 0;
    }

    function renderHot(el) {
        const v = hotValue(el);
        const empty = v === null;
        el.classList.toggle('empty', empty);
        el.textContent = empty ? (el.dataset.empty || '—')
            : v.toFixed(decimals(el.dataset.step || 1)) + (el.dataset.unit || '');
    }

    function setHot(el, value) {
        const allowEmpty = 'empty' in el.dataset;
        if (value === null || value === '' || Number.isNaN(value)) {
            el.dataset.value = allowEmpty ? '' : el.dataset.default;
        } else {
            const min = Number(el.dataset.min), max = Number(el.dataset.max);
            const step = Number(el.dataset.step || 1);
            const snapped = Math.round(value / step) * step;
            el.dataset.value = String(Math.min(max, Math.max(min, Number(snapped.toFixed(decimals(step))))));
        }
        renderHot(el);
        state.hot = state.hot || {};
        state.hot[el.id] = el.dataset.value;
        save();
    }

    function editHot(el) {
        const input = document.createElement('input');
        input.className = 'hot-input';
        input.value = el.dataset.value || '';
        input.placeholder = el.dataset.empty || '';
        el.hidden = true;
        el.after(input);
        input.focus();
        input.select();

        let done = false;
        const finish = (commit) => {
            if (done) return;
            done = true;
            if (commit) {
                const text = input.value.trim();
                const value = text === '' ? null : Number(text.replace(',', '.'));
                // Опечатка ("abc") не сбрасывает поле — остаётся прежнее значение
                if (!Number.isNaN(value)) setHot(el, value);
            }
            input.remove();
            el.hidden = false;
        };
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') finish(true);
            if (e.key === 'Escape') finish(false);
        });
        input.addEventListener('blur', () => finish(true));
    }

    function initHot(el) {
        const saved = state.hot && state.hot[el.id];
        el.dataset.value = el.dataset.default;
        // Через setHot: сохранённое старой версией значение могло выйти за
        // нынешние min/max или стать пустым там, где пусто нельзя
        setHot(el, saved == null || saved === '' ? (saved === '' ? null : hotValue(el)) : Number(saved));
        el.tabIndex = 0;

        el.addEventListener('pointerdown', (e) => {
            if (e.button !== 0 || el.classList.contains('disabled')) return;
            const startX = e.clientX;
            const step = Number(el.dataset.step || 1);
            const start = hotValue(el) !== null ? hotValue(el) : Number(el.dataset.min);
            let dragged = false;
            try {
                el.setPointerCapture(e.pointerId);
            } catch (err) {
                // синтетическое событие без настоящего указателя
            }

            const move = (ev) => {
                const dx = ev.clientX - startX;
                if (!dragged && Math.abs(dx) < 3) return;
                dragged = true;
                const factor = ev.shiftKey ? 10 : 1;
                setHot(el, start + Math.round(dx / 4) * step * factor);
            };
            const finish = (edit) => {
                el.removeEventListener('pointermove', move);
                el.removeEventListener('pointerup', up);
                el.removeEventListener('pointercancel', cancel);
                if (edit && !dragged) editHot(el);
            };
            const up = () => finish(true);
            const cancel = () => finish(false);
            el.addEventListener('pointermove', move);
            el.addEventListener('pointerup', up);
            el.addEventListener('pointercancel', cancel);
        });
        el.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !el.classList.contains('disabled')) editHot(el);
        });
    }

    // ---------- busy state, progress, status ----------

    const TAB_PARTS = {
        fill: { go: 'btn-inpaint', stop: 'btn-stop', progress: 'fill-progress', status: 'fill-status', panel: 'tab-fill' },
        upscale: { go: 'btn-upscale', stop: 'btn-upscale-stop', progress: 'upscale-progress', status: 'upscale-status', panel: 'tab-upscale' },
    };

    function setBusy(tab, busy) {
        const p = TAB_PARTS[tab];
        $(p.go).hidden = busy;
        $(p.stop).hidden = !busy;
        $(p.stop).disabled = false;
        $(p.progress).hidden = !busy;
        if (busy) progress(tab, null);
        // Пока идёт задача, второй её не запустить ни с одной вкладки
        Object.values(TAB_PARTS).forEach(other => { $(other.go).disabled = busy; });
        $(p.panel).querySelectorAll('.segmented button, textarea, .options input').forEach(el => { el.disabled = busy; });
        $(p.panel).querySelectorAll('.hot').forEach(el => el.classList.toggle('disabled', busy));
    }

    function stopping(tab) {
        $(TAB_PARTS[tab].stop).disabled = true;
        status(tab, 'Stopping…');
    }

    // fraction 0..1, или null — неизвестно сколько осталось
    function progress(tab, fraction) {
        const box = $(TAB_PARTS[tab].progress);
        box.classList.toggle('indeterminate', fraction === null);
        box.querySelector('.progress-bar').style.width = fraction === null ? '' : `${Math.round(fraction * 100)}%`;
    }

    function status(tab, text, kind = 'info') {
        const el = $(TAB_PARTS[tab].status);
        el.textContent = text || '';
        el.className = `status ${kind}`;
    }

    // ---------- activity log ----------

    function log(message, kind = 'info') {
        const list = $('log');
        const item = document.createElement('li');
        item.className = kind;
        const time = document.createElement('time');
        time.textContent = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        item.append(time, document.createTextNode(message));
        list.appendChild(item);
        while (list.children.length > 100) list.removeChild(list.firstChild);
        list.scrollTop = list.scrollHeight;
    }

    function serverState(kind) {
        const box = $('server-status');
        box.classList.toggle('online', kind === 'online');
        box.classList.toggle('starting', kind === 'starting');
        // "Idle", а не "Off": до первого запуска сервер выключен штатно
        $('server-text').textContent = { online: 'Ready', starting: 'Starting…', off: 'Idle' }[kind] || kind;
    }

    // ---------- first run ----------

    function confirmFirstRun() {
        return new Promise((resolve) => {
            const modal = $('first-run-modal');
            modal.hidden = false;
            const onKey = (e) => {
                if (e.key === 'Escape') close(false);
            };
            const close = (answer) => {
                modal.hidden = true;
                $('btn-first-run-continue').onclick = null;
                $('btn-first-run-cancel').onclick = null;
                document.removeEventListener('keydown', onKey);
                resolve(answer);
            };
            $('btn-first-run-continue').onclick = () => close(true);
            $('btn-first-run-cancel').onclick = () => close(false);
            document.addEventListener('keydown', onKey);
            // Фокус в окно: иначе Enter/Space снова жмёт Inpaint под ним
            $('btn-first-run-continue').focus();
        });
    }

    // ---------- settings ----------

    function fillSettings() {
        const num = (id) => hotValue($(id));
        return {
            prompt: $('prompt').value.trim(),
            expand: num('expand') || 0,
            feather: num('feather') || 0,
            steps: num('steps'),
            guidance: num('guidance'),
            strength: num('strength') !== null ? num('strength') : 1,
            seed: num('seed'),
            invertMask: $('invert-mask').checked,
            fillTransparent: $('fill-transparent').checked,
        };
    }

    function upscaleSettings() {
        return {
            scale: parseInt(segmentedValue('scale'), 10) || 4,
            modelType: segmentedValue('upscale-model') || 'anime',
        };
    }

    // ---------- init ----------

    function init(handlers) {
        load();
        initTheme();

        document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => selectTab(t.dataset.tab)));
        if (state.tab) selectTab(state.tab);

        initSegmented('mode', applyMode);
        initSegmented('scale');
        initSegmented('upscale-model');
        applyMode(segmentedValue('mode'));

        document.querySelectorAll('.hot').forEach(initHot);

        ['invert-mask', 'fill-transparent'].forEach(id => {
            const el = $(id);
            if (state[id] != null) el.checked = state[id];
            el.addEventListener('change', () => { state[id] = el.checked; save(); });
        });

        const prompt = $('prompt');
        if (state.prompt) prompt.value = state.prompt;
        prompt.addEventListener('input', () => { state.prompt = prompt.value; save(); });

        const options = $('fill-options');
        if (state.optionsOpen) options.open = true;
        options.addEventListener('toggle', () => { state.optionsOpen = options.open; save(); });

        $('btn-inpaint').addEventListener('click', handlers.onInpaint);
        $('btn-stop').addEventListener('click', handlers.onStop);
        $('btn-upscale').addEventListener('click', handlers.onUpscale);
        $('btn-upscale-stop').addEventListener('click', handlers.onStop);
        if (handlers.onDebugToggle) $('btn-debug-mode').addEventListener('click', handlers.onDebugToggle);
    }

    root.UI = {
        init,
        mode: () => segmentedValue('mode'),
        fillSettings,
        upscaleSettings,
        setBusy,
        stopping,
        progress,
        status,
        log,
        serverState,
        confirmFirstRun,
        showDevTools: () => { $('dev-tools').hidden = false; },
        setDebugLabel: (on) => { $('btn-debug-mode').textContent = `CEP Debug: ${on ? 'ON' : 'OFF'}`; },
        applyTheme,  // для превью/тестов
    };
})(typeof window !== 'undefined' ? window : globalThis);
