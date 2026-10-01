// UI панели в настоящем Chrome (headless): горячие числа, ввод, Auto,
// блокировка во время работы, сохранение настроек. Заглушки CEP/Node —
// tests/ui/stubs.js. Если Chrome не найден, тест пропускается.
// Запуск: node --test extension/tests/*.test.js
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const EXT = path.resolve(__dirname, '..');
const CHROMES = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome', '/usr/bin/chromium-browser', '/usr/bin/chromium',
];
const chrome = CHROMES.find(p => fs.existsSync(p));

function harness(scenario) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ae-inpaint-ui-'));
    let html = fs.readFileSync(path.join(EXT, 'index.html'), 'utf8')
        .replace(/href="css\//g, `href="file://${EXT}/css/`)
        .replace('<script src="js/CSInterface.js"></script>', `<script src="file://${__dirname}/ui/stubs.js"></script>`)
        .replace(/src="js\//g, `src="file://${EXT}/js/`)
        .replace('</body>', `<script src="file://${__dirname}/ui/${scenario}"></script></body>`);
    const file = path.join(dir, 'index.html');
    fs.writeFileSync(file, html);
    return { dir, file };
}

test('panel UI interactions', { skip: !chrome && 'Chrome not found' }, () => {
    const { dir, file } = harness('interact.js');
    try {
        const dom = execFileSync(chrome, [
            '--headless=new', '--disable-gpu', '--allow-file-access-from-files',
            `--user-data-dir=${dir}/profile`, '--virtual-time-budget=1500', '--dump-dom', `file://${file}`,
        ], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 60000 });
        const m = dom.match(/<title>RESULT (.*)<\/title>/);
        assert.ok(m, 'scenario did not finish (JS error in the panel?)');
        const r = JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&amp;/g, '&'));

        assert.strictEqual(r.expandMax, 50, 'drag clamps at max');
        assert.strictEqual(r.settings.expand, 3, 'drag back to 3');
        assert.strictEqual(r.settings.feather, 20, 'shift-drag ×10');
        assert.ok(r.inputShown, 'click opens the text field');
        assert.strictEqual(r.settings.seed, 42, 'typed seed');
        assert.strictEqual(r.stepsTyped, 8);
        assert.strictEqual(r.settings.steps, null, 'cleared field → Auto');
        assert.strictEqual(r.stepsText, 'Auto');
        assert.strictEqual(r.settings.strength, 1, 'Escape cancels editing');
        assert.strictEqual(r.mode, 'ai');
        assert.strictEqual(r.genHidden, false, 'Generate shows its options');
        assert.ok(r.busyGoHidden && r.busyUpscaleDisabled && r.busySegDisabled, 'busy state locks the panel');
        assert.strictEqual(r.afterBusy, false, 'and unlocks after');
        assert.strictEqual(r.saved.mode, 'ai', 'mode remembered');
        assert.strictEqual(r.saved.hot.seed, '42', 'values remembered');
        assert.strictEqual(r.actionLabel, 'Generate', 'button named after the mode');
        assert.strictEqual(r.darkBg, '#232323', 'theme read via getHostEnvironment()');
        assert.strictEqual(r.lightBg, '#b8b8b8', 'theme follows ThemeColorChanged');
        assert.ok(parseInt(r.lightText.slice(1, 3), 16) < 0x60, 'dark text on light AE theme');
        // На стандартной тёмной теме — фирменные синий и красный, а не белёсые
        assert.strictEqual(r.darkHot, '#4ca2f5');
        assert.strictEqual(r.darkDanger, '#ff6b66');
        assert.ok(parseInt(r.lightHot.slice(5, 7), 16) > parseInt(r.lightHot.slice(1, 3), 16), 'still blue on light theme');
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
});
