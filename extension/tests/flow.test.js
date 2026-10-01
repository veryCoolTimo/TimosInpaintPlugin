// Сценарии целиком: настоящая панель в headless Chrome, After Effects и
// сервер — заглушки (tests/ui/flow-stubs.js). Проверяется то, что видит
// пользователь: статус, журнал, сколько задач ушло на сервер и сколько
// слоёв импортировано. Без Chrome тест пропускается.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const EXT = path.resolve(__dirname, '..');
const chrome = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome', '/usr/bin/chromium-browser', '/usr/bin/chromium',
].find(p => fs.existsSync(p));

function run(scenario) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ae-inpaint-flow-'));
    try {
        const html = fs.readFileSync(path.join(EXT, 'index.html'), 'utf8')
            .replace(/href="css\//g, `href="file://${EXT}/css/`)
            .replace('<script src="js/CSInterface.js"></script>',
                `<script>localStorage.clear()</script><script src="file://${__dirname}/ui/flow-stubs.js"></script>`)
            .replace(/src="js\//g, `src="file://${EXT}/js/`)
            .replace('</body>', `<script src="file://${__dirname}/ui/flow-scenarios.js"></script></body>`);
        fs.writeFileSync(path.join(dir, 'h.html'), html);
        const dom = execFileSync(chrome, [
            '--headless=new', '--disable-gpu', '--allow-file-access-from-files', `--user-data-dir=${dir}/p`,
            '--virtual-time-budget=12000', '--dump-dom', `file://${dir}/h.html#${scenario}`,
        ], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 120000 });
        const m = dom.match(/<title>RESULT (.*?)<\/title>/s);
        assert.ok(m, `${scenario}: did not finish`);
        const r = JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&'));
        r.created = r.log.filter(l => l.startsWith('success:Created')).length;
        return r;
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}

const skip = !chrome && 'Chrome not found';

test('Remove: one layer, success status, panel unlocked', { skip }, () => {
    const r = run('remove');
    assert.strictEqual(r.created, 1);
    assert.match(r.fill.text, /^Done in/);
    assert.ok(!r.goHidden && !r.goDisabled && r.stopHidden);
});

test('Generate: "Not now" on the download dialog runs nothing', { skip }, () => {
    const r = run('notnow');
    assert.ok(r.modalShown);
    assert.strictEqual(r.created, 0);
    assert.strictEqual(r.jobs, 0);
});

test('Generate: progress shows steps, model flag saved', { skip }, () => {
    const r = run('download');
    assert.match(r.mid, /step \d of 4/);
    assert.strictEqual(r.created, 1);
    assert.strictEqual(r.flag, 'true');
});

test('Stop mid-generation: nothing imported, panel unlocked', { skip }, () => {
    const r = run('stop');
    assert.strictEqual(r.created, 0);
    assert.ok(r.cancelled, '/cancel sent');
    assert.ok(r.log.includes('info:Stopped'));
    assert.ok(!r.goHidden && !r.goDisabled);
});

test('Server error shows red status and Activity entry', { skip }, () => {
    const r = run('err500');
    assert.match(r.fill.cls, /error/);
    assert.ok(r.log.some(l => l.startsWith('error:')));
});

test('Upscale: one of two layers fails → summary', { skip }, () => {
    const r = run('upscale');
    assert.match(r.up.text, /1 of 2 done, 1 failed/);
});

test('Upscale Stop is not reported as success', { skip }, () => {
    const r = run('stopUpscale');
    assert.match(r.up.text, /^Stopped/);
    assert.ok(!r.log.some(l => l.startsWith('success:Upscale')));
});

for (const scenario of ['dblRemove', 'dblAiCached', 'realDblCached', 'realDblFlag', 'dblAiModal']) {
    test(`double click starts one job (${scenario})`, { skip }, () => {
        const r = run(scenario);
        assert.strictEqual(r.created, 1);
    });
}
