// Тесты файловых хелперов панели. Запуск: node --test extension/tests/*.test.js
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const zlib = require('zlib');

const Files = require('../js/files.js');

// Минимальный валидный PNG 1×1
function tinyPng() {
    function chunk(type, data) {
        const len = Buffer.alloc(4);
        len.writeUInt32BE(data.length);
        const body = Buffer.concat([Buffer.from(type), data]);
        const crc = Buffer.alloc(4);
        crc.writeUInt32BE(zlib.crc32 ? zlib.crc32(body) : 0);
        return Buffer.concat([len, body, crc]);
    }
    const ihdr = Buffer.from([0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0]);
    const idat = zlib.deflateSync(Buffer.from([0, 255, 0, 0]));
    return Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
        chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0)),
    ]);
}

function tmp() {
    return fs.mkdtempSync(path.join(os.tmpdir(), 'ae-inpaint-test-'));
}

test('isPngComplete: full vs truncated', () => {
    const png = tinyPng();
    assert.ok(Files.isPngComplete(png));
    assert.ok(!Files.isPngComplete(png.subarray(0, png.length - 4)));
    assert.ok(!Files.isPngComplete(Buffer.from('not a png at all......')));
});

test('readRenderedPng returns immediately when the file is ready', async () => {
    const dir = tmp();
    const file = path.join(dir, 'frame.png');
    fs.writeFileSync(file, tinyPng());
    const started = Date.now();
    const b64 = await Files.readRenderedPng(file);
    assert.strictEqual(Buffer.from(b64, 'base64').length, tinyPng().length);
    // раньше было ~3 с фиксированного ожидания на каждый файл
    assert.ok(Date.now() - started < 500, `took ${Date.now() - started}ms`);
    Files.removeDir(dir);
});

test('readRenderedPng waits while AE is still writing the file', async () => {
    const dir = tmp();
    const file = path.join(dir, 'frame.png');
    const png = tinyPng();
    setTimeout(() => fs.writeFileSync(file, png.subarray(0, 20)), 100);
    setTimeout(() => fs.writeFileSync(file, png), 400);
    const b64 = await Files.readRenderedPng(file, 5000);
    assert.ok(Files.isPngComplete(Buffer.from(b64, 'base64')));
    Files.removeDir(dir);
});

test('readRenderedPng times out with a clear error', async () => {
    const dir = tmp();
    await assert.rejects(Files.readRenderedPng(path.join(dir, 'missing.png'), 300), /not ready/);
    Files.removeDir(dir);
});

test('job dirs are unique and removed completely', () => {
    const root = tmp();
    const a = Files.makeJobDir(root);
    const b = Files.makeJobDir(root);
    assert.notStrictEqual(a, b);
    fs.writeFileSync(path.join(a, 'image.png'), 'x');
    fs.writeFileSync(path.join(a, 'mask.png'), 'x');
    Files.removeDir(a);
    assert.ok(!fs.existsSync(a));
    assert.ok(fs.existsSync(b));
    Files.removeDir(root);
});

test('cleanStaleTemp removes only old job dirs', () => {
    const root = tmp();
    const old = Files.makeJobDir(root);
    const fresh = Files.makeJobDir(root);
    const log = path.join(root, 'server.log');
    fs.writeFileSync(log, 'log');
    const past = (Date.now() - 2 * 24 * 3600 * 1000) / 1000;
    fs.utimesSync(old, past, past);
    fs.utimesSync(log, past, past);
    Files.cleanStaleTemp(root);
    assert.ok(!fs.existsSync(old));
    assert.ok(fs.existsSync(fresh));
    assert.ok(fs.existsSync(log), 'server.log must survive');
    Files.cleanStaleTemp(path.join(root, 'does-not-exist'));  // не падает
    Files.removeDir(root);
});

test('results go next to the project, or to Documents when unsaved', () => {
    const project = tmp();
    const home = tmp();
    assert.strictEqual(Files.getResultsDir(project), path.join(project, Files.RESULTS_FOLDER_NAME));
    assert.strictEqual(Files.getResultsDir(null, home), path.join(home, 'Documents', Files.RESULTS_FOLDER_NAME));
    assert.ok(fs.existsSync(path.join(home, 'Documents', Files.RESULTS_FOLDER_NAME)));
    Files.removeDir(project);
    Files.removeDir(home);
});

test('base64ToFile creates missing folders', async () => {
    const root = tmp();
    const file = path.join(root, 'a', 'b', 'out.png');
    await Files.base64ToFile(tinyPng().toString('base64'), file);
    assert.ok(Files.isPngComplete(fs.readFileSync(file)));
    Files.removeDir(root);
});
