// Заглушки CEP/Node, чтобы открыть панель в обычном Chrome
window.require = (name) => ({
    path: { join: (...a) => a.join('/'), dirname: (p) => p.split('/').slice(0, -1).join('/') },
    fs: { readFileSync: () => '', realpathSync: (p) => p, existsSync: () => false, mkdirSync() {}, readdirSync: () => [], rmSync() {}, promises: { mkdir: async () => {}, writeFile: async () => {} } },
    os: { tmpdir: () => '/tmp', homedir: () => '/Users/me' },
    child_process: { spawn() {}, execSync: () => '', exec() {} },
})[name] || {};
window.process = { pid: 1, env: {} };
// Как поставляемый js/CSInterface.js: только getHostEnvironment(), без
// свойства hostEnvironment и без констант событий
window.CSInterface = function () {};
CSInterface.prototype.getSystemPath = () => '/ext';
CSInterface.prototype.evalScript = (s, cb) => cb && cb(s === 'app.version' ? '24.6' : 'function');
CSInterface.prototype.addEventListener = function (type, fn) { window.__themeListener = type === 'com.adobe.csxs.events.ThemeColorChanged' ? fn : window.__themeListener; };
CSInterface.prototype.getHostEnvironment = () => ({ appSkinInfo: { panelBackgroundColor: { color: window.__skin || { red: 35, green: 35, blue: 35 } } } });
window.fetch = () => Promise.reject(new Error('offline'));
