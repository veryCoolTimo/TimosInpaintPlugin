// Заглушки After Effects + локального сервера для сценарных тестов панели (flow.test.js)
// E2E stubs: AE (evalScript) + Node (fs) + server (fetch)
class Buf extends Uint8Array {
  static from(x, enc) {
    if (typeof x === 'string' && enc === 'base64') { const s = atob(x); const b = new Buf(s.length); for (let i=0;i<s.length;i++) b[i]=s.charCodeAt(i); return b; }
    const b = new Buf(x.length); b.set(x); return b;
  }
  subarray(a, e) { const u = Uint8Array.prototype.subarray.call(this, a, e); return Buf.from(u); }
  equals(o) { if (o.length !== this.length) return false; for (let i=0;i<o.length;i++) if (o[i]!==this[i]) return false; return true; }
  toString(enc) { let s=''; for (const c of this) s+=String.fromCharCode(c); return enc==='base64'?btoa(s):s; }
}
window.Buffer = Buf;
const PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
window.T = { evals: [], fetches: [], writes: 0, cfg: window.__cfg || {}, statuses: [], progress: [] };
const fsStub = {
  readFileSync: (p, enc) => enc ? '' : Buf.from(PNG_B64, 'base64'),
  realpathSync: (p) => p, existsSync: () => true, mkdirSync() {}, readdirSync: () => [], rmSync() {}, statSync: () => ({ isDirectory: () => true, mtimeMs: Date.now() }),
  promises: { mkdir: async () => {}, writeFile: async () => { T.writes++; } },
};
window.require = (name) => ({
  path: { join: (...a) => a.join('/'), dirname: (p) => p.split('/').slice(0, -1).join('/') },
  fs: fsStub,
  os: { tmpdir: () => '/tmp', homedir: () => '/Users/me' },
  child_process: { spawn() { T.spawned = true; throw new Error('spawn'); }, execSync: () => '', exec() {} },
})[name] || {};
window.process = { pid: 1, env: {} };
const AE = {
  getProjectInfo: () => ({ compName: 'Comp 1', currentFrame: 120, projectPath: '/Users/me/proj' }),
  getSelectedLayerWithMask: () => ({ index: 2, name: 'Plate', selectedMaskIndex: 1, selectedMaskName: 'Mask 1' }),
  exportForInpaint: () => ({ imagePath: '/tmp/x/img.png', maskPath: '/tmp/x/mask.png' }),
  exportLayerFrame: (args) => (T.cfg.failLayer && args.startsWith(String(T.cfg.failLayer)) ? { error: 'Layer is a camera' } : { imagePath: '/tmp/x/l.png', layerIndex: Number(args.split(',')[0]) }),
  importResultAsLayer: (args) => ({ layerName: JSON.parse('[' + args + ']')[2] }),
  getSelectedLayers: () => ({ layers: [{ index: 1, name: 'BG' }, { index: 3, name: 'Cam' }] }),
};
window.CSInterface = function () {
  this.getSystemPath = () => '/ext';
  this.evalScript = (s, cb) => {
    const m = s.match(/^(\w+)\((.*)\)$/s);
    let r = 'function';
    if (s === 'app.version') r = '24.6';
    else if (m && AE[m[1]]) { T.evals.push(m[1]); r = JSON.stringify(AE[m[1]](m[2])); }
    setTimeout(() => cb && cb(r), 30);
  };
  this.addEventListener = () => {};
  this.hostEnvironment = { appSkinInfo: { panelBackgroundColor: { color: { red: 35, green: 35, blue: 35 } } } };
  this.getHostEnvironment = () => this.hostEnvironment;
};
CSInterface.THEME_COLOR_CHANGED_EVENT = 'theme';
let job = null; // {start, total}
const resp = (status, body) => ({ ok: status < 400, status, json: async () => body });
window.fetch = (url, opt = {}) => {
  const p = url.replace('http://127.0.0.1:7860', '');
  T.fetches.push(p);
  if (p === '/health') return new Promise(r=>setTimeout(r,T.cfg.healthDelay||0)).then(()=>resp(200, { status: 'ok', engine: 'flux2', model_cached: !!T.cfg.cached }));
  if (p === '/attach') return Promise.resolve(resp(200, { status: 'ok' }));
  if (p === '/cancel') { T.cancelled = true; return Promise.resolve(resp(200, { ok: true })); }
  if (p === '/progress') {
    if (!job) return Promise.resolve(resp(200, { stage: 'idle' }));
    const step = Math.min(4, 1 + Math.floor((Date.now() - job.start) / 250));
    return Promise.resolve(resp(200, { stage: 'inpainting', step, total_steps: 4 }));
  }
  if (p === '/inpaint' || p === '/upscale') {
    T.jobs = (T.jobs || 0) + 1;
    return new Promise((res, rej) => {
      job = { start: Date.now() };
      const t = setTimeout(() => {
        job = null;
        if (T.cfg.fail500 && p === '/inpaint') res(resp(500, { detail: 'CUDA out of memory' }));
        else res(resp(200, { result: PNG_B64 }));
      }, T.cfg.delay || 1000);
      opt.signal && opt.signal.addEventListener('abort', () => { clearTimeout(t); job = null; const e = new Error('aborted'); e.name = 'AbortError'; rej(e); });
    });
  }
  return Promise.reject(new Error('unexpected ' + p));
};
// sample status/progress
setInterval(() => {
  const s = document.getElementById('fill-status'); if (!s) return;
  const txt = s.textContent; if (T.statuses[T.statuses.length - 1] !== txt) T.statuses.push(txt);
  const w = document.querySelector('#fill-progress .progress-bar').style.width;
  if (w && T.progress[T.progress.length - 1] !== w) T.progress.push(w);
}, 50);
window.wait = (ms) => new Promise(r => setTimeout(r, ms));
window.$ = (s) => document.querySelector(s);
window.dump = (extra) => {
  const st = (id) => ({ text: $(id).textContent, cls: $(id).className });
  document.title = 'RESULT ' + JSON.stringify(Object.assign({
    fill: st('#fill-status'), up: st('#upscale-status'),
    goHidden: $('#btn-inpaint').hidden, goDisabled: $('#btn-inpaint').disabled, stopHidden: $('#btn-stop').hidden,
    progHidden: $('#fill-progress').hidden, modalHidden: $('#first-run-modal').hidden,
    log: [...document.querySelectorAll('#log li')].map(li => li.className + ':' + li.textContent.slice(8)),
    evals: T.evals, jobs: T.jobs || 0, writes: T.writes, statuses: T.statuses, progress: T.progress, cancelled: !!T.cancelled,
    flag: localStorage.getItem('ae_inpaint_model_downloaded'),
  }, extra || {}));
};
window.onerror = (m) => { document.title = 'JSERR ' + m; };
