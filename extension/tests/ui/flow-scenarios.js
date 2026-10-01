// Сценарии: клики по настоящей панели, результат — JSON в document.title (dump())
const SC = {
  async remove() { $('#mode button[data-value="remove"]').click(); $('#btn-inpaint').click(); await wait(2000); dump(); },
  async notnow() { $('#mode button[data-value="ai"]').click(); $('#btn-inpaint').click(); await wait(300); const shown = !$('#first-run-modal').hidden; $('#btn-first-run-cancel').click(); await wait(200); dump({ modalShown: shown }); },
  async download() { $('#mode button[data-value="ai"]').click(); $('#btn-inpaint').click(); await wait(300); $('#btn-first-run-continue').click(); await wait(800); const mid = $('#fill-status').textContent; await wait(2000); dump({ mid }); },
  async stop() { $('#mode button[data-value="ai"]').click(); T.cfg.cached = true; $('#btn-inpaint').click(); await wait(700); const mid = $('#fill-status').textContent; $('#btn-stop').click(); await wait(1500); dump({ mid }); },
  async err500() { T.cfg.fail500 = true; $('#mode button[data-value="remove"]').click(); $('#btn-inpaint').click(); await wait(2000); dump({ activityOpen: $('#activity').open }); },
  async upscale() { T.cfg.failLayer = 3; $('.tab[data-tab="upscale"]').click(); $('#btn-upscale').click(); await wait(4000); dump(); },
  async dblRemove() { $('#mode button[data-value="remove"]').click(); $('#btn-inpaint').click(); $('#btn-inpaint').click(); await wait(2500); dump(); },
  async dblAiCached() { T.cfg.cached = true; $('#mode button[data-value="ai"]').click(); $('#btn-inpaint').click(); $('#btn-inpaint').click(); await wait(3500); dump(); },
  async dblAiFlag() { localStorage.setItem('ae_inpaint_model_downloaded','true'); $('#mode button[data-value="ai"]').click(); $('#btn-inpaint').click(); $('#btn-inpaint').click(); await wait(3500); dump(); },
  async dblAiModal() { $('#mode button[data-value="ai"]').click(); $('#btn-inpaint').click(); await wait(100); $('#btn-inpaint').click(); await wait(200); $('#btn-first-run-continue').click(); await wait(3500); dump(); },
  async stopUpscale() { T.cfg.delay = 1500; $('.tab[data-tab="upscale"]').click(); $('#btn-upscale').click(); await wait(600); $('#btn-upscale-stop').click(); await wait(2500); dump(); },
};
SC.realDblFlag = async () => { localStorage.setItem('ae_inpaint_model_downloaded','true'); $('#mode button[data-value="ai"]').click(); $('#btn-inpaint').click(); await wait(0); $('#btn-inpaint').click(); await wait(3500); dump(); };
SC.realDblCached = async () => { T.cfg.cached = true; T.cfg.healthDelay = 50; $('#mode button[data-value="ai"]').click(); $('#btn-inpaint').click(); await wait(20); $('#btn-inpaint').click(); await wait(3500); dump(); };
SC.realDblRemoveSlowHealth = async () => { T.cfg.healthDelay = 50; $('#mode button[data-value="remove"]').click(); $('#btn-inpaint').click(); await wait(20); $('#btn-inpaint').click(); await wait(3500); dump(); };
const name = location.hash.slice(1);
setTimeout(() => SC[name]().catch(e => { document.title = 'SCERR ' + e.message; }), 300);
