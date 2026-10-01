localStorage.clear();
setTimeout(() => {
    const out = {};
    const ev = (el, type, x, extra = {}) => el.dispatchEvent(new PointerEvent(type, { bubbles: true, clientX: x, pointerId: 1, ...extra }));
    document.getElementById('fill-options').open = true;
    document.querySelector('#mode button[data-value="ai"]').click();

    // drag Mask expand +40px → +10
    const expand = document.getElementById('expand');
    ev(expand, 'pointerdown', 100); ev(expand, 'pointermove', 140); ev(expand, 'pointerup', 140);
    // shift-drag Feather +8px → +2*10 = 20
    const feather = document.getElementById('feather');
    ev(feather, 'pointerdown', 100); ev(feather, 'pointermove', 108, { shiftKey: true }); ev(feather, 'pointerup', 108);
    // drag beyond max
    ev(expand, 'pointerdown', 100); ev(expand, 'pointermove', 900); ev(expand, 'pointerup', 900);
    out.expandMax = UI.fillSettings().expand;
    ev(expand, 'pointerdown', 100); ev(expand, 'pointermove', 100 - 4 * 47); ev(expand, 'pointerup', 0);

    // click Seed → type 42 → Enter
    const seed = document.getElementById('seed');
    ev(seed, 'pointerdown', 50); ev(seed, 'pointerup', 50);
    const input = document.querySelector('.hot-input');
    out.inputShown = !!input;
    input.value = '42';
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    // Steps: type 8, then clear back to Auto
    const steps = document.getElementById('steps');
    ev(steps, 'pointerdown', 50); ev(steps, 'pointerup', 50);
    let i2 = document.querySelector('.hot-input'); i2.value = '8'; i2.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    out.stepsTyped = UI.fillSettings().steps;
    ev(steps, 'pointerdown', 50); ev(steps, 'pointerup', 50);
    i2 = document.querySelector('.hot-input'); i2.value = ''; i2.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    // Strength: escape cancels
    const strength = document.getElementById('strength');
    ev(strength, 'pointerdown', 50); ev(strength, 'pointerup', 50);
    i2 = document.querySelector('.hot-input'); i2.value = '0.3'; i2.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    out.settings = UI.fillSettings();
    out.stepsText = steps.textContent;
    out.mode = UI.mode();
    out.genHidden = document.querySelector('.gen-only').hidden;
    UI.setBusy('fill', true);
    out.busyGoHidden = document.getElementById('btn-inpaint').hidden;
    out.busyUpscaleDisabled = document.getElementById('btn-upscale').disabled;
    out.busySegDisabled = document.querySelector('#mode button').disabled;
    UI.setBusy('fill', false);
    out.afterBusy = document.querySelector('#mode button').disabled;
    out.saved = JSON.parse(localStorage.getItem('ae_inpaint_ui_v1'));
    out.actionLabel = document.getElementById('btn-inpaint').textContent;
    const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
    out.darkBg = css('--bg');
    out.darkHot = css('--hot');
    out.darkDanger = css('--danger');
    window.__skin = { red: 184, green: 184, blue: 184 };
    if (window.__themeListener) window.__themeListener();
    out.lightBg = css('--bg');
    out.lightText = css('--text');
    out.lightHot = css('--hot');
    document.title = 'RESULT ' + JSON.stringify(out);
}, 100);
