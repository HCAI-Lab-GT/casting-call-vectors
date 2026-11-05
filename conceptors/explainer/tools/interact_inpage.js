// In-page version of interact.js for environments without Playwright (a browser console, or a browser
// automation tool's "evaluate JavaScript"). Paste it into the page and await the result:
//   await (async () => { /* this file */ })()
// It clicks every control group, sweeps every slider, hovers the charts, opens a multi-turn answer, flips the
// theme, and returns {errors, missing, tables, mtAns}. Load-time errors happen before it runs: read the console too.
(async () => {
  const errs = [];
  const onErr = e => errs.push('PAGEERROR: ' + e.message);
  window.addEventListener('error', onErr);
  const origErr = console.error;
  console.error = (...a) => { errs.push(a.join(' ')); origErr(...a); };
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const q = s => Array.from(document.querySelectorAll(s));
  const clickAll = async sel => { for (const b of q(sel)) { b.click(); await sleep(15); } };
  const missing = [];
  for (const id of ['m3', 'm4', 'm5', 'm6', 'm6c', 'm8', 'mapView', 'mapPool', 'stripPool', 'logicFit', 'logicLayer', 'mtConv']) {
    if (!document.getElementById(id)) missing.push(id); else await clickAll('#' + id + ' button');
  }
  document.getElementById('preset6')?.click();
  for (const id of ['caa', 'gate', 'cap', 'liseco', 'pass', 'sea', 'mags', 'conceptor', 'coast', 'ot', 'manifold', 'gcad']) {
    const c = document.querySelector('#zooList .chip[data-m="' + id + '"]');
    if (c) { c.click(); await sleep(15); } else missing.push('zoo:' + id);
  }
  document.querySelector('#zooList .chip[data-m="ot"]')?.click();
  await clickAll('#zooOpt button');
  for (const id of ['p1s', 'p1t', 'p2a', 'p3s', 'p3a', 'p4b', 'p4a', 'p5t', 'p6b', 'p6a', 'p8b', 'p9a', 'p9c', 'mapLayer', 'zs1', 'simPull', 'simMem']) {
    const el = document.getElementById(id);
    if (!el) { missing.push(id); continue; }
    for (const v of [el.max, el.min, el.value]) { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); await sleep(10); }
  }
  for (const id of ['cSweepAuc', 'cMap', 'cStrip', 'cCover', 'cMT', 'cSimP', 'c9', 'cGcadC']) {
    const el = document.getElementById(id);
    if (!el) { missing.push(id); continue; }
    const b = el.getBoundingClientRect();
    for (const fx of [0.3, 0.5, 0.8]) {
      el.dispatchEvent(new MouseEvent('mousemove', { clientX: b.x + b.width * fx, clientY: b.y + b.height * 0.5, bubbles: true }));
      await sleep(10);
    }
  }
  const mt = document.getElementById('cMT');
  if (mt) {
    const b = mt.getBoundingClientRect();
    mt.dispatchEvent(new MouseEvent('click', { clientX: b.x + b.width * 0.45, clientY: b.y + b.height * 0.5, bubbles: true }));
  }
  document.documentElement.setAttribute('data-theme', 'dark'); await sleep(100);
  document.documentElement.setAttribute('data-theme', 'light'); await sleep(100);
  const tables = q('details.dtab').map(d => d.querySelectorAll('tbody tr').length);
  console.error = origErr;
  window.removeEventListener('error', onErr);
  return { errors: errs, missing, tables, mtAns: (document.querySelector('#mtAns .annot') || {}).textContent?.slice(0, 80) };
})();
