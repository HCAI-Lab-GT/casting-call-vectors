const { chromium } = require('playwright');
(async () => {
  const mode = process.argv[2] || 'light', width = +(process.argv[3] || 1280);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width, height: 900 }, deviceScaleFactor: 1, colorScheme: mode });
  const logs = [];
  page.on('console', m => { if (['error', 'warning'].includes(m.type())) logs.push(m.type() + ': ' + m.text()); });
  page.on('pageerror', e => logs.push('PAGEERROR: ' + e.message));
  await page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
  await page.goto('file://' + require('path').resolve(__dirname, '../persona_steering_field_guide.html'));
  await page.waitForTimeout(1200);
  const ids = process.argv.slice(4);
  for (const id of ids) {
    const el = await page.$('#' + id);
    if (!el) { logs.push('missing #' + id); continue; }
    await el.scrollIntoViewIfNeeded();
    await page.waitForTimeout(250);
    await el.screenshot({ path: require('path').resolve(__dirname, '../screenshots') + `/${mode}-${width}-${id}.png` });
  }
  const sw = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, h: document.documentElement.scrollHeight }));
  console.log(JSON.stringify(sw));
  console.log(logs.join('\n') || 'no console errors');
  await browser.close();
})();
