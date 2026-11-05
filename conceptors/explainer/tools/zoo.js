const { chromium } = require('playwright');
(async () => {
  const mode = process.argv[2] || 'light', width = +(process.argv[3] || 1280);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width, height: 1000 }, deviceScaleFactor: 1, colorScheme: mode });
  const logs = [];
  page.on('pageerror', e => logs.push('PAGEERROR: ' + e.message));
  await page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
  await page.goto('file://' + require('path').resolve(__dirname, '../persona_steering_field_guide.html'));
  await page.waitForTimeout(800);
  for (const m of process.argv.slice(4)) {
    await page.click('#zooList .chip[data-m="' + m + '"]');
    await page.waitForTimeout(300);
    const el = await page.$('#zoo-explorer');
    await el.screenshot({ path: require('path').resolve(__dirname, '../screenshots') + `/zoo-${mode}-${width}-${m}.png` });
  }
  console.log(logs.join('\n') || 'no page errors');
  await browser.close();
})();
