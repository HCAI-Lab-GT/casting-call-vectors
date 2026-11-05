const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errs = [];
  page.on('pageerror', e => errs.push('PAGEERROR: ' + e.message));
  page.on('console', m => { if (m.type() === 'error' && !/ERR_FAILED/.test(m.text())) errs.push(m.text()); });
  await page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
  await page.goto('file://' + require('path').resolve(__dirname, '../persona_steering_field_guide.html'));
  await page.waitForTimeout(600);
  const clickAll = async sel => { for (const b of await page.$$(sel)) { await b.scrollIntoViewIfNeeded(); await b.click(); await page.waitForTimeout(60); } };
  for (const id of ['m3','m4','m5','m6','m6c','m8','mapView','mapPool','stripPool','logicFit','logicLayer','mtConv']) await clickAll('#' + id + ' button');
  await page.click('#preset6');
  for (const id of ZOOIDS()) { await page.click('#zooList .chip[data-m="' + id + '"]'); await page.waitForTimeout(40); }
  await page.click('#zooList .chip[data-m="ot"]'); await clickAll('#zooOpt button');
  // hero chip scrolls to zoo
  await page.click('#shapegrid .chip[data-m="gcad"]'); await page.waitForTimeout(700);
  const zooTop = await page.evaluate(() => document.getElementById('zoo-explorer').getBoundingClientRect().top);
  // sliders
  for (const id of ['p1s','p1t','p2a','p3s','p3a','p4b','p4a','p5t','p6b','p6a','p8b','p9a','p9c','mapLayer','zs1','simPull','simMem']) {
    await page.$eval('#' + id, el => { el.value = el.max; el.dispatchEvent(new Event('input')); el.value = el.min; el.dispatchEvent(new Event('input')); });
  }
  // hover charts and scatter
  for (const id of ['cSweepAuc','cMap','cStrip','cCover','cMT','cSimP','c9','cGcadC']) {
    const el = await page.$('#' + id); await el.scrollIntoViewIfNeeded(); const b = await el.boundingBox();
    for (const fx of [.3,.5,.8]) { await page.mouse.move(b.x + b.width * fx, b.y + b.height * .5); await page.waitForTimeout(30); }
  }
  const tipShown = await page.evaluate(() => !document.getElementById('tip').hidden);
  // click a turn in the multi-turn chart
  const mt = await page.$('#cMT'); const bb = await mt.boundingBox(); await page.mouse.click(bb.x + bb.width * .45, bb.y + bb.height * .5); await page.waitForTimeout(100);
  const mtHead = await page.$eval('#mtAns .annot', el => el.textContent);
  // drag in the zoo
  await page.click('#zooList .chip[data-m="cap"]');
  const z = await page.$('#cZoo'); const zb = await z.boundingBox();
  await page.mouse.move(zb.x + zb.width * .3, zb.y + zb.height * .6); await page.mouse.down(); await page.mouse.move(zb.x + zb.width * .5, zb.y + zb.height * .5, { steps: 4 }); await page.mouse.up();
  const zr = await page.$eval('#zooRead', el => el.textContent.slice(0, 80));
  // theme flip
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark')); await page.waitForTimeout(200);
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light')); await page.waitForTimeout(200);
  // tables
  const tables = await page.$$eval('details.dtab', d => d.map(x => x.querySelectorAll('tbody tr').length));
  console.log(JSON.stringify({ zooTop: Math.round(zooTop), tipShown, mtHead, zr, tables }));
  console.log(errs.join('\n') || 'no errors');
  await browser.close();
  function ZOOIDS(){return ['caa','gate','cap','liseco','pass','sea','mags','conceptor','coast','ot','manifold','gcad'];}
})();
