/* ================= real data: Part 1, Plates 2.7 and 2.10, Part 4.2 and 4.3 ================= */
const D=JSON.parse($('derived-data').textContent);
const GROUPS=['assistant','doctor','consultant','teacher','robot','ghost','bard','pirate','pirate_doctor','pirate_narrated'];
const RINFO={
  assistant:{label:'Assistant',col:'assist',shape:'o'},
  pirate:{label:'Pirate',col:'pirate',shape:'o'},
  pirate_doctor:{label:'Pirate ship\'s doctor',short:'Ship\'s doctor',col:'pirate',shape:'dia',hollow:true},
  pirate_narrated:{label:'Assistant describing a pirate',short:'Describing a pirate',col:'orange',shape:'tri'},
  doctor:{label:'Doctor',col:'other',shape:'o',small:true},
  teacher:{label:'Teacher',col:'other',shape:'o',small:true},
  bard:{label:'Bard',col:'other',shape:'o',small:true},
  robot:{label:'Robot',col:'other',shape:'o',small:true},
  consultant:{label:'Consultant',col:'other',shape:'o',small:true},
  ghost:{label:'Ghost',col:'other',shape:'o',small:true}
};
const keySVG=r=>{const i=RINFO[r];return shapeSVG(i.shape,'var(--'+i.col+')',i.hollow);};
const DRAW_ORDER=['doctor','teacher','bard','robot','consultant','ghost','assistant','pirate','pirate_doctor','pirate_narrated'];
const PT=D.points;
const byRole={};PT.forEach((p,i)=>{(byRole[p.r]=byRole[p.r]||[]).push(i);});
const QTEXT={};PT.forEach(p=>{if(p.r==='assistant'&&p.s===0)QTEXT[p.q]={u:p.u,t:p.t};});
const QIDS=Object.keys(QTEXT).map(Number).sort((a,b)=>a-b);
const trunc=(s,n)=>s.length>n?s.slice(0,n-1).trimEnd()+'…':s;

/* ---------- Plate 1.2: answer reader ---------- */
(function(){
  $('qsel').innerHTML=QIDS.map(q=>'<option value="'+q+'">'+esc(QTEXT[q].t+' · '+QTEXT[q].u)+'</option>').join('');
  $('qsel').value='23';
  const extra=['bard','ghost','robot','doctor','teacher','consultant'];
  $('rolesel').innerHTML=extra.map(r=>'<option value="'+r+'">'+RINFO[r].label+'</option>').join('');
  $('rolesel').value='bard';
  $('qsel').addEventListener('change',upd12);$('rolesel').addEventListener('change',upd12);
})();
function ansCard(p,note){
  const i=RINFO[p.r];
  const badges='<span class="badge">pirate words '+pct1(p.lex)+'</span>'+(p.brk?'<span class="badge flag">says it is an AI</span>':'');
  return '<article class="ans"><header>'+keySVG(p.r)+esc(i.label)+(note?' <span class="who">'+esc(note)+'</span>':'')+badges+'</header>'+
    '<p class="txt">'+esc(p.a.length>=259?p.a.trimEnd()+'…':p.a)+'</p></article>';
}
function upd12(){
  const q=+$('qsel').value,extra=$('rolesel').value;
  const roles=['assistant','pirate','pirate_doctor','pirate_narrated',extra];
  $('answers').innerHTML=roles.map(r=>{
    const p=PT.find(x=>x.r===r&&x.q===q&&x.s===0);if(!p)return '';
    return ansCard(p,r==='pirate_narrated'?'assistant system prompt, asked how a pirate captain would answer':'');
  }).join('');
}

/* ---------- Plate 1.3: the judge decides ---------- */
function upd13(){
  const F=D.filter,g=D.groups.resp,ang=Math.round(Math.acos(Math.max(-1,Math.min(1,F.cos)))/deg);
  tiles($('judgeTiles'),[
    ['Pirate answers that pass',F.n_pass+' of '+F.n_pirate,'needs at least one pirate word'],
    ['Pirate words per answer',pct1(g.pirate_narrated.lex)+' vs '+pct1(g.pirate.lex),'assistant describing a pirate vs the pirate'],
    ['Filtered vs unfiltered direction','cosine '+F.cos.toFixed(2),'the fitted direction turns by '+ang+'°']
  ]);
}

/* ---------- Plate 1.4: layer sweep ---------- */
const XL=D.sweep.resp.map(s=>s.layer);
const LT=(()=>{const n=D.meta.n_layers,st=n<=24?4:n<=36?6:8;return Array.from({length:Math.floor(n/st)+1},(_,i)=>i*st);})();
const sweepSeries=key=>()=>[
  {name:'answer average',short:'avg',color:COL.pirate,values:D.sweep.resp.map(s=>s[key])},
  {name:'last prompt token',short:'last',color:COL.orange,values:D.sweep.last.map(s=>s[key])}];
const chAuc=LineChart('cSweepAuc',{aspect:.78,xs:()=>XL,yRange:()=>[.4,1.02],yTicks:()=>[.5,.6,.7,.8,.9,1],yFmt:v=>v.toFixed(1),
  xTicks:()=>LT,xTitle:'layer',mr:50,refs:()=>[{y:.5,label:'chance'}],series:sweepSeries('lr_auc'),
  tipTitle:i=>'layer '+XL[i],vFmt:v=>'AUC '+v.toFixed(3)});
/* role share tops out near 30% on the 0.5B pilot but passes 50% at SmolLM3's last prompt token */
const VMAX=Math.max(.32,1.05*Math.max(...D.sweep.resp.map(s=>s.role_var),...D.sweep.last.map(s=>s.role_var)));
const chVar=LineChart('cSweepVar',{aspect:.78,xs:()=>XL,yRange:()=>[0,VMAX],
  yTicks:()=>Array.from({length:Math.floor(VMAX*10+1e-9)+1},(_,i)=>i/10),yFmt:v=>Math.round(v*100)+'%',
  xTicks:()=>LT,xTitle:'layer',mr:50,series:sweepSeries('role_var'),
  tipTitle:i=>'layer '+XL[i],vFmt:v=>pct1(v)});
const tabSweep=dataTable($('cSweepVar').closest('.figrow'),'Show the numbers');
tabSweep.set(['layer','AUC, answer avg','AUC, last token','role share, answer avg','role share, last token','question share, answer avg'],
  XL.map((l,i)=>[String(l),D.sweep.resp[i].lr_auc.toFixed(3),D.sweep.last[i].lr_auc.toFixed(3),pct1(D.sweep.resp[i].role_var),pct1(D.sweep.last[i].role_var),pct1(D.sweep.resp[i].q_var)]));

/* ---------- Plate 1.5: persona map ---------- */
const s15={view:'plane',pool:'resp',li:3,hover:-1};
const ML=D.meta.map_layers;
(function(){
  const n=PT.length.toLocaleString('en-US'),th=D.meta.thinned_samples;  /* build.py --samples thins repeat draws */
  $('mapN').textContent=n+' real answers'+(th?', the first '+(th===1?'sample':th+' samples')+' per prompt and question':'');
  $('cMap').setAttribute('aria-label','Scatter of '+n+' answers in the chosen projection');
  const items=[['assistant','Assistant'],['pirate','Pirate'],['pirate_doctor','Pirate ship\'s doctor'],['pirate_narrated','Assistant describing a pirate'],['doctor','Other roles (6)']];
  $('mapLegend').innerHTML=items.map(([r,l])=>'<li>'+keySVG(r)+esc(l)+'</li>').join('');
})();
const coords15=()=>D.maps[s15.pool][String(ML[s15.li])][s15.view];
const M15={l:40,r:14,t:14,b:34};
let scale15=null;
const fig15=Canvas('cMap',.8,(c,w,h)=>{/* scatter */
  const P=coords15();
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  P.forEach(p=>{x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0]);y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1]);});
  if(s15.view==='plane'){x0=Math.min(x0,-.2);x1=Math.max(x1,1.2);}
  const px=(x1-x0)*.05,py=(y1-y0)*.06;x0-=px;x1+=px;y0-=py;y1+=py;
  const X=x=>M15.l+(x-x0)/(x1-x0)*(w-M15.l-M15.r),Y=y=>M15.t+(1-(y-y0)/(y1-y0))*(h-M15.t-M15.b);
  scale15={X,Y};
  c.font='10px '+MONO;
  c.strokeStyle=COL.grid;c.lineWidth=1;c.beginPath();
  const xt=ticks(x0,x1,6),yt=ticks(y0,y1,5);
  xt.forEach(v=>{const x=Math.round(X(v))+.5;c.moveTo(x,M15.t);c.lineTo(x,h-M15.b);});
  yt.forEach(v=>{const y=Math.round(Y(v))+.5;c.moveTo(M15.l,y);c.lineTo(w-M15.r,y);});c.stroke();
  c.fillStyle=COL.muted;c.textAlign='center';c.textBaseline='top';
  xt.forEach(v=>c.fillText(+v.toFixed(2)+'',X(v),h-M15.b+6));
  c.textAlign='right';c.textBaseline='middle';
  yt.forEach(v=>c.fillText(+v.toFixed(2)+'',M15.l-6,Y(v)));
  const ev=D.maps[s15.pool][String(ML[s15.li])].pca_ev;
  const xl=s15.view==='plane'?'persona score (0 = assistant mean, 1 = pirate mean)':'top variance direction 1 ('+pct(ev[0])+' of variance)';
  const yl=s15.view==='plane'?'largest leftover variation':'direction 2 ('+pct(ev[1])+')';
  text(c,xl,w-M15.r,h-8,{size:10,align:'right',halo:false});
  text(c,yl,M15.l+4,M15.t+6,{size:10,halo:false});
  if(s15.view==='plane'){
    [[0,'assistant mean','right',-4],[1,'pirate mean','left',4]].forEach(([v,l,al,dx])=>{const x=Math.round(X(v))+.5;c.setLineDash([4,4]);c.strokeStyle=COL.muted;c.lineWidth=1;c.beginPath();c.moveTo(x,M15.t);c.lineTo(x,h-M15.b);c.stroke();c.setLineDash([]);text(c,l,x+dx,h-M15.b-10,{size:10,align:al});});
  }
  DRAW_ORDER.forEach(r=>{
    const info=RINFO[r];
    (byRole[r]||[]).forEach(i=>{const p=P[i];mark(c,X(p[0]),Y(p[1]),info.shape,info.small?2.6:info.shape==='tri'?3.6:3.3,COL[info.col],info.hollow,true);});
  });
  if(s15.hover>=0){const p=P[s15.hover];c.beginPath();c.arc(X(p[0]),Y(p[1]),8,0,2*Math.PI);c.lineWidth=2;c.strokeStyle=COL.ink;c.stroke();}
},{minH:300});
hoverable(fig15.cv,(px,py,e)=>{
  if(!scale15)return;const P=coords15();let best=-1,bd=144;
  P.forEach((p,i)=>{const dx=scale15.X(p[0])-px,dy=scale15.Y(p[1])-py,d=dx*dx+dy*dy;if(d<bd){bd=d;best=i;}});
  if(best!==s15.hover){s15.hover=best;fig15.draw();}
  if(best<0){hideTip();return;}
  const p=PT[best],q=P[best];
  showTip('<div class="row">'+keySVG(p.r)+'<span class="tv">'+esc(RINFO[p.r].label)+'</span></div><div class="tl">'+esc(p.t)+' · '+esc(trunc(QTEXT[p.q].u,70))+'</div><div>'+esc(trunc(p.a,170))+'</div><div class="tl">'+(s15.view==='plane'?'persona score '+q[0].toFixed(2):'('+q[0].toFixed(2)+', '+q[1].toFixed(2)+')')+'</div>',e.clientX,e.clientY);
},()=>{s15.hover=-1;fig15.draw();hideTip();});
function upd15(){
  out('mapLayer',String(ML[s15.li]));
  const P=coords15();
  if(s15.view==='plane'){
    const m=r=>(byRole[r].reduce((s,i)=>s+P[i][0],0)/byRole[r].length);
    rows($('mapRead'),[['Average persona score',''],['  pirate ship\'s doctor',m('pirate_doctor').toFixed(2)],['  assistant describing a pirate',m('pirate_narrated').toFixed(2)],['  bard',m('bard').toFixed(2)],['  doctor',m('doctor').toFixed(2)]]);
  }else{
    let sq=0,sr=0;
    P.forEach((p,i)=>{let b=-1,bd=1e9;P.forEach((o,j)=>{if(j===i)return;const d=(o[0]-p[0])**2+(o[1]-p[1])**2;if(d<bd){bd=d;b=j;}});if(PT[b].q===PT[i].q)sq++;if(PT[b].r===PT[i].r)sr++;});
    const ev=D.maps[s15.pool][String(ML[s15.li])].pca_ev;
    rows($('mapRead'),[['Two directions explain',pct(ev[0]+ev[1])+' of the variance'],['Nearest neighbour on screen…',''],['  answers the same question',pct(sq/P.length)],['  is the same role',pct(sr/P.length)]]);
  }
  fig15.draw();
}
onSeg('mapView',v=>{s15.view=v;upd15();});
onSeg('mapPool',v=>{s15.pool=v;upd15();});
onRange('mapLayer',v=>{s15.li=v;upd15();});

/* ---------- Plate 1.6: hard-negative strip plot ---------- */
const s16={pool:'resp',hover:null};
const M16={l:184,r:52,t:26,b:30};
let scale16=null;
const fig16=Canvas('cStrip',.62,(c,w,h)=>{
  const G=D.groups[s16.pool];
  let x0=Infinity,x1=-Infinity;GROUPS.forEach(g=>G[g].vals.forEach(v=>{x0=Math.min(x0,v);x1=Math.max(x1,v);}));
  x0=Math.min(x0,-.3);x1=Math.max(x1,1.3);const pad=(x1-x0)*.03;x0-=pad;x1+=pad;
  const L=Math.min(M16.l,Math.max(w*.36,132));
  const X=x=>L+(x-x0)/(x1-x0)*(w-L-M16.r);
  const rowH=(h-M16.t-M16.b)/GROUPS.length,Yc=k=>M16.t+(k+.5)*rowH;
  scale16={X,Yc,rowH};
  c.font='10px '+MONO;
  const xt=ticks(x0,x1,6);
  c.strokeStyle=COL.grid;c.lineWidth=1;c.beginPath();xt.forEach(v=>{const x=Math.round(X(v))+.5;c.moveTo(x,M16.t);c.lineTo(x,h-M16.b);});c.stroke();
  c.fillStyle=COL.muted;c.textAlign='center';c.textBaseline='top';xt.forEach(v=>c.fillText(+v.toFixed(1)+'',X(v),h-M16.b+6));
  [[0,'assistant mean','right',-4],[1,'pirate mean','left',4]].forEach(([v,l,al,dx])=>{const x=Math.round(X(v))+.5;c.setLineDash([4,4]);c.strokeStyle=COL.muted;c.lineWidth=1;c.beginPath();c.moveTo(x,M16.t-4);c.lineTo(x,h-M16.b);c.stroke();c.setLineDash([]);text(c,l,x+dx,M16.t-12,{size:10,align:al});});
  GROUPS.forEach((g,k)=>{
    const y=Yc(k),info=RINFO[g];
    c.font='10px '+MONO;const lab=(info.short&&c.measureText(info.label).width>L-14)?info.short:info.label;  /* the deck scales fonts */
    text(c,lab,L-10,y,{size:10,align:'right',color:COL.ink,halo:false});
    const vals=G[g].vals,r=rng(1000+k);
    vals.forEach((v,j)=>{const jy=(r()-.5)*Math.min(14,rowH*.6);mark(c,X(v),y+jy,info.shape,info.shape==='tri'?3.2:2.8,COL[info.col],info.hollow,true);});
    const mx=X(G[g].mean);c.strokeStyle=COL.ink;c.lineWidth=2.5;c.beginPath();c.moveTo(mx,y-rowH*.42);c.lineTo(mx,y+rowH*.42);c.stroke();
    text(c,G[g].mean.toFixed(2),w-M16.r+8,y,{size:10,color:COL.ink,halo:false});
  });
  if(s16.hover){const [k,j]=s16.hover,v=G[GROUPS[k]].vals[j];c.beginPath();c.arc(X(v),Yc(k),7.5,0,2*Math.PI);c.lineWidth=2;c.strokeStyle=COL.ink;c.stroke();}
},{minH:300,get maxH(){return window.FIG_MAX_H?Math.round(innerHeight*.8):500;}});  /* ten rows: on a slide the strip may take 80% of the height */
hoverable(fig16.cv,(px,py,e)=>{
  if(!scale16)return;const G=D.groups[s16.pool];
  const k=Math.floor((py-M16.t)/scale16.rowH);
  if(k<0||k>=GROUPS.length){if(s16.hover){s16.hover=null;fig16.draw();}hideTip();return;}
  const vals=G[GROUPS[k]].vals;let best=-1,bd=10;
  vals.forEach((v,j)=>{const d=Math.abs(scale16.X(v)-px);if(d<bd){bd=d;best=j;}});
  const nh=best<0?null:[k,best];
  if(JSON.stringify(nh)!==JSON.stringify(s16.hover)){s16.hover=nh;fig16.draw();}
  if(best<0){hideTip();return;}
  const p=PT[byRole[GROUPS[k]][best]];
  showTip('<div class="row">'+keySVG(p.r)+'<span class="tv">'+esc(RINFO[p.r].label)+'</span></div><div class="tl">'+esc(trunc(QTEXT[p.q].u,70))+'</div><div>'+esc(trunc(p.a,170))+'</div><div class="tl">persona score '+vals[best].toFixed(2)+'</div>',e.clientX,e.clientY);
},()=>{s16.hover=null;fig16.draw();hideTip();});
const tab16=dataTable('cStrip','Show group means');
function upd16(){
  const G=D.groups[s16.pool];
  tab16.set(['group','mean','10th pct','90th pct','pirate words'],GROUPS.map(g=>[RINFO[g].label,G[g].mean.toFixed(2),G[g].q10.toFixed(2),G[g].q90.toFixed(2),pct1(G[g].lex)]));
  fig16.draw();
}
onSeg('stripPool',v=>{s16.pool=v;s16.hover=null;upd16();});

/* ---------- Plate 1.7: the persona is a thin slice ---------- */
function upd17(){
  const L=D.meta.steer_hidden,W=D.within;  /* the saved layer nearest the steering layer (exact when all are saved) */
  const at=k=>D.sweep[k].reduce((b,s)=>Math.abs(s.layer-L)<Math.abs(b.layer-L)?s:b),r=at('resp'),l=at('last');
  const oneTask=new Set(PT.map(p=>p.t)).size<2;  /* a single question set has no task split to report */
  tiles($('sliceTiles'),[
    ['Role\'s share of the variance',pct1(r.role_var),'answer average, layer '+r.layer],
    ['Question\'s share',pct(r.q_var),'same activations'],
    oneTask?['Inside the pirate cloud, the question explains',pct(W.pirate.question_frac),'one question set, so no task split']
      :['Inside the pirate cloud, task type explains',pct(W.pirate.task_frac),'question: '+pct(W.pirate.question_frac)],
    ['Role\'s share at the last prompt token',pct1(l.role_var),'layer '+l.layer+', before the answer starts']
  ]);
}

/* ---------- Plate 2.7: the rank trap ---------- */
const RK=D.rank;
function upd27(){
  tiles($('rankTiles'),[
    ['Pirate answers',String(RK.n),'one vector each'],
    ['Dimensions',String(RK.d),D.meta.model.split('/').pop()+' hidden size'],
    ['Rank of the spread',String(RK.rank),'at most n − 1'],
    ['Directions with keep = 0',String(RK.d-RK.rank),'deleted by C·h']
  ]);
  bars($('rankBars'),[
    ['Centered C',RK.kept_C,'var(--pirate)',pct(RK.kept_C)],
    ['NOT (I − C)',RK.kept_notC,'var(--pirate)',pct(RK.kept_notC)],
    ['Blend, β 0.3',RK.kept_interp,'var(--pirate)',pct(RK.kept_interp)],
    ['Raw C, all',RK.raw_keep_total,'var(--none)',pct(RK.raw_keep_total)],
    ['Raw C, varying part',RK.raw_keep_content,'var(--none)',pct(RK.raw_keep_content)]
  ],1,false);
  const n=document.createElement('p');n.className='note';
  n.textContent='Grey rows: a raw (uncentered) fit seems to keep '+pct(RK.raw_keep_total)+' of an assistant answer, but most of that is the offset every answer shares. Of the part that varies between answers, it keeps '+pct(RK.raw_keep_content)+'.';
  const old=$('rankBars').nextElementSibling;if(old&&old.classList.contains('note'))old.remove();
  $('rankBars').insertAdjacentElement('afterend',n);
}
const XC=RK.cover.map(c=>c.n);
const XCT=(()=>{const mx=Math.max(...XC),st=[8,16,32,64,128,256,512,1024,2048,4096].find(s=>mx/s<=5)||8192,t=[XC[0]];
  for(let v=st;v<mx-st/3;v+=st)t.push(v);t.push(mx);return t;})();  // 0.5B run: 2, 16, 32, 48, 64, 71
const chCover=LineChart('cCover',{aspect:.5,xs:()=>XC,xMin:0,xMax:Math.max(...XC)+1,yRange:()=>[0,1],yTicks:()=>[0,.25,.5,.75,1],yFmt:v=>Math.round(v*100)+'%',
  xTicks:()=>XCT,xTitle:'n training answers',mr:18,direct:false,
  series:()=>[{name:'share inside the span',color:COL.pirate,values:RK.cover.map(c=>c.inside),dots:true}],
  tipTitle:i=>'n = '+XC[i],vFmt:v=>pct1(v)});
dataTable('cCover','Show the numbers').set(['n','share inside span'],RK.cover.map(c=>[String(c.n),pct1(c.inside)]));

/* ---------- Plate 2.10: conceptor logic on real data ---------- */
const LOGIC_LAYERS=D.logic?Object.keys(D.logic.sims).sort((a,b)=>a-b):[];
const s210={fit:'diff',layer:LOGIC_LAYERS.includes(String(D.meta.steer_hidden))?String(D.meta.steer_hidden):LOGIC_LAYERS[0]};
$('logicLayer').replaceChildren(...LOGIC_LAYERS.map(l=>{const b=document.createElement('button');b.type='button';b.dataset.v=l;
  b.setAttribute('aria-pressed',String(l===s210.layer));b.textContent='Layer '+l;return b;}));
const LOGIC_LABEL={'pirate AND doctor':'Pirate AND doctor','pirate OR doctor':'Pirate OR doctor',pirate:'Pirate',doctor:'Doctor',assistant:'Assistant',bard:'Bard',ghost:'Ghost',robot:'Robot',teacher:'Teacher',consultant:'Consultant'};
function upd210(){
  const S=D.logic.sims[s210.layer][s210.fit],keys=Object.keys(S);
  bars($('logicBars'),keys.map(k=>[LOGIC_LABEL[k]||k,S[k],/AND|OR/.test(k)?'var(--orange)':'var(--pirate)',S[k].toFixed(3)]),1,false);
  let cap;
  if(s210.fit==='diff'){
    const and=S['pirate AND doctor'],beat=keys.filter(k=>!/AND/.test(k)&&S[k]>and&&k!=='pirate'&&k!=='pirate OR doctor').map(k=>LOGIC_LABEL[k].toLowerCase());
    cap='Orange: combinations computed from the pirate and doctor conceptors. Teal: conceptors fitted on one role. Pirate alone scores '+S.pirate.toFixed(2)+', OR '+S['pirate OR doctor'].toFixed(2)+', AND '+and.toFixed(2)+(beat.length?', below '+beat.join(', '):'')+'.';
  }else{
    const mn=Math.min(...keys.map(k=>S[k]));
    cap='Orange: combinations. Teal: single roles. Every conceptor scores at least '+mn.toFixed(2)+': fitted on raw clouds they mostly learn the shared questions, so they all look alike.';
  }
  $('logicCap').textContent=cap;
}
onSeg('logicFit',v=>{s210.fit=v;upd210();});
onSeg('logicLayer',v=>{s210.layer=v;upd210();});

/* ---------- Plate 4.2: real multi-turn pilot ---------- */
const MT=D.multiturn.rows;
const CONDS=[['none','prompt only','prompt','none',[5,4]],['add','add vector','add','orange'],['cap','cap','cap','assist'],['conceptor','conceptor','conceptor','pirate']];
const KEYT={switch:{5:'"drop the act"',9:'"are you an AI?"',12:'"who are you?"'},benign:{},replace:{3:'"you are a teacher now"',12:'"your name?"'}};
const KEYT_ROW=t=>({5:0,9:1,12:2,3:0}[t]||0);
const s42={conv:'switch',sel:11};
const rowsOf=(c,v)=>MT.filter(r=>r.c===c&&r.v===v).sort((a,b)=>a.k-b.k);
const XT=Array.from({length:12},(_,i)=>i+1);
const yMT=(()=>{const v=MT.map(r=>r.s18);return [Math.floor(Math.min(...v,-.2)*2)/2,Math.ceil(Math.max(...v)*2)/2];})();
const chMT=LineChart('cMT',{aspect:.56,canvasOpt:{minH:260},xs:()=>XT,yRange:()=>yMT,yFmt:v=>v.toFixed(1),xTitle:'turn',mr:84,
  refs:()=>[{y:0,label:'assistant mean'},{y:1,label:'pirate mean'}],
  bands:()=>Object.keys(KEYT[s42.conv]).map(t=>({x0:+t-.06,x1:+t+.06,color:COL.hair,alpha:1,label:KEYT[s42.conv][t],row:s42.conv==='replace'&&+t===12?1:KEYT_ROW(+t)})),
  series:()=>CONDS.map(k=>{const R=rowsOf(k[0],s42.conv);return {name:k[1],short:k[2],color:COL[k[3]],dash:k[4],values:R.map(r=>r.s18),rings:R.map((r,i)=>r.brk?i:-1).filter(i=>i>=0)};}),
  selected:()=>s42.sel,
  tipTitle:i=>'turn '+(i+1),
  tipExtra:i=>{const u=rowsOf('none',s42.conv)[i].u;return '<div class="tl" style="margin-top:4px;white-space:normal">user: '+esc(trunc(u,90))+'</div><div class="tl">click to read the answers</div>';},
  vFmt:(v,s)=>v.toFixed(2),
  onPick:i=>{s42.sel=i;upd42();}});
const tab42=dataTable('cMT','Show the persona scores');
function upd42(){
  chMT.f.draw();
  const i=s42.sel,u=rowsOf('none',s42.conv)[i].u;
  $('mtAns').innerHTML='<p class="annot">Turn '+(i+1)+' · user: '+esc(u)+'</p>'+CONDS.map(k=>{
    const r=rowsOf(k[0],s42.conv)[i];
    return '<article class="ans"><header><span class="key line" style="background:var(--'+k[3]+')"></span>'+esc(k[1])+
      '<span class="badge">persona score '+r.s18.toFixed(2)+'</span><span class="badge">pirate words '+pct1(r.lex)+'</span>'+(r.brk?'<span class="badge flag">says it is an AI</span>':'')+'</header>'+
      '<p class="txt">'+esc(r.a.length>=299?r.a.trimEnd()+'…':r.a)+'</p></article>';
  }).join('');
  tab42.set(['turn','user'].concat(CONDS.map(k=>k[2])),XT.map((t,j)=>[String(t),trunc(rowsOf('none',s42.conv)[j].u,48)].concat(CONDS.map(k=>{const r=rowsOf(k[0],s42.conv)[j];return r.s18.toFixed(2)+(r.brk?' (AI)':'');}))));
}
function tiles42(){
  tiles($('mtTiles'),CONDS.map(k=>{const R=MT.filter(r=>r.c===k[0]),b=R.filter(r=>r.brk).length,lx=R.reduce((s,r)=>s+r.lex,0)/R.length;
    return [k[1],b+' / '+R.length,'answers that say they are an AI · pirate words '+pct1(lx)];}));
}
onSeg('mtConv',v=>{s42.conv=v;upd42();});

/* ---------- Plate 4.3: published GCAD numbers ---------- */
const XG=[1,5,10];
const GC_COH={vec:[76.6,65.6,57.9],gcad:[88.6,88.2,86.7]},GC_TR={vec:[86.4,81.9,78.0],gcad:[91.5,91.8,93.1]};
const gSeries=S=>()=>[{name:'residual persona vector',short:'vector',color:COL.orange,values:S.vec,dots:true},{name:'GCAD',short:'GCAD',color:COL.pirate,values:S.gcad,dots:true}];
const gOpt={aspect:.72,xs:()=>XG,xMin:1,xMax:10,yRange:()=>[50,100],yTicks:()=>[50,60,70,80,90,100],yFmt:v=>String(v),xTicks:()=>XG,xFmt:v=>'turn '+v,mr:66,tipTitle:i=>'turn '+XG[i],vFmt:v=>v.toFixed(1)};
LineChart('cGcadC',Object.assign({},gOpt,{series:gSeries(GC_COH)}));
LineChart('cGcadT',Object.assign({},gOpt,{series:gSeries(GC_TR)}));
dataTable($('cGcadT').closest('.figrow'),'Show the numbers').set(['turn','coherence, vector','coherence, GCAD','trait, vector','trait, GCAD'],XG.map((t,i)=>[String(t),GC_COH.vec[i].toFixed(1),GC_COH.gcad[i].toFixed(1),GC_TR.vec[i].toFixed(1),GC_TR.gcad[i].toFixed(1)]));

/* ---------- part nav: mark the part in view ---------- */
(function(){
  const ids=['top','data','conceptors','zoo','turns','recipe'],links=ids.map(id=>document.querySelector('.partnav a[href="#'+id+'"]'));
  let ticking=false;
  const set=()=>{ticking=false;let cur=0;ids.forEach((id,i)=>{const el=$(id);if(el&&el.getBoundingClientRect().top<140)cur=i;});links.forEach((a,i)=>a&&a.classList.toggle('on',i===cur));};
  window.addEventListener('scroll',()=>{if(!ticking){ticking=true;requestAnimationFrame(set);}},{passive:true});
  set();
})();

/* ---------- boot ---------- */
function bootAll(){
  upd1();upd2();cfg3();upd3();upd4();upd5();upd6();upd8();upd9();updZoo();upd41();
  upd12();upd13();chAuc.f.draw();chVar.f.draw();upd15();upd16();upd17();upd27();chCover.f.draw();upd210();upd42();tiles42();
}
bootAll();
const redraw=()=>{readColors();figs.forEach(f=>f.draw());};
if(window.matchMedia)matchMedia('(prefers-color-scheme: dark)').addEventListener('change',redraw);
new MutationObserver(redraw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
if(document.fonts&&document.fonts.ready)document.fonts.ready.then(redraw);
