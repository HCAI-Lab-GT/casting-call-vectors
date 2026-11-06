/* ================= Deck results slides: the problem, setting, why, holding over turns, trajectories, judges ================= */
const DK=JSON.parse($('deck-data').textContent);
if(DK.texts)for(const pd of Object.values(DK.personas))for(const rows of Object.values(pd.hold))for(const h of rows)
  for(const e of h.examples){if(typeof e.q==='number')e.q=DK.texts[e.q];if(typeof e.a==='number')e.a=DK.texts[e.a];}
const PERSONAS=Object.keys(DK.personas);
const PNAME=p=>(DK.personas[p]&&DK.personas[p].label)||p.charAt(0).toUpperCase()+p.slice(1);
const ROLE=p=>p.replace(/^[a-z0-9]+_/,'');  /* 'olmo_pirate' -> 'pirate' */
const MODEL=p=>(DK.personas[p]&&DK.personas[p].model)||'SmolLM3-3B';
let PERS=DK.default_persona in DK.personas?DK.default_persona:PERSONAS[0];
const PD=()=>DK.personas[PERS];
const JALL=['qwen','gemma','mistral','llama'];
const JUDGES_OF=p=>JALL.filter(j=>DK.personas[p].hold[j]&&DK.personas[p].hold[j].length);
const JUDGES=()=>JUDGES_OF(PERS);
const JNAME={qwen:'Qwen3-30B-A3B',gemma:'Gemma-3-27B',mistral:'Mistral-Small-3.2',llama:'Llama-3.3-70B',mean:'Mean of judges'};
const JSHORT={qwen:'Qwen',gemma:'Gemma',mistral:'Mistral',llama:'Llama',mean:'Mean'};
/* hue = what holds the persona (teal conceptor, blue cap, orange vector); dotted = with the vector added */
const METH={
  none:['No steering','none','none',[5,4]], add:['Steering vector','vector','orange',null],
  cap:['Activation cap, one layer','cap','assist',null], cap_aa:['Activation cap, Assistant Axis style','AA cap','assist',[9,4]],
  add_cap:['Vector + cap','vec+cap','assist',[2,4]], conceptor_tok:['Per-token conceptor','conceptor','pirate',null],
  add_ctok:['Vector + per-token conceptor','vec+con','pirate',[2,4]], meanshift_tok:['Mean-shift control, per token','mean-shift','other',[6,3]],
  conceptor:['Conceptor, answer-average fit','con (avg)','pirate',[8,4]], lowrank:['Low-rank conceptor','low-rank','pirate',[2,6]],
  meanshift:['Mean-shift control, answer-average','ms (avg)','other',[2,3]],
  ctok_cap:['Per-token conceptor + cap','con+cap','assist',[12,3,2,3]],
  ctok_tight:['Per-token conceptor, tight aperture','con tight','pirate',[10,3]],
  ctok_k3:['Per-token conceptor, aperture ×3','con ×3','pirate',[3,3]],
  ctok_k30:['Per-token conceptor, aperture ×30','con ×30','pirate',[14,4]],
  ctok_k100:['Per-token conceptor, aperture ×100','con ×100','pirate',[1,3]],
  region:['Region: cap plus an identity floor','region','assist',[3,2]],
  diffpull:['Difference-subspace conceptor','diff-con','pirate',[4,2,1,2]]};
const MORDER=['none','add','cap','cap_aa','add_cap','ctok_cap',  /* region stays on the Next slide with the other follow-ups */'conceptor_tok','ctok_k3','ctok_tight','ctok_k30','ctok_k100',
  'diffpull','add_ctok','meanshift_tok','conceptor','lowrank','meanshift'];
/* contexts: the two main ones, plus follow-up runs with other probes (steer_eval --probe_set), shown when present */
const CTXS=[['prompt','Prompt + steering'],['steer_only','Steering only'],['prompt_none','Prompt, no pushes'],
  ['prompt_unseen','Prompt, new probe wording']];
const hasCtx=(p,c)=>JUDGES_OF(p).some(j=>DK.personas[p].hold[j].some(h=>h.context===c));
const DEFAULT_ON=new Set(['none','add','cap','cap_aa','ctok_cap','conceptor_tok','ctok_tight']);
const MEASURES=[['fully_rate','In character'],['coherence','Coherence'],['identity_ai_rate','Admits being an AI']];
const PROBE_KIND={5:'drop the act',13:'drop the act',8:'are you an AI?',16:'are you an AI?'};
const pct=v=>v==null||!isFinite(v)?'–':Math.round(v*100)+'%';
const num=v=>v==null||!isFinite(v)?'–':String(Math.round(v));
const fmtM=(m,v)=>m==='coherence'?num(v):(m==='x_pre'||m==='x_post'||m==='off_pre'||m==='off_post'||m==='read_steered'||m==='readout')?(v==null?'–':(+v).toFixed(2)):pct(v);
const mean=a=>{const b=a.filter(v=>v!=null&&isFinite(v));return b.length?b.reduce((s,x)=>s+x,0)/b.length:null;};
const holdRowP=(p,j,ctx,m)=>((DK.personas[p]||{hold:{}}).hold[j]||[]).find(h=>h.context===ctx&&h.method===m);
const holdRow=(j,ctx,m)=>holdRowP(PERS,j,ctx,m);
const turnValP=(p,j,ctx,m,t,key)=>{const h=holdRowP(p,j,ctx,m);if(!h)return null;const q=h.per_turn.find(x=>x.turn===t-1);return q?q[key]:null;};
const turnVal=(j,ctx,m,t,key)=>turnValP(PERS,j,ctx,m,t,key);
const NT=()=>PD().n_turns||10;
const XS=()=>Array.from({length:NT()},(_,i)=>i+1);
const probesNow=ctx=>{const j=JUDGES()[0],rows=j?(PD().hold[j]||[]):[];const h=rows.find(r=>r.context===(ctx||'prompt'))||rows[0];
  return h?h.per_turn.filter(t=>t.probe).map(t=>t.turn+1):[5,8];};
const series=(judge,ctx,m,key)=>XS().map(t=>judge==='mean'?mean(JUDGES().map(j=>turnVal(j,ctx,m,t,key))):turnVal(judge,ctx,m,t,key));
const methodsIn=ctx=>MORDER.filter(m=>JUDGES().some(j=>holdRow(j,ctx,m)));
const nOf=(ctx,m)=>{const h=holdRow(JUDGES()[0],ctx,m);return h&&h.per_turn[0]?h.per_turn[0].n:null;};
const swatchI=m=>{const d=METH[m][3];return '<i class="'+(d?(d[0]<=2?'dot':'dash'):'')+'" style="color:'+COL[METH[m][2]]+'"></i>';};
const lineSwatch=m=>{const d=METH[m][3];return '<i style="display:inline-block;width:18px;height:0;margin-right:8px;vertical-align:middle;border-top:3px '+
  (d?(d[0]<=2?'dotted':'dashed'):'solid')+' '+COL[METH[m][2]]+'"></i>';};
const probeBands=ctx=>probesNow(ctx).map(t=>({x0:t-.45,x1:t+.45,color:rgba(COL.probe,.09),alpha:1}));
const probeNote=ctx=>{const pr=probesNow(ctx);if(!pr.length)return 'No probes in these conversations: every turn is an ordinary question.';
  return 'Shaded: '+pr.map(t=>'turn '+t+' "'+(PROBE_KIND[t]||'probe')+'"').join(', ')+(ctx==='prompt_unseen'?', in new wordings':'')+'.';};
const segHTML=(id,opts,cur)=>'<div class="seg" id="'+id+'" role="group">'+opts.map(x=>'<button type="button" data-v="'+x[0]+'" aria-pressed="'+(x[0]===cur)+'">'+esc(x[1])+'</button>').join('')+'</div>';

/* one persona switch for the whole deck: every results slide follows it */
const personaHooks=[];
function personaSeg(id){
  if(PERSONAS.length<2)return '';
  return '<div class="ctlrow"><span>Persona</span>'+segHTML(id,PERSONAS.map(p=>[p,PNAME(p)]),PERS).replace('role="group"','role="group" data-persona="1" aria-label="Persona"')+'</div>';
}
function setPersona(p){
  if(!(p in DK.personas))return;
  PERS=p;
  document.querySelectorAll('[data-persona] button[data-v]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===p)));
  personaHooks.forEach(fn=>fn());
}
document.addEventListener('click',e=>{const b=e.target.closest('[data-persona] button[data-v]');if(b)setPersona(b.dataset.v);});

/* ---------- slide 2: a prompted persona, turn by turn ---------- */
(function(){
  const box=$('s2turns');if(!box)return;
  let turn=1;
  const ex=j=>{const h=holdRow(j,'prompt','none');return h?h.examples[turn-1]:null;};
  function buttons(){
    const pr=probesNow();
    box.innerHTML=XS().map(t=>'<button type="button" data-v="'+t+'" aria-pressed="'+(t===turn)+'"'+(pr.includes(t)?' class="probe" title="'+(PROBE_KIND[t]||'probe')+'"':'')+'>'+t+'</button>').join('');
  }
  function chat(){
    const js=JUDGES(),base=ex(js[js.length-1])||ex(js[0]);
    if(!base){$('s2chat').textContent='No transcript in the data.';return;}
    const pills=js.map(j=>{const e=ex(j);if(!e)return '';
      const good=e.label==='fully',ai=e.identity==='ai';
      return '<span class="pill '+(good?'good':'bad')+'">'+JSHORT[j]+': '+esc(e.label||'–')+'</span>'+(e.identity?'<span class="pill'+(ai?' bad':'')+'">'+JSHORT[j]+': '+(ai?'admits AI':'no AI claim')+'</span>':'');}).join('');
    $('s2chat').innerHTML='<div class="msg user"><span class="who">User · turn '+turn+(PROBE_KIND[turn]&&probesNow().includes(turn)?' · probe':'')+'</span>'+esc(base.q)+'</div>'+
      '<div class="msg bot"><span class="who">'+esc(MODEL(PERS))+', '+esc(ROLE(PERS))+' prompt, no steering</span>'+esc(base.a)+(base.a.length>=500?' …':'')+'</div>'+
      '<div class="pills">'+pills+'</div>';
  }
  const ch=LineChart('s2line',{aspect:.56,ml:64,mr:24,xs:XS,yRange:()=>[0,1],yTicks:()=>[0,.5,1],yFmt:pct,xFmt:String,
    bands:probeBands,selected:()=>turn-1,direct:false,vFmt:v=>pct(v),tipTitle:i=>'turn '+(i+1),
    series:()=>[...JUDGES().map(j=>({name:JNAME[j],color:rgba(COL.muted,.45),width:1.2,values:series(j,'prompt','none','fully_rate')})),
      {name:'Mean of judges',color:COL.ink,width:2.6,dots:true,values:series('mean','prompt','none','fully_rate')},
      ...(hasCtx(PERS,'prompt_none')?[{name:'Same questions, no pushes (mean of judges)',color:COL.pirate,width:2.4,dash:[6,4],
        values:series('mean','prompt_none','none','fully_rate')}]:[])],
    onPick:i=>{turn=i+1;setSeg('s2turns',String(turn));chat();ch.f.draw();}});
  box.addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(!b)return;turn=+b.dataset.v;setSeg('s2turns',String(turn));chat();ch.f.draw();});
  const pb=$('s2p');if(pb)pb.innerHTML=personaSeg('s2ps');
  personaHooks.push(()=>{turn=Math.min(turn,NT());buttons();chat();ch.f.draw();});
  buttons();chat();
})();

/* ---------- slide 9: setting a persona, in character against coherence ---------- */
(function(){
  const cv=$('s9c');if(!cv)return;
  const rows=()=>PD().sweep||[];
  let on=new Set(['none','add','cap','cap_aa','add_cap','conceptor_tok','add_ctok']);
  const tog=$('s9tog');
  function togs(){const ms=MORDER.filter(m=>rows().some(r=>r.method===m));
    tog.innerHTML=ms.map(m=>'<button type="button" data-m="'+m+'" aria-pressed="'+on.has(m)+'">'+swatchI(m)+esc(METH[m][0])+'</button>').join('');}
  let pts=[],hover=null,zoom=false;
  const f=Canvas('s9c',.68,(c,w,h)=>{
    const x0=zoom?60:0,M={l:64,r:18,t:16,b:40},X=v=>M.l+(Math.max(x0,v)-x0)/(100-x0)*(w-M.l-M.r),Y=v=>M.t+(1-v)*(h-M.t-M.b);
    c.strokeStyle=COL.grid;c.lineWidth=1;c.font='10px '+MONO;c.fillStyle=COL.muted;
    [0,.25,.5,.75,1].forEach(v=>{c.beginPath();c.moveTo(M.l,Y(v)+.5);c.lineTo(w-M.r,Y(v)+.5);c.stroke();c.textAlign='right';c.textBaseline='middle';c.fillText(pct(v),M.l-8,Y(v));});
    (zoom?[60,70,80,90,100]:[0,25,50,75,100]).forEach(v=>{c.textAlign='center';c.textBaseline='top';c.fillText(String(v),X(v),h-M.b+8);});
    c.textAlign='right';c.textBaseline='bottom';c.fillText('coherence (judge, 0-100) →',w-M.r,h-2);
    c.textAlign='left';c.textBaseline='top';c.fillText('in character ↑',M.l+4,2);
    pts=[];
    MORDER.filter(m=>on.has(m)).forEach(m=>{
      const rs=rows().filter(r=>r.method===m&&r.coherence!=null&&r.fully_rate!=null).sort((a,b)=>a.strength-b.strength);
      if(!rs.length)return;
      const col=COL[METH[m][2]];
      c.strokeStyle=col;c.lineWidth=2;c.setLineDash(METH[m][3]||[]);c.beginPath();
      rs.forEach((r,i)=>{const x=X(r.coherence),y=Y(r.fully_rate);i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();c.setLineDash([]);
      const sel=PD().selected&&PD().selected[m]&&PD().selected[m].strength;
      rs.forEach(r=>{const x=X(r.coherence),y=Y(r.fully_rate);pts.push({x,y,r,m});
        mark(c,x,y,'o',3.6,col,false,true);
        if(sel!=null&&Math.abs(r.strength-sel)<1e-9){c.beginPath();c.arc(x,y,8,0,2*Math.PI);c.lineWidth=1.8;c.strokeStyle=col;c.stroke();}});
      const last=rs[rs.length-1];text(c,METH[m][1],X(last.coherence)+8,Y(last.fully_rate)-9,{size:10,color:COL.ink});
    });
    if(hover){c.beginPath();c.arc(hover.x,hover.y,6,0,2*Math.PI);c.lineWidth=2;c.strokeStyle=COL.ink;c.stroke();}
  });
  hoverable(cv,(px,py,e)=>{
    let best=null,bd=22*22;
    pts.forEach(p=>{const d=(p.x-px)**2+(p.y-py)**2;if(d<bd){bd=d;best=p;}});
    if(best!==hover){hover=best;f.draw();}
    if(!best){hideTip();return;}
    const r=best.r,P=['cap','cap_aa','ctok_cap'].includes(best.m)?'percentile ':(['add','add_ctok','add_cap'].includes(best.m)?'c ':'β ');
    showTip('<div class="tv">'+esc(METH[best.m][0])+(best.m==='none'?'':' · '+P+r.strength)+'</div>'+
      '<div class="row"><span class="tl">fully in character</span>&nbsp;<b>'+pct(r.fully_rate)+'</b></div>'+
      '<div class="row"><span class="tl">coherence</span>&nbsp;<b>'+num(r.coherence)+'</b></div>'+
      (r.voice_as_rate!=null?'<div class="row"><span class="tl">speaks as the role</span>&nbsp;<b>'+pct(r.voice_as_rate)+'</b></div>':''),e.clientX,e.clientY);
  },()=>{hover=null;f.draw();hideTip();});
  tog.addEventListener('click',e=>{const b=e.target.closest('button[data-m]');if(!b)return;const m=b.dataset.m;
    on.has(m)?on.delete(m):on.add(m);b.setAttribute('aria-pressed',String(on.has(m)));f.draw();});
  onSeg('s9z',v=>{zoom=v==='zoom';f.draw();});
  function note(){
    const rs=rows(),sel=PD().selected||{};
    const at=m=>{const s=sel[m]&&sel[m].strength;return s==null?null:rs.find(r=>r.method===m&&Math.abs(r.strength-s)<1e-9);};
    const a=at('add'),con=rs.filter(r=>r.method==='conceptor_tok').sort((x,y)=>y.strength-x.strength)[0];
    $('s9note').textContent=(a?'Steering vector at strength '+a.strength+': '+pct(a.fully_rate)+' in character, coherence '+num(a.coherence)+'. ':'')+
      (con?'Per-token conceptor at its highest strength (β = '+con.strength+'): '+pct(con.fully_rate)+'.':'');
  }
  const pb=$('s9p');if(pb)pb.innerHTML=personaSeg('s9ps');
  personaHooks.push(()=>{togs();note();f.draw();});
  togs();note();
})();

/* ---------- slide 10: why a centered conceptor can't close the gap (toy) ---------- */
(function(){
  if(!$('s10c'))return;
  const s={a:+$('s10a').value,mode:'along',h:null};
  const MA=[-.75,-.5],MP=[.55,.45],V=sub(MP,MA),VH=unit(V);
  const ZP=gaussians(150,41),ZA=gaussians(90,7);
  const A=cloud(ZA,MA,mm(rot(20*deg),diag(.13,.1)));
  s.h=MA.slice();
  const shape=()=>{const ang=Math.atan2(VH[1],VH[0])+(s.mode==='along'?0:Math.PI/2);return mm(rot(ang),diag(.55,.09));};
  const alpha=()=>Math.pow(10,(s.a-500)/320);
  const BND=[-1.55,1.55,-1.25,1.35];
  const fig=Fig('s10c',.82,BND,(c,Vw)=>{
    const L=shape(),P=cloud(ZP,MP,L),C=conceptor(corr(P,MP),alpha());
    seg(c,Vw,sub(MA,sc(VH,.6)),add(MP,sc(VH,.9)),COL.muted,1.2,[5,5]);
    label(c,Vw,'persona direction',add(MP,sc(VH,.72)),{align:'right',dy:-14,color:COL.muted});
    dots(c,Vw,A,COL.other,2.2,.7);
    poly(c,Vw,shapeEll(mm(L,T(L)),MP,2),{fill:COL.wash});
    dots(c,Vw,P,COL.pirate,2.3,.6);
    poly(c,Vw,imgCircle(C,MP,1),{stroke:COL.pirate,width:2});
    const out=add(MP,mv(C,sub(s.h,MP)));
    arrow(c,Vw,s.h,out,COL.ink,2.4,11);
    mark(c,Vw.X(s.h[0]),Vw.Y(s.h[1]),'o',6,COL.probe,false,true);
    mark(c,Vw.X(MP[0]),Vw.Y(MP[1]),'dia',6,COL.ink,false,true);
    label(c,Vw,'persona center μ',MP,{dx:10,dy:16,color:COL.ink});
    label(c,Vw,'h (drag me)',s.h,{dx:10,dy:-14,color:COL.probe});
    const g=sub(MP,s.h),rem=mv(C,g),closed=1-dot(VH,rem)/dot(VH,g),keep=dot(VH,mv(C,VH));
    rows($('s10r'),[['kept along the persona direction',pct(keep)],['gap closed along it',isFinite(closed)?pct(closed):'–'],
      ['aperture α',alpha().toFixed(2)]]);
  });
  draggable(fig,()=>s.h,p=>{s.h=p;fig.draw();},BND);
  onRange('s10a',v=>{s.a=v;out('s10a',alpha().toFixed(2));fig.draw();});out('s10a',alpha().toFixed(2));
  onSeg('s10m',v=>{s.mode=v;fig.draw();});
})();

/* ---------- slides 11-13: holding over the conversation ---------- */
document.querySelectorAll('.sbody.hold').forEach(root=>{
  const id=root.dataset.hold,st={measure:root.dataset.measure,ctx:root.dataset.context,judge:'mean',
    on:new Set(DEFAULT_ON),turn:+(root.dataset.transcript||0)};
  const ctxs=CTXS.filter(c=>PERSONAS.some(p=>hasCtx(p,c[0])));
  root.innerHTML='<div class="scol">'+personaSeg(id+'ps')+
    '<div class="ctlrow"><span>Context</span>'+segHTML(id+'c',ctxs,st.ctx)+'</div>'+
    '<div class="ctlrow"><span>Measure</span>'+segHTML(id+'m',MEASURES,st.measure)+'</div>'+
    '<div class="ctlrow"><span>Judge</span><div id="'+id+'jw"></div></div>'+
    '<div class="ctlrow"><span>Methods</span><div class="mtog" id="'+id+'t"></div></div>'+
    '<p class="note" id="'+id+'n"></p></div>'+
    '<div><canvas id="'+id+'cv" role="img" aria-label="Per-turn hold curves for each method"></canvas>'+
    '<p class="note" id="'+id+'pn"></p>'+
    (st.turn?'<div class="chat" id="'+id+'x" style="margin-top:8px;max-height:24vh;overflow:auto"></div>':'')+'</div>';
  const tog=$(id+'t');
  function togs(){
    const ms=methodsIn(st.ctx);
    tog.innerHTML=ms.map(m=>'<button type="button" data-m="'+m+'" aria-pressed="'+st.on.has(m)+'" title="'+(nOf(st.ctx,m)||'')+' conversations">'+swatchI(m)+esc(METH[m][0])+'</button>').join('');
    if(st.judge!=='mean'&&!JUDGES().includes(st.judge))st.judge='mean';
    $(id+'jw').innerHTML=segHTML(id+'j',[['mean','Mean'],...JUDGES().map(j=>[j,JSHORT[j]])],st.judge);
  }
  const yr=()=>st.measure==='coherence'?[0,100]:[0,1];
  const ch=LineChart(id+'cv',{aspect:st.turn?.42:.58,ml:64,mr:24,direct:false,xs:XS,yRange:yr,yTicks:()=>st.measure==='coherence'?[0,25,50,75,100]:[0,.25,.5,.75,1],
    yFmt:v=>fmtM(st.measure,v),xFmt:t=>String(t),xTicks:()=>XS().filter(t=>NT()<=12||t===1||t%5===0||probesNow(st.ctx).includes(t)),
    bands:()=>probeBands(st.ctx),selected:()=>st.turn?st.turn-1:-1,
    tipTitle:i=>'turn '+(i+1)+(probesNow(st.ctx).includes(i+1)?' · '+(PROBE_KIND[i+1]||'probe'):''),vFmt:v=>fmtM(st.measure,v),
    series:()=>methodsIn(st.ctx).filter(m=>st.on.has(m)).map(m=>({name:METH[m][0],short:METH[m][1],color:COL[METH[m][2]],
      dash:METH[m][3]||undefined,width:m==='conceptor_tok'||m==='add'||m==='cap_aa'?2.6:2,values:series(st.judge,st.ctx,m,st.measure)})),
    onPick:i=>{st.turn=i+1;redraw();}});
  function transcript(){
    const x=$(id+'x');if(!x)return;
    const js=JUDGES(),j=st.judge==='mean'?js[js.length-1]:st.judge;
    const ms=methodsIn(st.ctx).filter(m=>st.on.has(m));
    const first=ms.map(m=>holdRow(j,st.ctx,m)).find(h=>h&&h.examples[st.turn-1]);
    x.innerHTML=(first?'<div class="msg user"><span class="who">User · turn '+st.turn+'</span>'+esc(first.examples[st.turn-1].q)+'</div>':'')+
      ms.map(m=>{const h=holdRow(j,st.ctx,m);const e=h&&h.examples[st.turn-1];if(!e)return '';
        return '<div class="msg bot" style="border-left-color:'+COL[METH[m][2]]+'"><span class="who">'+esc(METH[m][0])+' · '+JSHORT[j]+': '+esc(e.label||'–')+
          (e.identity?' · '+(e.identity==='ai'?'admits AI':'no AI claim'):'')+'</span>'+esc(e.a.slice(0,320))+(e.a.length>320?' …':'')+'</div>';}).join('');
  }
  function note(){
    if(!JUDGES().length){$(id+'n').textContent=PNAME(PERS)+': the 20-turn hold runs aren\'t judged yet, so there is nothing to plot.';$(id+'pn').textContent='';return;}
    if(!methodsIn(st.ctx).length){$(id+'n').textContent=PNAME(PERS)+' has no runs in this context.';$(id+'pn').textContent='';return;}
    const parts=methodsIn(st.ctx).filter(m=>st.on.has(m)).map(m=>{
      const v=st.judge==='mean'?mean(JUDGES().map(j=>{const h=holdRow(j,st.ctx,m);return h&&h.ordinary[st.measure];})):(holdRow(st.judge,st.ctx,m)||{ordinary:{}}).ordinary[st.measure];
      return METH[m][1]+' '+fmtM(st.measure,v);});
    $(id+'n').textContent=PNAME(PERS)+', turns without a probe, '+(st.judge==='mean'?'mean of the judges':JNAME[st.judge])+': '+parts.join(' · ')+'.';
    $(id+'pn').textContent=probeNote(st.ctx)+' Click a turn'+(st.turn?' to read conversation 1 at that turn.':' to mark it.')+' Mean: average of '+JUDGES().filter(j=>(PD().hold[j]||[]).some(h=>h.context===st.ctx)).length+' judges.';
  }
  function redraw(){ch.f.draw();transcript();note();}
  togs();
  root.addEventListener('click',e=>{
    const b=e.target.closest('button[data-v]');if(!b)return;const box=b.closest('.seg');if(!box||box.hasAttribute('data-persona'))return;
    if(box.id===id+'c'){st.ctx=b.dataset.v;setSeg(box.id,st.ctx);togs();redraw();}
    else if(box.id===id+'m'){st.measure=b.dataset.v;setSeg(box.id,st.measure);redraw();}
    else if(box.id===id+'j'){st.judge=b.dataset.v;setSeg(box.id,st.judge);redraw();}
  });
  tog.addEventListener('click',e=>{const b=e.target.closest('button[data-m]');if(!b)return;const m=b.dataset.m;
    st.on.has(m)?st.on.delete(m):st.on.add(m);b.setAttribute('aria-pressed',String(st.on.has(m)));redraw();});
  personaHooks.push(()=>{if(st.turn)st.turn=Math.min(st.turn,NT());togs();redraw();});
  transcript();note();
});

/* ---------- trajectory slide: where each method puts the activations, tied to the outputs ---------- */
(function(){
  if(!$('stc'))return;
  const st={ctx:'prompt',turn:8,on:new Set(['none','add','cap','cap_aa','add_cap','conceptor_tok','add_ctok']),view:'mean',layer:'steer'};
  const hasTraj=()=>JUDGES().some(j=>(PD().hold[j]||[]).some(h=>h.per_turn.some(t=>t.x_pre!=null)));
  /* the Assistant Axis cap acts downstream (modules 25-30 on SmolLM3), so it shows only at the read layer */
  const readLayer=()=>st.layer==='read'&&hasTraj();
  const J=()=>{const js=JUDGES();return js.includes('llama')?'llama':js[js.length-1];};
  /* mean over conversations, or conversation 1 alone (the paper's figure follows one conversation) */
  const val=(m,t,key)=>{if(st.view==='mean')return turnVal(J(),st.ctx,m,t,key);
    const h=holdRow(J(),st.ctx,m),e=h&&h.examples[t-1];return e&&e[key]!=null?e[key]:null;};
  const ms=()=>methodsIn(st.ctx).filter(m=>st.on.has(m));
  $('stp').innerHTML=personaSeg('stps');
  $('stcw').innerHTML=segHTML('stcs',[['prompt','Prompt + steering'],['steer_only','Steering only']],st.ctx);
  $('stvw').innerHTML=segHTML('stvs',[['mean','Mean of conversations'],['conv','Conversation 1']],st.view);
  $('stlw').innerHTML=segHTML('stls',[['steer','Steering layer'],['read','Read layer']],st.layer);
  const tog=$('stt');
  function togs(){tog.innerHTML=methodsIn(st.ctx).map(m=>'<button type="button" data-m="'+m+'" aria-pressed="'+st.on.has(m)+'">'+swatchI(m)+esc(METH[m][0])+'</button>').join('');}
  const key=()=>readLayer()?'read_steered':(hasTraj()?'x_post':'readout');
  /* a "before" line only for methods that edit the steering layer (the AA cap and no steering never do) */
  const edits=m=>!readLayer()&&hasTraj()&&XS().some(t=>{const a=val(m,t,'x_pre'),b=val(m,t,'x_post');return a!=null&&b!=null&&Math.abs(a-b)>1e-4;});
  /* two charts and a note share the right column: cap each so the slide fits a 16:9 screen without scrolling */
  const fitH={get maxH(){return Math.round(innerHeight*.27);}};
  const line=LineChart('stc',{aspect:.34,ml:64,mr:24,direct:false,canvasOpt:fitH,xs:XS,yRange:()=>{
      const vals=ms().flatMap(m=>XS().flatMap(t=>[val(m,t,key()),edits(m)?val(m,t,'x_pre'):null])).filter(v=>v!=null);
      return [Math.min(-.25,Math.floor((Math.min(...vals,0))*4)/4),Math.max(1.25,Math.ceil((Math.max(...vals,1))*4)/4)];},
    yFmt:v=>(+v).toFixed(2),xFmt:String,xTicks:()=>XS().filter(t=>NT()<=12||t===1||t%5===0||probesNow().includes(t)),bands:probeBands,selected:()=>st.turn-1,
    refs:()=>[{y:0,label:'assistant'},{y:1,label:ROLE(PERS)},...(hasTraj()&&!readLayer()?Object.entries(PD().floors||{}).filter(([m])=>st.on.has(m)).map(([m,v])=>({y:v,label:METH[m][1]+' floor',align:'right'})):[])],
    tipTitle:i=>'turn '+(i+1)+(probesNow().includes(i+1)?' · '+(PROBE_KIND[i+1]||'probe'):''),vFmt:v=>(+v).toFixed(2),
    series:()=>ms().flatMap(m=>{const c=COL[METH[m][2]],d=METH[m][3]||undefined;
      const post={name:METH[m][0]+(edits(m)?', after':''),color:c,dash:d,width:2.4,values:XS().map(t=>val(m,t,key()))};
      return edits(m)?[{name:METH[m][0]+', before',color:rgba(c,.35),dash:[2,3],width:1.4,values:XS().map(t=>val(m,t,'x_pre'))},post]:[post];}),
    onPick:i=>{st.turn=i+1;redraw();}});
  /* activation space at one turn: persona-axis position against off-axis spread, before -> after each method */
  let hits=[];
  const plane=Canvas('stm',.34,(c,w,h)=>{  // fitH (its last argument) caps the height as for the turn chart
    const M={l:56,r:16,t:14,b:36};hits=[];
    if(!hasTraj()){c.font='12px '+MONO;c.fillStyle=COL.muted;c.textAlign='center';c.textBaseline='middle';
      c.fillText(JUDGES().length?'Before/after positions arrive with the 20-turn runs.':'The 20-turn hold runs for this persona aren\'t judged yet.',w/2,h/2);return;}
    const P=ms().filter(m=>val(m,st.turn,'off_pre')!=null);
    const xs=P.flatMap(m=>[val(m,st.turn,'x_pre'),val(m,st.turn,'x_post')]).concat([0,1]);
    const ys=P.flatMap(m=>[val(m,st.turn,'off_pre'),val(m,st.turn,'off_post')]).concat([1]);
    const x0=Math.min(...xs)-.15,x1=Math.max(...xs)+.15,y0=Math.max(0,Math.min(...ys)-.1),y1=Math.max(...ys)+.1;
    const X=v=>M.l+(v-x0)/(x1-x0)*(w-M.l-M.r),Y=v=>M.t+(1-(v-y0)/(y1-y0))*(h-M.t-M.b);
    c.strokeStyle=COL.grid;c.lineWidth=1;c.font='10px '+MONO;c.fillStyle=COL.muted;
    ticks(y0,y1,4).forEach(v=>{c.beginPath();c.moveTo(M.l,Y(v)+.5);c.lineTo(w-M.r,Y(v)+.5);c.stroke();c.textAlign='right';c.textBaseline='middle';c.fillText(v.toFixed(2),M.l-6,Y(v));});
    ticks(x0,x1,5).forEach(v=>{c.textAlign='center';c.textBaseline='top';c.fillText(v.toFixed(1),X(v),h-M.b+7);});
    [[0,'assistant'],[1,ROLE(PERS)]].forEach(([v,l])=>{c.setLineDash([4,4]);c.strokeStyle=COL.muted;c.beginPath();c.moveTo(X(v)+.5,M.t);c.lineTo(X(v)+.5,h-M.b);c.stroke();c.setLineDash([]);
      text(c,l,X(v)+4,M.t+8,{size:10});});
    mark(c,X(1),Y(1),'dia',6,COL.pirate,false,true);text(c,'real '+ROLE(PERS)+' tokens',X(1)+9,Y(1)+12,{size:10,color:COL.ink});
    c.textAlign='right';c.textBaseline='bottom';c.fillStyle=COL.muted;c.fillText('position on the persona axis →',w-M.r,h-1);
    c.textAlign='left';c.textBaseline='top';c.fillText('off-axis spread ↑',M.l+4,1);
    P.forEach(m=>{const col=COL[METH[m][2]];
      const a=[val(m,st.turn,'x_pre'),val(m,st.turn,'off_pre')],b=[val(m,st.turn,'x_post'),val(m,st.turn,'off_post')];
      mark(c,X(a[0]),Y(a[1]),'o',4,col,true,true);
      if(Math.hypot(X(b[0])-X(a[0]),Y(b[1])-Y(a[1]))>3){c.strokeStyle=col;c.lineWidth=2.2;c.setLineDash(METH[m][3]||[]);c.beginPath();c.moveTo(X(a[0]),Y(a[1]));c.lineTo(X(b[0]),Y(b[1]));c.stroke();c.setLineDash([]);
        const ang=Math.atan2(Y(b[1])-Y(a[1]),X(b[0])-X(a[0]));c.fillStyle=col;c.beginPath();c.moveTo(X(b[0]),Y(b[1]));
        c.lineTo(X(b[0])-10*Math.cos(ang-.4),Y(b[1])-10*Math.sin(ang-.4));c.lineTo(X(b[0])-10*Math.cos(ang+.4),Y(b[1])-10*Math.sin(ang+.4));c.fill();}
      mark(c,X(b[0]),Y(b[1]),'o',4.5,col,false,true);
      c.font='500 10px '+MONO;const fits=X(b[0])+8+c.measureText(METH[m][1]).width<=w-M.r;  /* keep labels inside the plot */
      text(c,METH[m][1],fits?X(b[0])+8:X(b[0])-8,Y(b[1])-9,{size:10,color:COL.ink,align:fits?'left':'right'});
      hits.push({x:X(b[0]),y:Y(b[1]),m,a,b});});
  },fitH);
  hoverable($('stm'),(px,py,e)=>{let best=null,bd=20*20;hits.forEach(p=>{const d=(p.x-px)**2+(p.y-py)**2;if(d<bd){bd=d;best=p;}});
    if(!best){hideTip();return;}
    showTip('<div class="tv">'+esc(METH[best.m][0])+' · turn '+st.turn+'</div><div class="row"><span class="tl">axis position</span>&nbsp;<b>'+best.a[0].toFixed(2)+' → '+best.b[0].toFixed(2)+'</b></div>'+
      '<div class="row"><span class="tl">off-axis spread</span>&nbsp;<b>'+best.a[1].toFixed(2)+' → '+best.b[1].toFixed(2)+'</b></div>',e.clientX,e.clientY);},()=>hideTip());
  function chat(){
    const j=J(),pick=ms();
    const first=pick.map(m=>holdRow(j,st.ctx,m)).find(h=>h&&h.examples[st.turn-1]);
    $('stx').innerHTML=(first?'<div class="msg user"><span class="who">User · turn '+st.turn+(probesNow().includes(st.turn)?' · '+(PROBE_KIND[st.turn]||'probe'):'')+'</span>'+esc(first.examples[st.turn-1].q)+'</div>':'')+
      pick.map(m=>{const h=holdRow(j,st.ctx,m);const e=h&&h.examples[st.turn-1];if(!e)return '';
        const pos=e.x_pre!=null?' · axis '+e.x_pre.toFixed(2)+' → '+e.x_post.toFixed(2):'';
        return '<div class="msg bot" style="border-left-color:'+COL[METH[m][2]]+'"><span class="who">'+esc(METH[m][0])+pos+' · '+JSHORT[j]+': '+esc(e.label||'–')+
          (e.identity?' · '+(e.identity==='ai'?'admits AI':'no AI claim'):'')+'</span>'+esc(e.a.slice(0,260))+(e.a.length>260?' …':'')+'</div>';}).join('');
    const F=PD().fits||{},band=(F.cap_aa_modules||[]).length?' (modules '+F.cap_aa_modules[0]+'–'+F.cap_aa_modules[F.cap_aa_modules.length-1]+')':'';
    const where=readLayer()?'at a later layer (hidden '+(F.read_hidden??'?')+'), where the Assistant Axis cap'+band+' acts. The bottom chart stays at the steering layer.'
      :'at the steering layer'+(F.module!=null?' (module '+F.module+')':'')+'. Dashed: before the edit. Solid: after. The Assistant Axis cap acts later; switch to the read layer to see it.';
    $('stn').textContent=(st.view==='conv'?'Conversation 1 only. ':'')+(hasTraj()?MODEL(PERS)+'. Top: each method\'s place on the '+ROLE(PERS)+' axis per turn (0 = assistant, 1 = '+ROLE(PERS)+'), '+where
      :'Top: where each answer sits on the persona axis without steering (0 = assistant, 1 = persona).')+
      ' Click a turn to see it below and read conversation 1 (labels: '+JNAME[j]+').';
  }
  function redraw(){line.f.draw();plane.draw();chat();}
  $('stcw').addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(!b)return;st.ctx=b.dataset.v;setSeg('stcs',st.ctx);togs();redraw();});
  $('stvw').addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(!b)return;st.view=b.dataset.v;setSeg('stvs',st.view);redraw();});
  $('stlw').addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(!b)return;st.layer=b.dataset.v;setSeg('stls',st.layer);redraw();});
  tog.addEventListener('click',e=>{const b=e.target.closest('button[data-m]');if(!b)return;const m=b.dataset.m;
    st.on.has(m)?st.on.delete(m):st.on.add(m);b.setAttribute('aria-pressed',String(st.on.has(m)));redraw();});
  personaHooks.push(()=>{st.turn=Math.min(st.turn,NT());togs();redraw();});
  togs();chat();
})();

/* ---------- mechanism: what the per-token conceptor keeps, direction by direction ---------- */
(function(){
  if(!$('skc'))return;
  const M=DK.mechanism||{};
  const key=()=>MODEL(PERS)==='SmolLM3-3B'&&M[ROLE(PERS)]?ROLE(PERS):null;
  $('skp').innerHTML=personaSeg('skps');
  let hit=[];
  const f=Canvas('skc',.6,(c,w,h)=>{
    hit=[];
    const k=key();
    if(!k){c.font='12px '+MONO;c.fillStyle=COL.muted;c.textAlign='center';c.textBaseline='middle';
      c.fillText('No spectrum for this persona yet.',w/2,h/2);return;}
    const S=M[k],B=S.bins,d=S.d;
    const L={l:58,r:18},top=[16,Math.round(h*.54)],bot=[Math.round(h*.65),h-50];
    const X=r=>L.l+Math.log(r)/Math.log(d+1)*(w-L.l-L.r);
    const Yt=v=>top[0]+(1-v)*(top[1]-top[0]);
    const vmax=Math.max(...B.map(b=>Math.max(b.v,b.gap)))||1;
    const Yb=v=>bot[1]-v/vmax*(bot[1]-bot[0]);
    c.font='10px '+MONO;c.strokeStyle=COL.grid;c.lineWidth=1;c.fillStyle=COL.muted;
    [0,.25,.5,.75,1].forEach(v=>{c.beginPath();c.moveTo(L.l,Yt(v)+.5);c.lineTo(w-L.r,Yt(v)+.5);c.stroke();c.textAlign='right';c.textBaseline='middle';c.fillText(pct(v),L.l-6,Yt(v));});
    c.textAlign='left';c.textBaseline='top';c.fillText('kept by the conceptor',L.l+4,2);
    c.setLineDash([6,4]);c.strokeStyle=COL.assist;c.lineWidth=1.6;c.beginPath();c.moveTo(L.l,Yt(1));c.lineTo(w-L.r,Yt(1));c.stroke();c.setLineDash([]);
    text(c,'cap: every direction kept',w-L.r-4,Yt(1)+12,{size:10,align:'right',color:COL.ink});
    c.strokeStyle=COL.pirate;c.lineWidth=2.4;c.beginPath();
    B.forEach((b,i)=>{const x0=X(b.lo),x1=X(b.hi+1);i?c.lineTo(x0,Yt(b.keep)):c.moveTo(x0,Yt(b.keep));c.lineTo(x1,Yt(b.keep));});c.stroke();
    c.strokeStyle=COL.grid;c.beginPath();c.moveTo(L.l,bot[1]+.5);c.lineTo(w-L.r,bot[1]+.5);c.stroke();
    B.forEach(b=>{const x0=X(b.lo)+1,x1=X(b.hi+1)-1,wd=Math.max(1,(x1-x0)/2-.5);
      c.fillStyle=rgba(COL.muted,.55);c.fillRect(x0,Yb(b.gap),wd,bot[1]-Yb(b.gap));
      c.fillStyle=COL.orange;c.fillRect(x0+wd,Yb(b.v),wd,bot[1]-Yb(b.v));
      hit.push({x0:X(b.lo),x1:X(b.hi+1),b});});
    [1,10,100,1000].filter(r=>r<=d).forEach(r=>{c.fillStyle=COL.muted;c.textAlign='center';c.textBaseline='top';c.fillText(String(r),X(r),h-46);});
    c.textAlign='right';c.textBaseline='bottom';c.fillText('eigen-direction rank of the persona token cloud (log) →',w-L.r,h-1);
    text(c,'where it lies: grey = assistant gap, orange = persona direction',L.l+4,bot[0]-8,{size:10,color:COL.ink});
  });
  hoverable($('skc'),(px,py,e)=>{const t=hit.find(q=>px>=q.x0&&px<q.x1);if(!t){hideTip();return;}const b=t.b;
    showTip('<div class="tv">directions '+b.lo+(b.hi>b.lo?'-'+b.hi:'')+'</div><div class="row"><span class="tl">kept by the conceptor</span>&nbsp;<b>'+pct(b.keep)+'</b></div>'+
      '<div class="row"><span class="tl">share of the persona direction</span>&nbsp;<b>'+(100*b.v).toFixed(1)+'%</b></div>'+
      '<div class="row"><span class="tl">share of the assistant gap</span>&nbsp;<b>'+(100*b.gap).toFixed(1)+'%</b></div>',e.clientX,e.clientY);},()=>hideTip());
  function info(){
    const k=key();
    if(!k){$('skr').innerHTML='';$('skn').textContent='This chart has data only for SmolLM3\'s pirate, ghost, bard and consultant.';return;}
    const S=M[k];
    rows($('skr'),[['kept along the persona axis',pct(S.keep_along_vhat)],['share of the assistant gap kept',pct(S.gap_energy_kept)],
      ['share of the gap in directions cut below 50%',pct(S.gap_share_in_directions_kept_below_half)],
      ['directions kept > 90% / < 10%',S.n_kept_above_090+' / '+S.n_kept_below_010+' of '+S.d]]);
    $('skn').textContent='SmolLM3-3B, '+k+': per-token cloud of '+S.n_tokens.toLocaleString()+' tokens at layer 17, aperture '+S.aperture.toPrecision(3)+'.';
  }
  personaHooks.push(()=>{f.draw();info();});
  info();
})();

/* ---------- judges slide ---------- */
(function(){
  if(!$('s14t'))return;
  const hb=(el,list)=>{el.innerHTML=list.map(r=>'<div class="hbar"><span>'+esc(r[0])+'</span><span class="track"><i style="width:'+(100*Math.max(0,Math.min(1,r[1]||0))).toFixed(1)+'%"></i></span><b>'+pct(r[1])+'</b></div>').join('');};
  const cj=JALL.filter(j=>DK.calib[j]);
  hb($('s14rp'),cj.map(j=>[JSHORT[j],DK.calib[j].roleplay_fully_on_ai]));
  hb($('s14id'),cj.map(j=>[JSHORT[j],DK.calib[j].holdout]));
  const st={m:'t5',ctx:'prompt'};
  function opts(){
    const pr=probesNow(),o=[];
    pr.forEach(t=>o.push(PROBE_KIND[t]==='are you an AI?'?['t'+t,'"Are you an AI?" (turn '+t+'): admits AI']:['t'+t,'"Drop the act" (turn '+t+'): in character']));
    o.push(['ord','Ordinary turns: in character'],['coh','Ordinary turns: coherence']);
    if(!o.some(x=>x[0]===st.m))st.m=o[0][0];
    $('s14m').innerHTML=o.map(x=>'<button type="button" data-v="'+x[0]+'" aria-pressed="'+(x[0]===st.m)+'">'+x[1]+'</button>').join('');
  }
  $('s14c').innerHTML=[['prompt','Prompt + steering'],['steer_only','Steering only']].map(x=>'<button type="button" data-v="'+x[0]+'" aria-pressed="'+(x[0]===st.ctx)+'">'+x[1]+'</button>').join('');
  const val=(j,m)=>{const h=holdRow(j,st.ctx,m);if(!h)return null;
    if(st.m[0]==='t'){const t=+st.m.slice(1);return turnVal(j,st.ctx,m,t,PROBE_KIND[t]==='are you an AI?'?'identity_ai_rate':'fully_rate');}
    if(st.m==='ord')return h.ordinary.fully_rate;return h.ordinary.coherence;};
  function draw(){
    const coh=st.m==='coh',js=JUDGES();
    const cell=v=>{if(v==null)return '<td class="cell">–</td>';const t=coh?v/100:v;
      return '<td class="cell" style="background:'+rgba('#003057',(.06+.62*t).toFixed(2))+';color:'+(t>.55?'#fbfaf7':'#1a1a18')+'">'+(coh?num(v):pct(v))+'</td>';};
    $('s14t').innerHTML='<thead><tr><th>Method</th>'+js.map(j=>'<th>'+JSHORT[j]+'</th>').join('')+'<th>Spread</th></tr></thead><tbody>'+
      methodsIn(st.ctx).filter(m=>!/^ctok_k/.test(m)&&m!=='region').map(m=>{const vs=js.map(j=>val(j,m));const ok=vs.filter(v=>v!=null);  /* the aperture sweep has its own slide; region appears only in the follow-up contexts */
        const spread=ok.length?Math.max(...ok)-Math.min(...ok):null;
        return '<tr><td>'+lineSwatch(m)+esc(METH[m][0])+'</td>'+vs.map(cell).join('')+
          '<td>'+(spread==null?'–':(coh?num(spread):pct(spread)))+'</td></tr>';}).join('')+'</tbody>';
  }
  onSeg('s14m',v=>{st.m=v;draw();});
  onSeg('s14c',v=>{st.ctx=v;draw();});
  const pb=$('s14p');if(pb)pb.innerHTML=personaSeg('s14ps');
  personaHooks.push(()=>{opts();draw();});
  opts();draw();
})();

/* ---------- takeaway numbers, filled from the data (persona 'all' = mean over the main model's personas, judges always
   averaged; a second model's personas, which carry a "model" label, never mix into the headline means) ---------- */
const MAIN=PERSONAS.filter(q=>!DK.personas[q].model);
const statFor=(p,ctx,m,where,key)=>mean(JUDGES_OF(p).map(j=>{const h=holdRowP(p,j,ctx,m);if(!h)return null;
  if(where==='ordinary')return h.ordinary[key];return turnValP(p,j,ctx,m,+where.slice(1),key);}));
/* who: 'all' (the main model's personas), 'm:<model>' (one model's personas, e.g. m:OLMo-3-7B) or a persona */
const GROUP=w=>w==='all'?MAIN:w.startsWith('m:')?PERSONAS.filter(q=>MODEL(q)===w.slice(2)):null;
function fillStats(){
  document.querySelectorAll('[data-stat]').forEach(el=>{
    const p=el.dataset.stat.split('|');let v=null;const asNum=p[p.length-1]==='num';
    if(p[0]==='njudges'){const n=JUDGES().length;el.textContent=['zero','one','two','three','four','five'][n]||String(n);return;}
    if(p[0]==='npersonas'){el.textContent=String(MAIN.length);return;}
    if(p[0]==='othermodels'){  /* " · OLMo-3-7B · 3 personas" for each extra model */
      const by={};PERSONAS.filter(q=>DK.personas[q].model).forEach(q=>{const m=DK.personas[q].model;by[m]=(by[m]||0)+1;});
      el.textContent=Object.entries(by).map(([m,n])=>' · '+m+' · '+n+' persona'+(n>1?'s':'')).join('');return;}
    if(p[0]==='sweepsel'){  /* sweepsel|method|key: at each persona's selected strength, mean over personas */
      v=mean(MAIN.map(q=>{const d=DK.personas[q],s=d.selected&&d.selected[p[1]]&&d.selected[p[1]].strength;
        if(s==null||!d.sweep)return null;const r=d.sweep.find(r=>r.method===p[1]&&Math.abs(r.strength-s)<1e-9);return r?r[p[2]]:null;}));
      el.textContent=asNum||p[2]==='coherence'?num(v):pct(v);return;}
    if(p[0]==='spread'){  /* spread|ctx|method|tN|key: min-max over judges and personas */
      const vs=[];MAIN.forEach(q=>JUDGES_OF(q).forEach(j=>{const x=turnValP(q,j,p[1],p[2],+p[3].slice(1),p[4]);if(x!=null)vs.push(x);}));
      el.textContent=vs.length?pct(Math.min(...vs))+'–'+pct(Math.max(...vs)):'–';return;}
    if(p[0]==='range'){  /* range|who|ctx|method|where|key: lowest-highest persona in the group, judges averaged */
      const vs=(GROUP(p[1])||[p[1]]).map(q=>statFor(q,p[2],p[3],p[4],p[5])).filter(x=>x!=null);
      const f=p[5]==='coherence'?num:pct;el.textContent=vs.length?f(Math.min(...vs))+'–'+f(Math.max(...vs)):'–';return;}
    const [who,ctx,m,where,key]=p;const g=GROUP(who);
    v=g?mean(g.map(q=>statFor(q,ctx,m,where,key))):statFor(who,ctx,m,where,key);
    el.textContent=asNum||key==='coherence'?num(v):pct(v);
  });
}
fillStats();

/* ---------- aperture slide: hold and coherence against the conceptor's aperture multiple k ---------- */
(function(){
  if(!$('sapc'))return;
  const AP=[[1,'conceptor_tok'],[3,'ctok_k3'],[10,'ctok_tight'],[30,'ctok_k30'],[100,'ctok_k100']];
  const XSa=()=>AP.map(a=>Math.log10(a[0]));
  const hasAp=p=>JUDGES_OF(p).some(j=>(DK.personas[p].hold[j]||[]).some(h=>h.context==='prompt'&&h.method==='ctok_k3'));
  let view='persona';
  const who=()=>(view==='persona'?[PERS]:view==='smol'?GROUP('all'):GROUP('m:OLMo-3-7B')).filter(hasAp);
  const val=(m,where,key)=>{const ps=who();if(!ps.length)return null;
    return mean(ps.map(p=>where==='push'?mean([statFor(p,'prompt',m,'t5',key),statFor(p,'prompt',m,'t13',key)]):statFor(p,'prompt',m,where,key)));};
  const kFmt=x=>'×'+Math.round(Math.pow(10,x));
  const fitH={get maxH(){return Math.round(innerHeight*.28);}};
  const hold=LineChart('sapc',{aspect:.42,ml:64,mr:110,canvasOpt:fitH,xs:XSa,yRange:()=>[0,1],yTicks:()=>[0,.25,.5,.75,1],yFmt:pct,xFmt:kFmt,
    tipTitle:i=>'aperture k = '+AP[i][0],vFmt:pct,xTitle:'aperture multiple k →',
    series:()=>[{name:'In character, ordinary turns',short:'ordinary turns',color:COL.pirate,width:2.6,dots:true,values:AP.map(a=>val(a[1],'ordinary','fully_rate'))},
      {name:'In character at "drop the act"',short:'at the pushes',color:COL.pirate,width:2,dash:[6,4],dots:true,values:AP.map(a=>val(a[1],'push','fully_rate'))}]});
  const coh=LineChart('sapk',{aspect:.3,ml:64,mr:110,canvasOpt:fitH,xs:XSa,yRange:()=>[0,100],yTicks:()=>[0,50,100],yFmt:num,xFmt:kFmt,
    tipTitle:i=>'aperture k = '+AP[i][0],vFmt:num,
    series:()=>[{name:'Coherence',short:'coherence',color:COL.ink,width:2.4,dots:true,values:AP.map(a=>val(a[1],'ordinary','coherence'))}]});
  function note(){
    const ps=who();
    $('sapn').textContent=ps.length?(view==='persona'?PNAME(PERS):ps.length+' personas: '+ps.map(PNAME).join(', '))+'.':
      PNAME(PERS)+' has no aperture runs yet.';
  }
  function redraw(){hold.f.draw();coh.f.draw();note();}
  const pb=$('sap');if(pb)pb.innerHTML=personaSeg('saps');
  onSeg('sapv',v=>{view=v;redraw();});
  personaHooks.push(redraw);
  note();
})();
