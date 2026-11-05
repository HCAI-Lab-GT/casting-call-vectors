/* ================= Deck results slides: the problem, setting, why, holding over turns, judges, takeaways ================= */
const DK=JSON.parse($('deck-data').textContent);
const JUDGES=['qwen','gemma','mistral','llama'].filter(j=>DK.hold[j]);
const JNAME={qwen:'Qwen3-30B-A3B',gemma:'Gemma-3-27B',mistral:'Mistral-Small-3.2',llama:'Llama-3.3-70B',mean:'Mean of judges'};
const JSHORT={qwen:'Qwen',gemma:'Gemma',mistral:'Mistral',llama:'Llama',mean:'Mean'};
const METH={  // name, short (line label), color key, dash
  none:['No steering','none','none',[5,4]], add:['Steering vector','vector','orange',null],
  add_ctok:['Vector + per-token conceptor','vec+con','orange',[2,4]], cap:['Activation cap','cap','assist',null],
  conceptor_tok:['Per-token conceptor','conceptor','pirate',null], conceptor:['Conceptor, answer-average fit','con (avg)','pirate',[8,4]],
  lowrank:['Low-rank conceptor','low-rank','pirate',[2,3]], meanshift_tok:['Mean-shift control, per token','mean-shift','other',[6,3]],
  meanshift:['Mean-shift control, answer-average','ms (avg)','other',[2,3]]};
const MORDER=['none','add','add_ctok','cap','conceptor_tok','conceptor','lowrank','meanshift_tok','meanshift'];
const DEFAULT_ON=new Set(['none','add','add_ctok','cap','conceptor_tok']);
const MEASURES=[['fully_rate','In character'],['coherence','Coherence'],['identity_ai_rate','Admits being an AI']];
const PROBES=[5,8];
const pct=v=>v==null||!isFinite(v)?'–':Math.round(v*100)+'%';
const num=v=>v==null||!isFinite(v)?'–':String(Math.round(v));
const fmtM=(m,v)=>m==='coherence'?num(v):pct(v);
const mean=a=>{const b=a.filter(v=>v!=null&&isFinite(v));return b.length?b.reduce((s,x)=>s+x,0)/b.length:null;};
const holdRow=(j,ctx,m)=>(DK.hold[j]||[]).find(h=>h.context===ctx&&h.method===m);
const turnVal=(j,ctx,m,t,key)=>{const h=holdRow(j,ctx,m);if(!h)return null;const p=h.per_turn.find(x=>x.turn===t-1);return p?p[key]:null;};
const series=(judge,ctx,m,key)=>{
  const turns=Array.from({length:10},(_,i)=>i+1);
  return turns.map(t=>judge==='mean'?mean(JUDGES.map(j=>turnVal(j,ctx,m,t,key))):turnVal(judge,ctx,m,t,key));
};
const methodsIn=ctx=>MORDER.filter(m=>JUDGES.some(j=>holdRow(j,ctx,m)));
const nOf=(ctx,m)=>{const h=holdRow(JUDGES[0],ctx,m);return h&&h.per_turn[0]?h.per_turn[0].n:null;};
const swatchI=m=>{const d=METH[m][3];return '<i class="'+(d?(d[0]<=2?'dot':'dash'):'')+'" style="color:'+COL[METH[m][2]]+'"></i>';};
const lineSwatch=m=>{const d=METH[m][3];return '<i style="display:inline-block;width:18px;height:0;margin-right:8px;vertical-align:middle;border-top:3px '+
  (d?(d[0]<=2?'dotted':'dashed'):'solid')+' '+COL[METH[m][2]]+'"></i>';};
const probeBands=()=>PROBES.map(t=>({x0:t-.45,x1:t+.45,color:rgba(COL.probe,.09),alpha:1}));
const XS10=Array.from({length:10},(_,i)=>i+1);

/* ---------- slide 2: a prompted pirate, turn by turn ---------- */
(function(){
  const box=$('s2turns');if(!box)return;
  let turn=1;
  box.innerHTML=XS10.map(t=>'<button type="button" data-v="'+t+'" aria-pressed="'+(t===1)+'"'+(PROBES.includes(t)?' class="probe" title="probe turn"':'')+'>'+t+'</button>').join('');
  const ex=j=>{const h=holdRow(j,'prompt','none');return h?h.examples[turn-1]:null;};
  function chat(){
    const base=ex(JUDGES[JUDGES.length-1])||ex(JUDGES[0]);
    if(!base){$('s2chat').textContent='No transcript in the data.';return;}
    const pills=JUDGES.map(j=>{const e=ex(j);if(!e)return '';
      const good=e.label==='fully',ai=e.identity==='ai';
      return '<span class="pill '+(good?'good':'bad')+'">'+JSHORT[j]+': '+esc(e.label||'–')+'</span>'+(e.identity?'<span class="pill'+(ai?' bad':'')+'">'+JSHORT[j]+': '+(ai?'admits AI':'no AI claim')+'</span>':'');}).join('');
    $('s2chat').innerHTML='<div class="msg user"><span class="who">User · turn '+turn+(PROBES.includes(turn)?' · probe':'')+'</span>'+esc(base.q)+'</div>'+
      '<div class="msg bot"><span class="who">SmolLM3-3B, pirate prompt, no steering</span>'+esc(base.a)+(base.a.length>=500?' …':'')+'</div>'+
      '<div class="pills">'+pills+'</div>';
  }
  const ch=LineChart('s2line',{aspect:.56,ml:64,mr:24,xs:()=>XS10,yRange:()=>[0,1],yTicks:()=>[0,.5,1],yFmt:pct,xFmt:String,
    bands:probeBands,selected:()=>turn-1,direct:false,vFmt:v=>pct(v),tipTitle:i=>'turn '+(i+1),
    series:()=>[...JUDGES.map(j=>({name:JNAME[j],color:rgba(COL.muted,.45),width:1.2,values:series(j,'prompt','none','fully_rate')})),
      {name:'Mean of judges',color:COL.ink,width:2.6,dots:true,values:series('mean','prompt','none','fully_rate')}],
    onPick:i=>{turn=i+1;setSeg('s2turns',String(turn));chat();ch.f.draw();}});
  onSeg('s2turns',v=>{turn=+v;chat();ch.f.draw();});
  chat();
})();

/* ---------- slide 9: setting a persona, in character against coherence ---------- */
(function(){
  const cv=$('s9c');if(!cv)return;
  const rows=DK.sweep;const ms=MORDER.filter(m=>rows.some(r=>r.method===m));
  const on=new Set(ms.filter(m=>['none','add','add_ctok','cap','conceptor_tok','conceptor'].includes(m)));
  const tog=$('s9tog');
  tog.innerHTML=ms.map(m=>'<button type="button" data-m="'+m+'" aria-pressed="'+on.has(m)+'">'+swatchI(m)+esc(METH[m][0])+'</button>').join('');
  let pts=[],hover=null,zoom=false;
  const f=Canvas('s9c',.68,(c,w,h)=>{
    const x0=zoom?60:0,M={l:64,r:18,t:16,b:40},X=v=>M.l+(Math.max(x0,v)-x0)/(100-x0)*(w-M.l-M.r),Y=v=>M.t+(1-v)*(h-M.t-M.b);
    c.strokeStyle=COL.grid;c.lineWidth=1;c.font='10px '+MONO;c.fillStyle=COL.muted;
    [0,.25,.5,.75,1].forEach(v=>{c.beginPath();c.moveTo(M.l,Y(v)+.5);c.lineTo(w-M.r,Y(v)+.5);c.stroke();c.textAlign='right';c.textBaseline='middle';c.fillText(pct(v),M.l-8,Y(v));});
    (zoom?[60,70,80,90,100]:[0,25,50,75,100]).forEach(v=>{c.textAlign='center';c.textBaseline='top';c.fillText(String(v),X(v),h-M.b+8);});
    c.textAlign='right';c.textBaseline='bottom';c.fillText('coherence (judge, 0-100) →',w-M.r,h-2);
    c.textAlign='left';c.textBaseline='top';c.fillText('in character ↑',M.l+4,2);
    pts=[];
    ms.filter(m=>on.has(m)).forEach(m=>{
      const rs=rows.filter(r=>r.method===m&&r.coherence!=null&&r.fully_rate!=null).sort((a,b)=>a.strength-b.strength);
      const col=COL[METH[m][2]];
      c.strokeStyle=col;c.lineWidth=2;c.setLineDash(METH[m][3]||[]);c.beginPath();
      rs.forEach((r,i)=>{const x=X(r.coherence),y=Y(r.fully_rate);i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();c.setLineDash([]);
      const sel=DK.selected[m]&&DK.selected[m].strength;
      rs.forEach(r=>{const x=X(r.coherence),y=Y(r.fully_rate);pts.push({x,y,r,m});
        mark(c,x,y,'o',3.6,col,false,true);
        if(sel!=null&&Math.abs(r.strength-sel)<1e-9){c.beginPath();c.arc(x,y,8,0,2*Math.PI);c.lineWidth=1.8;c.strokeStyle=col;c.stroke();}});
      const last=rs[rs.length-1];if(last)text(c,METH[m][1],X(last.coherence)+8,Y(last.fully_rate)-9,{size:10,color:COL.ink});
    });
    if(hover){c.beginPath();c.arc(hover.x,hover.y,6,0,2*Math.PI);c.lineWidth=2;c.strokeStyle=COL.ink;c.stroke();}
  });
  hoverable(cv,(px,py,e)=>{
    let best=null,bd=22*22;
    pts.forEach(p=>{const d=(p.x-px)**2+(p.y-py)**2;if(d<bd){bd=d;best=p;}});
    if(best!==hover){hover=best;f.draw();}
    if(!best){hideTip();return;}
    const r=best.r,P=best.m==='cap'?'percentile ':(['add','add_ctok'].includes(best.m)?'c ':'β ');
    showTip('<div class="tv">'+esc(METH[best.m][0])+(best.m==='none'?'':' · '+P+r.strength)+'</div>'+
      '<div class="row"><span class="tl">fully in character</span>&nbsp;<b>'+pct(r.fully_rate)+'</b></div>'+
      '<div class="row"><span class="tl">coherence</span>&nbsp;<b>'+num(r.coherence)+'</b></div>'+
      (r.voice_as_rate!=null?'<div class="row"><span class="tl">speaks as the pirate</span>&nbsp;<b>'+pct(r.voice_as_rate)+'</b></div>':''),e.clientX,e.clientY);
  },()=>{hover=null;f.draw();hideTip();});
  tog.addEventListener('click',e=>{const b=e.target.closest('button[data-m]');if(!b)return;const m=b.dataset.m;
    on.has(m)?on.delete(m):on.add(m);b.setAttribute('aria-pressed',String(on.has(m)));f.draw();});
  onSeg('s9z',v=>{zoom=v==='zoom';f.draw();});
  const add1=rows.find(r=>r.method==='add'&&r.strength===1),con=rows.filter(r=>r.method==='conceptor_tok').sort((a,b)=>b.strength-a.strength)[0];
  if(add1&&con)$('s9note').textContent='Steering vector at c 1.0: '+pct(add1.fully_rate)+' in character at coherence '+num(add1.coherence)+
    '. Per-token conceptor at its strongest (β '+con.strength+'): '+pct(con.fully_rate)+', about where no steering sits.';
})();

/* ---------- slide 10: why a centered conceptor can't close the gap (toy) ---------- */
(function(){
  if(!$('s10c'))return;
  const s={a:+$('s10a').value,mode:'along',h:null};
  const MA=[-.75,-.5],MP=[.55,.45],V=sub(MP,MA),VH=unit(V),NP=[-VH[1],VH[0]];
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

/* ---------- slides 11-13: holding over ten turns ---------- */
document.querySelectorAll('.sbody.hold').forEach(root=>{
  const id=root.dataset.hold,st={measure:root.dataset.measure,ctx:root.dataset.context,judge:'mean',
    on:new Set(DEFAULT_ON),turn:+(root.dataset.transcript||0)};
  const ctxs=[['prompt','Prompt + steering'],['steer_only','Steering only']];
  const judges=[['mean','Mean'],...JUDGES.map(j=>[j,JSHORT[j]])];
  root.innerHTML='<div class="scol">'+
    '<div class="ctlrow"><span>Context</span><div class="seg" id="'+id+'c" role="group" aria-label="Context">'+ctxs.map(x=>'<button type="button" data-v="'+x[0]+'" aria-pressed="'+(x[0]===st.ctx)+'">'+x[1]+'</button>').join('')+'</div></div>'+
    '<div class="ctlrow"><span>Measure</span><div class="seg" id="'+id+'m" role="group" aria-label="Measure">'+MEASURES.map(x=>'<button type="button" data-v="'+x[0]+'" aria-pressed="'+(x[0]===st.measure)+'">'+x[1]+'</button>').join('')+'</div></div>'+
    '<div class="ctlrow"><span>Judge</span><div class="seg" id="'+id+'j" role="group" aria-label="Judge">'+judges.map(x=>'<button type="button" data-v="'+x[0]+'" aria-pressed="'+(x[0]===st.judge)+'">'+x[1]+'</button>').join('')+'</div></div>'+
    '<div class="ctlrow"><span>Methods</span><div class="mtog" id="'+id+'t"></div></div>'+
    '<p class="note" id="'+id+'n"></p></div>'+
    '<div><canvas id="'+id+'cv" role="img" aria-label="Per-turn hold curves for each method"></canvas>'+
    '<p class="note">Shaded: turn 5 "drop the act", turn 8 "are you an AI?". Click a turn'+(st.turn?' to read conversation 1 at that turn.':' to mark it.')+' Mean: average of '+JUDGES.length+' judges.</p>'+
    (st.turn?'<div class="chat" id="'+id+'x" style="margin-top:8px;max-height:24vh;overflow:auto"></div>':'')+'</div>';
  const tog=$(id+'t');
  function togs(){
    const ms=methodsIn(st.ctx);
    tog.innerHTML=ms.map(m=>'<button type="button" data-m="'+m+'" aria-pressed="'+st.on.has(m)+'" title="'+(nOf(st.ctx,m)||'')+' conversations">'+swatchI(m)+esc(METH[m][0])+'</button>').join('');
  }
  const yr=()=>st.measure==='coherence'?[0,100]:[0,1];
  const ch=LineChart(id+'cv',{aspect:st.turn?.42:.58,ml:64,mr:24,direct:false,xs:()=>XS10,yRange:yr,yTicks:()=>st.measure==='coherence'?[0,25,50,75,100]:[0,.25,.5,.75,1],
    yFmt:v=>fmtM(st.measure,v),xFmt:t=>'turn '+t,xTicks:()=>[1,2,3,4,5,6,7,8,9,10],bands:probeBands,selected:()=>st.turn?st.turn-1:-1,
    tipTitle:i=>'turn '+(i+1)+(PROBES.includes(i+1)?' · probe':''),vFmt:v=>fmtM(st.measure,v),
    series:()=>methodsIn(st.ctx).filter(m=>st.on.has(m)).map(m=>({name:METH[m][0],short:METH[m][1],color:COL[METH[m][2]],
      dash:METH[m][3]||undefined,width:m==='conceptor_tok'||m==='add'?2.6:2,values:series(st.judge,st.ctx,m,st.measure)})),
    onPick:i=>{st.turn=i+1;redraw();}});
  function transcript(){
    const x=$(id+'x');if(!x)return;
    const j=st.judge==='mean'?JUDGES[JUDGES.length-1]:st.judge;
    const ms=methodsIn(st.ctx).filter(m=>st.on.has(m));
    const first=ms.map(m=>holdRow(j,st.ctx,m)).find(h=>h&&h.examples[st.turn-1]);
    x.innerHTML=(first?'<div class="msg user"><span class="who">User · turn '+st.turn+'</span>'+esc(first.examples[st.turn-1].q)+'</div>':'')+
      ms.map(m=>{const h=holdRow(j,st.ctx,m);const e=h&&h.examples[st.turn-1];if(!e)return '';
        return '<div class="msg bot" style="border-left-color:'+COL[METH[m][2]]+'"><span class="who">'+esc(METH[m][0])+' · '+JSHORT[j]+': '+esc(e.label||'–')+
          (e.identity?' · '+(e.identity==='ai'?'admits AI':'no AI claim'):'')+'</span>'+esc(e.a.slice(0,320))+(e.a.length>320?' …':'')+'</div>';}).join('');
  }
  function note(){
    const parts=methodsIn(st.ctx).filter(m=>st.on.has(m)).map(m=>{
      const v=st.judge==='mean'?mean(JUDGES.map(j=>{const h=holdRow(j,st.ctx,m);return h&&h.ordinary[st.measure];})):(holdRow(st.judge,st.ctx,m)||{ordinary:{}}).ordinary[st.measure];
      return METH[m][1]+' '+fmtM(st.measure,v);});
    $(id+'n').textContent='Ordinary turns (no probe), '+(st.judge==='mean'?'mean of judges':JNAME[st.judge])+': '+parts.join(' · ')+'.';
  }
  function redraw(){ch.f.draw();transcript();note();}
  togs();
  onSeg(id+'c',v=>{st.ctx=v;togs();redraw();});
  onSeg(id+'m',v=>{st.measure=v;redraw();});
  onSeg(id+'j',v=>{st.judge=v;redraw();});
  tog.addEventListener('click',e=>{const b=e.target.closest('button[data-m]');if(!b)return;const m=b.dataset.m;
    st.on.has(m)?st.on.delete(m):st.on.add(m);b.setAttribute('aria-pressed',String(st.on.has(m)));redraw();});
  transcript();note();
});

/* ---------- slide 14: judges ---------- */
(function(){
  if(!$('s14t'))return;
  const hb=(el,list)=>{el.innerHTML=list.map(r=>'<div class="hbar"><span>'+esc(r[0])+'</span><span class="track"><i style="width:'+(100*Math.max(0,Math.min(1,r[1]||0))).toFixed(1)+'%"></i></span><b>'+pct(r[1])+'</b></div>').join('');};
  hb($('s14rp'),JUDGES.filter(j=>DK.calib[j]).map(j=>[JSHORT[j],DK.calib[j].roleplay_fully_on_ai]));
  hb($('s14id'),JUDGES.filter(j=>DK.calib[j]).map(j=>[JSHORT[j],DK.calib[j].holdout]));
  const st={m:'t5',ctx:'prompt'};
  const MS=[['t5','"Drop the act" (turn 5): in character'],['t8','"Are you an AI?" (turn 8): admits AI'],['ord','Ordinary turns: in character'],['coh','Ordinary turns: coherence']];
  $('s14m').innerHTML=MS.map(x=>'<button type="button" data-v="'+x[0]+'" aria-pressed="'+(x[0]===st.m)+'">'+x[1]+'</button>').join('');
  $('s14c').innerHTML=[['prompt','Prompt + steering'],['steer_only','Steering only']].map(x=>'<button type="button" data-v="'+x[0]+'" aria-pressed="'+(x[0]===st.ctx)+'">'+x[1]+'</button>').join('');
  const val=(j,m)=>{const h=holdRow(j,st.ctx,m);if(!h)return null;
    if(st.m==='t5')return turnVal(j,st.ctx,m,5,'fully_rate');if(st.m==='t8')return turnVal(j,st.ctx,m,8,'identity_ai_rate');
    if(st.m==='ord')return h.ordinary.fully_rate;return h.ordinary.coherence;};
  function draw(){
    const coh=st.m==='coh';
    const cell=v=>{if(v==null)return '<td class="cell">–</td>';const t=coh?v/100:v;
      return '<td class="cell" style="background:'+rgba('#003057',(.06+.62*t).toFixed(2))+';color:'+(t>.55?'#fbfaf7':'#1a1a18')+'">'+(coh?num(v):pct(v))+'</td>';};
    $('s14t').innerHTML='<thead><tr><th>Method</th>'+JUDGES.map(j=>'<th>'+JSHORT[j]+'</th>').join('')+'<th>Spread</th></tr></thead><tbody>'+
      methodsIn(st.ctx).map(m=>{const vs=JUDGES.map(j=>val(j,m));const ok=vs.filter(v=>v!=null);
        const spread=ok.length?Math.max(...ok)-Math.min(...ok):null;
        return '<tr><td>'+lineSwatch(m)+esc(METH[m][0])+'</td>'+vs.map(cell).join('')+
          '<td>'+(spread==null?'–':(coh?num(spread):pct(spread)))+'</td></tr>';}).join('')+'</tbody>';
  }
  onSeg('s14m',v=>{st.m=v;draw();});
  onSeg('s14c',v=>{st.ctx=v;draw();});
  draw();
})();

/* ---------- takeaway numbers, filled from the data ---------- */
document.querySelectorAll('[data-stat]').forEach(el=>{
  const p=el.dataset.stat.split('|');let v=null,asNum=p[p.length-1]==='num';
  if(p[0]==='sweep'){const r=DK.sweep.find(r=>r.method===p[1]&&Math.abs(r.strength-(+p[2]))<1e-9);v=r?r[p[3]]:null;}
  else if(p[0]==='njudges'){el.textContent=['zero','one','two','three','four','five'][JUDGES.length]||String(JUDGES.length);return;}
  else if(p[0]==='judge-spread'){const vs=JUDGES.map(j=>turnVal(j,'prompt','conceptor_tok',5,'fully_rate')).filter(x=>x!=null);
    el.textContent=vs.length?pct(Math.min(...vs))+'–'+pct(Math.max(...vs)):'–';return;}
  else{const [ctx,m,where,key]=p;
    v=mean(JUDGES.map(j=>{const h=holdRow(j,ctx,m);if(!h)return null;if(where==='ordinary')return h.ordinary[key];
      const t=+where.slice(1);return turnVal(j,ctx,m,t,key);}));}
  el.textContent=asNum||(p[3]==='coherence')?num(v):pct(v);
});
