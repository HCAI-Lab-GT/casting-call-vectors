/* ================= Part 2 toy plates (2.1–2.6, 2.8, 2.9), Part 3 zoo, Part 4.1 toy dynamics ================= */

/* ---------- Plate 2.1: an arrow forgets the shape ---------- */
const s1={stretch:.32,tilt:28};
const Z1p=gaussians(120,11),Z1a=gaussians(90,23);
const MU_A=[-.6,-.38],MU_P=[.45,.35];
const A1=cloud(Z1a,MU_A,mm(rot(15*deg),diag(.15,.115)));
const fig1=Fig('c1',.8,[-1.15,1.45,-1.0,1.3],(c,V)=>{
  const L=mm(rot(s1.tilt*deg),diag(s1.stretch,.085));
  const P=cloud(Z1p,MU_P,L);
  dots(c,V,A1,COL.assist,2.3,.6);
  poly(c,V,shapeEll(mm(L,T(L)),MU_P,2),{fill:COL.wash});
  dots(c,V,P,COL.pirate,2.4,.65);
  poly(c,V,shapeEll(mm(L,T(L)),MU_P,2),{stroke:COL.pirate,width:1.8});
  arrow(c,V,MU_A,MU_P,COL.ink,2.4,11);
  label(c,V,'assistant states',[-1.08,-0.92],{color:COL.ink});
  label(c,V,'pirate states and their shape',[1.40,1.22],{align:'right',color:COL.ink});
  const d=sub(MU_P,MU_A),n=norm(d),pp=[-d[1]/n,d[0]/n];
  label(c,V,'steering vector',add(lerp(MU_A,MU_P,.42),sc(pp,.11)),{align:'right',color:COL.ink});
});
function upd1(){
  out('p1s',s1.stretch.toFixed(2));out('p1t',s1.tilt+'°');
  const d=sub(MU_P,MU_A);
  rows($('r1'),[['Arrow',`(${f2(d[0])}, ${f2(d[1])})  unchanged`],['Shape',`long ${s1.stretch.toFixed(2)} · short 0.09 · tilt ${s1.tilt}°`]]);
  fig1.draw();
}
onRange('p1s',v=>{s1.stretch=v;upd1();});
onRange('p1t',v=>{s1.tilt=v;upd1();});

/* ---------- shared pirate cloud for plates 2.2, 2.3, 2.6 ---------- */
const Z2=gaussians(150,5);
const L2=mm(rot(28*deg),diag(.5,.17));
const P2=cloud(Z2,[0,0],L2);
const R2=corr(P2);
const E2=eigSym(R2);

/* ---------- Plate 2.2: the stencil ---------- */
const s2={alpha:5};
const fig2=Fig('c2',.86,[-1.3,1.3,-1.25,1.25],(c,V)=>{
  const C=conceptor(R2,s2.alpha);
  seg(c,V,sc(E2.u1,-1.25),sc(E2.u1,1.25),COL.hair,1,[3,4]);
  seg(c,V,sc(E2.u2,-1.25),sc(E2.u2,1.25),COL.hair,1,[3,4]);
  dots(c,V,P2,COL.pirate,2.1,.28);
  poly(c,V,circlePts([0,0],1),{stroke:COL.muted,width:1.2,dash:[4,4]});
  poly(c,V,imgCircle(C),{fill:COL.wash});
  for(let k=0;k<20;k++){const t=k/20*2*Math.PI,q=[Math.cos(t),Math.sin(t)];arrow(c,V,q,mv(C,q),COL.muted,1.1,6,.8);}
  poly(c,V,imgCircle(C),{stroke:COL.pirate,width:2.4});
  label(c,V,'long direction',sc(E2.u1,1.13),{align:'center',color:COL.ink});
  label(c,V,'short direction',sc(E2.u2,1.13),{align:'center',color:COL.ink});
});
function upd2(){
  out('p2a',fmtA(s2.alpha));
  const k1=keepOf(E2.l[0],s2.alpha),k2=keepOf(E2.l[1],s2.alpha);
  bars($('b2'),[['Long direction',k1,null,pct(k1)],['Short direction',k2,null,pct(k2)]],1,false);
  fig2.draw();
}
onRange('p2a',v=>{s2.alpha=sliderToAlpha(v);upd2();});

/* ---------- Plate 2.3: five edits ---------- */
const s3={mode:'conceptor',alpha:5,beta:1,push:.4,tau:.35,h:[.2,.75],t:1,from:null};
const U1=E2.u1,U2=E2.u2;
const G3=[];for(let i=-4;i<=4;i++)for(let j=-4;j<=4;j++)G3.push([i*.25,j*.25]);
const CAPS={
  push:'<strong>Push</strong> (steering vector). Everything shifts the same way by the same amount, wherever it started.',
  cap:'<strong>Cap</strong> (the Assistant Axis rule; LiSeCo with one direction). States past the line stay put. The rest get lifted just enough to reach it.',
  cut:'<strong>Hard cut</strong> (top‑1 PCA). Keep the long direction, delete everything else.',
  conceptor:'<strong>Conceptor</strong>. Keep each direction in proportion to how much the pirate cloud uses it, then multiply by the gain.',
  not:'<strong>NOT conceptor</strong> (I − C). The reverse: squeeze what the pirate cloud uses, keep what it ignores. This is how conceptors remove a concept.'
};
function F3(mode,p){
  switch(mode){
    case 'push':return add(p,sc(U1,s3.push));
    case 'cap':{const k=Math.max(0,s3.tau-dot(U1,p));return add(p,sc(U1,k));}
    case 'cut':return sc(U1,dot(U1,p));
    case 'conceptor':return sc(mv(conceptor(R2,s3.alpha),p),s3.beta);
    case 'not':return sc(mv(NOT(conceptor(R2,s3.alpha)),p),s3.beta);
  }
  return p;
}
function pos3(){
  const pts=G3.concat([s3.h]);const e=ease(s3.t);
  return pts.map((p,i)=>{const to=F3(s3.mode,p);return (s3.from&&s3.t<1)?lerp(s3.from[i],to,e):to;});
}
const B3=[-1.3,1.6,-1.25,1.45];
const fig3=Fig('c3',.86,B3,(c,V)=>{
  dots(c,V,P2,COL.pirate,2,.16);
  if(s3.mode==='conceptor'||s3.mode==='not'){
    const C=conceptor(R2,s3.alpha),M=s3.mode==='not'?NOT(C):C;
    poly(c,V,imgCircle(M),{stroke:COL.pirate,width:1.4,dash:[5,4],alpha:.8});
  }
  if(s3.mode==='cap'){
    const base=sc(U1,s3.tau);
    seg(c,V,add(base,sc(U2,-2.2)),add(base,sc(U2,2.2)),COL.pirate,1.8,[6,4]);
    label(c,V,'line',add(base,sc(U2,-1.05)),{dx:6,color:COL.ink});
  }
  if(s3.mode==='cut'||s3.mode==='push')seg(c,V,sc(U1,-1.8),sc(U1,1.9),COL.pirate,1.2,[6,4],.8);
  const P=pos3(),n=G3.length;
  dots(c,V,G3,COL.muted,1.9,.35);
  c.globalAlpha=.28;c.strokeStyle=COL.muted;c.lineWidth=1;c.beginPath();
  for(let i=0;i<n;i++){c.moveTo(V.X(G3[i][0]),V.Y(G3[i][1]));c.lineTo(V.X(P[i][0]),V.Y(P[i][1]));}
  c.stroke();c.globalAlpha=1;
  dots(c,V,P.slice(0,n),COL.ink,2.3,.9);
  const h=s3.h,h2=P[n];
  arrow(c,V,h,h2,COL.probe,2.2,9);
  dots(c,V,[h],COL.probe,6,1,true);
  dots(c,V,[h2],COL.probe,5,1);
  label(c,V,'yours',h,{dx:10,dy:-10,color:COL.ink});
});
function cfg3(){
  const m=s3.mode,s=$('p3s');
  $('cap3').innerHTML=CAPS[m];
  $('w3a').hidden=!(m==='conceptor'||m==='not');
  $('w3s').hidden=(m==='cut');
  if(m==='push'){$('l3s').textContent='Push amount';s.min=0;s.max=.6;s.step=.01;s.value=s3.push;}
  else if(m==='cap'){$('l3s').textContent='Line position';s.min=-.4;s.max=.9;s.step=.01;s.value=s3.tau;}
  else{$('l3s').textContent='Gain β';s.min=.25;s.max=2.5;s.step=.05;s.value=s3.beta;}
}
function upd3(){
  const m=s3.mode;
  out('p3s',m==='push'?s3.push.toFixed(2):m==='cap'?f2(s3.tau):s3.beta.toFixed(2));
  out('p3a',fmtA(s3.alpha));
  const h=s3.h,h2=F3(m,h);
  rows($('r3'),[
    ['Along long direction',`${f2(dot(U1,h))} → ${f2(dot(U1,h2))}`],
    ['Along short direction',`${f2(dot(U2,h))} → ${f2(dot(U2,h2))}`],
    ['Moved by',norm(sub(h2,h)).toFixed(2)]
  ]);
  fig3.draw();
}
onSeg('m3',v=>{if(v===s3.mode)return;s3.from=pos3();s3.mode=v;cfg3();upd3();animate(s3,fig3);});
onRange('p3s',v=>{if(s3.mode==='push')s3.push=v;else if(s3.mode==='cap')s3.tau=v;else s3.beta=v;s3.t=1;upd3();});
onRange('p3a',v=>{s3.alpha=sliderToAlpha(v);s3.t=1;upd3();});
draggable(fig3,()=>s3.h,p=>{s3.h=p;s3.t=1;upd3();},B3);

/* ---------- Plate 2.4: raw vs centered fit ---------- */
const s4={fit:'raw',alpha:5,beta:1,t:1,from:null};
const MU4P=[1.15,.85],MU4A=[.85,.12];
const P4=cloud(gaussians(130,41),MU4P,mm(rot(28*deg),diag(.3,.09)));
const A4=cloud(gaussians(90,57),MU4A,mm(rot(-12*deg),diag(.13,.1)));
const R4raw=corr(P4),R4cen=corr(P4,MU4P);
const E4raw=eigSym(R4raw),E4cen=eigSym(R4cen);
function g4(fit,p){
  if(fit==='raw')return sc(mv(conceptor(R4raw,s4.alpha),p),s4.beta);
  return add(MU4P,sc(mv(conceptor(R4cen,s4.alpha),sub(p,MU4P)),s4.beta));
}
function pos4(){const e=ease(s4.t);return A4.map((p,i)=>{const to=g4(s4.fit,p);return (s4.from&&s4.t<1)?lerp(s4.from[i],to,e):to;});}
const fig4=Fig('c4',.84,[-.4,2.1,-.4,1.7],(c,V)=>{
  const raw=s4.fit==='raw';
  const anchor=raw?[0,0]:MU4P,E=raw?E4raw:E4cen,C=conceptor(raw?R4raw:R4cen,s4.alpha),r=raw?.5:.42;
  seg(c,V,add(anchor,sc(E.u1,raw?-.6:-1.5)),add(anchor,sc(E.u1,raw?2.8:1.5)),COL.pirate,1.4,[6,4],.9);
  poly(c,V,circlePts(anchor,r),{stroke:COL.muted,width:1,dash:[3,4],alpha:.8});
  poly(c,V,imgCircle(C,anchor,r),{stroke:COL.pirate,width:2,fill:COL.wash});
  dots(c,V,P4,COL.pirate,2.3,.6);
  const P=pos4();
  dots(c,V,A4,COL.assist,2.4,.7,true);
  c.globalAlpha=.25;c.strokeStyle=COL.assist;c.lineWidth=1;c.beginPath();
  for(let i=0;i<A4.length;i++){c.moveTo(V.X(A4[i][0]),V.Y(A4[i][1]));c.lineTo(V.X(P[i][0]),V.Y(P[i][1]));}
  c.stroke();c.globalAlpha=1;
  dots(c,V,P,COL.assist,2.4,.95);
  const o=[V.X(0),V.Y(0)];c.strokeStyle=COL.ink;c.lineWidth=1.6;c.beginPath();c.moveTo(o[0]-6,o[1]-6);c.lineTo(o[0]+6,o[1]+6);c.moveTo(o[0]+6,o[1]-6);c.lineTo(o[0]-6,o[1]+6);c.stroke();
  label(c,V,'zero',[0,0],{dx:10,dy:12,color:COL.ink});
  label(c,V,'pirate states',[MU4P[0]+.05,MU4P[1]+.42],{align:'center',color:COL.ink});
  label(c,V,'assistant states',[MU4A[0]+.25,MU4A[1]-.3],{align:'center',color:COL.ink});
  label(c,V,raw?'fit anchored at zero':'fit anchored at the pirate center',[-.32,1.58],{color:COL.ink});
});
function upd4(){
  out('p4b',s4.beta.toFixed(2));out('p4a',fmtA(s4.alpha));
  const raw=s4.fit==='raw',E=raw?E4raw:E4cen;
  $('cap4').innerHTML=raw
    ?'<strong>h′ = β · C · h</strong>, with C fitted on raw pirate states. Its main direction runs from zero toward the cloud.'
    :'<strong>h′ = center + β · C · (h − center)</strong>, with C fitted on the spread around the pirate center. Its main direction follows the cloud\'s own stretch.';
  const before=A4.reduce((s,p)=>s+norm(sub(p,MU4P)),0)/A4.length;
  const after=A4.reduce((s,p)=>s+norm(sub(g4(s4.fit,p),MU4P)),0)/A4.length;
  const ang=Math.atan2(E.u1[1],E.u1[0])/deg;
  rows($('r4'),[
    ['Main direction',`${ang.toFixed(0)}° ${raw?'(points at the cloud)':'(the cloud\'s stretch)'}`],
    ['Keeps along main / other',`${pct(keepOf(E.l[0],s4.alpha))} / ${pct(keepOf(E.l[1],s4.alpha))}`],
    ['Assistant → pirate center',`avg distance ${before.toFixed(2)} → ${after.toFixed(2)}`]
  ]);
  fig4.draw();
}
onSeg('m4',v=>{if(v===s4.fit)return;s4.from=pos4();s4.fit=v;upd4();animate(s4,fig4);});
onRange('p4b',v=>{s4.beta=v;s4.t=1;upd4();});
onRange('p4a',v=>{s4.alpha=sliderToAlpha(v);s4.t=1;upd4();});

/* ---------- Plate 2.5: logic ---------- */
const s5={op:'and',theta:45,alpha:4};
const ZA5=gaussians(80,71),ZB5=gaussians(80,83);
const CAP5={
  A:'<strong>Pirate</strong> on its own.',
  B:'<strong>Doctor</strong> on its own.',
  or:'<strong>OR</strong>: room for either. Same as pooling both clouds before fitting.',
  and:'<strong>AND</strong>: only the room both stencils allow. It shrinks fast when the clouds point different ways.',
  notA:'<strong>NOT pirate</strong>: everything the pirate stencil squeezes, passed; everything it keeps, squeezed.',
  andnot:'<strong>Pirate AND NOT doctor</strong>: pirate room with the doctor part taken out.'
};
function mats5(){
  const LA=mm(rot(15*deg),diag(.5,.15)),LB=mm(rot((15+s5.theta)*deg),diag(.45,.16));
  const RA=mm(LA,T(LA)),RB=mm(LB,T(LB));
  const CA=conceptor(RA,s5.alpha),CB=conceptor(RB,s5.alpha);
  let C;
  switch(s5.op){
    case 'A':C=CA;break;case 'B':C=CB;break;
    case 'or':C=conceptor(madd(RA,RB),s5.alpha);break;
    case 'and':C=AND(CA,CB);break;
    case 'notA':C=NOT(CA);break;
    case 'andnot':C=AND(CA,NOT(CB));break;
  }
  return {LA,LB,CA,CB,C};
}
const fig5=Fig('c5',.86,[-1.25,1.25,-1.2,1.2],(c,V)=>{
  const m=mats5();
  dots(c,V,cloud(ZA5,[0,0],m.LA),COL.pirate,2,.22);
  dots(c,V,cloud(ZB5,[0,0],m.LB),COL.orange,2,.22);
  poly(c,V,circlePts([0,0],1),{stroke:COL.muted,width:1.1,dash:[4,4]});
  poly(c,V,imgCircle(m.CA),{stroke:COL.pirate,width:1.5});
  poly(c,V,imgCircle(m.CB),{stroke:COL.orange,width:1.5});
  poly(c,V,imgCircle(m.C),{stroke:COL.ink,width:2.8,fill:COL.ink,fillAlpha:.07});
  label(c,V,'pirate',mv(m.CA,[Math.cos(15*deg),Math.sin(15*deg)]),{dx:8,dy:-10,color:COL.ink});
  const tb=(15+s5.theta)*deg;
  label(c,V,'doctor',mv(m.CB,[Math.cos(tb),Math.sin(tb)]),{dx:8,dy:-10,color:COL.ink});
});
function upd5(){
  out('p5t',s5.theta+'°');
  $('cap5').innerHTML=CAP5[s5.op];
  const m=mats5(),room=C=>(C[0]+C[3])/2;
  rows($('r5'),[['Room left (average keep)',''],['  pirate',pct(room(m.CA))],['  doctor',pct(room(m.CB))],['  result',pct(room(m.C))]]);
  fig5.draw();
}
onSeg('m5',v=>{s5.op=v;upd5();});
onRange('p5t',v=>{s5.theta=v;upd5();});

/* ---------- Plate 2.6: gain check, with the matrix-removed control ---------- */
const s6={mode:'conceptor',ctl:'full',alpha:5,beta:1};
function fullMult6(){const k1=keepOf(E2.l[0],s6.alpha),k2=keepOf(E2.l[1],s6.alpha);return s6.mode==='not'?[s6.beta*(1-k1),s6.beta*(1-k2)]:[s6.beta*k1,s6.beta*k2];}
function fullG6(){const C=conceptor(R2,s6.alpha);return msc(s6.mode==='not'?NOT(C):C,s6.beta);}
const fig6=Fig('c6',.86,[-2.6,2.6,-2.5,2.5],(c,V)=>{
  const removed=s6.ctl==='removed',G=removed?msc(I2,s6.beta):fullG6();
  const m=removed?[s6.beta,s6.beta]:fullMult6(),stretch=Math.min(m[0],m[1])>1;
  seg(c,V,sc(E2.u1,-2.6),sc(E2.u1,2.6),COL.hair,1,[3,4]);
  seg(c,V,sc(E2.u2,-2.6),sc(E2.u2,2.6),COL.hair,1,[3,4]);
  poly(c,V,circlePts([0,0],1),{stroke:COL.muted,width:1.3,dash:[4,4]});
  if(removed)poly(c,V,imgCircle(fullG6()),{stroke:COL.ink,width:1.2,dash:[2,3],alpha:.6});
  for(let k=0;k<20;k++){const t=k/20*2*Math.PI,q=[Math.cos(t),Math.sin(t)];arrow(c,V,q,mv(G,q),COL.muted,1.1,6,.8);}
  poly(c,V,imgCircle(G),{stroke:stretch?COL.probe:COL.pirate,width:2.4});
  label(c,V,'long',sc(E2.u1,Math.max(1,m[0])+.22),{align:'center',color:COL.ink});
  label(c,V,'short',sc(E2.u2,Math.max(1,m[1])+.22),{align:'center',color:COL.ink});
  if(removed)label(c,V,'dotted: full map',[-2.45,-2.3],{color:COL.ink});
});
function upd6(){
  out('p6b',s6.beta.toFixed(2));out('p6a',fmtA(s6.alpha));
  const m=fullMult6(),col=x=>x>1?'var(--probe)':'var(--pirate)';
  bars($('b6'),[
    ['Full · long',m[0],col(m[0]),m[0].toFixed(2)+'×'],
    ['Full · short',m[1],col(m[1]),m[1].toFixed(2)+'×'],
    ['Matrix removed',s6.beta,'var(--none)',s6.beta.toFixed(2)+'×']
  ],3,true);
  const lo=Math.min(m[0],m[1]),hi=Math.max(m[0],m[1]),ratio=hi/Math.max(lo,1e-9);
  const gap=Math.max(Math.abs(m[0]-s6.beta),Math.abs(m[1]-s6.beta))/Math.max(s6.beta,1e-9);
  let s=lo>1?'Full map stretches every direction':hi<1?'Full map shrinks every direction':'Full map shrinks some directions, stretches others';
  s+=' · directions differ by '+ratio.toFixed(2)+'×';
  s+=' · at most '+pct(gap)+' away from the control';
  if(gap<.2)s+=', so it is mostly the gain';
  const el=$('s6');el.textContent=s;el.style.color=lo>1?'var(--probe)':'var(--ink)';
  fig6.draw();
}
onSeg('m6',v=>{s6.mode=v;upd6();});
onSeg('m6c',v=>{s6.ctl=v;upd6();});
onRange('p6b',v=>{s6.beta=v;upd6();});
onRange('p6a',v=>{s6.alpha=sliderToAlpha(v);upd6();});
$('preset6').addEventListener('click',()=>{
  // largest keep fraction 0.17 and gain 2: the spectrum of the audited operating point
  const a=E2.l[0]*(1/0.17-1);s6.alpha=1/Math.sqrt(a);s6.beta=2;s6.mode='not';
  setSeg('m6','not');$('p6b').value=2;$('p6a').value=alphaToSlider(s6.alpha);upd6();
});

/* ---------- Plate 2.8: six conceptor variants ---------- */
const MU8P=[1.05,.8],MU8A=[.72,.08];
const P8=cloud(gaussians(110,141),MU8P,mm(rot(28*deg),diag(.28,.085)));
const A8=cloud(gaussians(80,157),MU8A,mm(rot(-12*deg),diag(.13,.1)));
const DIR8=unit(sub(MU8A,MU8P));
const MU8F=lerp(MU8P,MU8A,.45);
const F8=cloud(gaussians(60,163),MU8F,mm(rot(Math.atan2(DIR8[1],DIR8[0])),diag(.17,.055)));
const MU8G=mean(P8.concat(A8));
const AL8=4;
const C8={
  raw:conceptor(corr(P8),AL8),
  global:conceptor(corr(P8,MU8G),AL8),
  target:conceptor(corr(P8,MU8A),AL8),
  bipolar:conceptor(corr(P8.concat(A8),MU8G),AL8),
  plus:conceptor(corr(P8,MU8G),AL8),
  minus:conceptor(corr(F8,MU8G),AL8),
  own:conceptor(corr(P8,MU8P),AL8)
};
C8.contrast=AND(C8.plus,NOT(C8.minus));
const U8=unit(sub(MU8P,MU8A)),L8=norm(sub(MU8P,MU8A));
const score8=h=>dot(sub(h,MU8A),U8)/L8;
const VAR8={
  raw:{b:1.3,name:'Raw',who:'Postmus &amp; Abreu 2024',eq:'h′ = β · C · h',
    fit:'C fitted on raw pirate states, with no centering.',
    why:'The first conceptor steering of LLMs. Its main direction mostly points at <em>where</em> pirate states sit, so every state slides along one line through zero (Plate 2.4).'},
  global:{b:1.0,name:'Global-centered',who:'Postmus &amp; Abreu 2024',eq:'h′ = μ + β · C · (h − μ)',
    fit:'μ = mean of all states (both clouds); C fitted on pirate states measured from μ.',
    why:'Subtracting the shared mean removes the common offset and improved their results. The stencil still contains the pirate cloud\'s offset from μ.'},
  target:{b:.6,name:'Target-centered NOT',who:'Razdan &amp; Balamurugan 2026 (audit)',eq:'h′ = m + β · (I − C) · (h − m)',
    fit:'m = the target center (here the assistant mean); C fitted on pirate states measured from m.',
    why:'Removes a trait: squeeze what the trait uses, around where you want to end up. The audit found that at small apertures I − C is close to I, so the map acts mostly as the gain β (Plate 2.6).'},
  bipolar:{b:.7,name:'Bipolar',who:'Triantafyllopoulos et al. 2026',eq:'h′ = (1 − β) · h + β · (μ + C(h − μ))',
    fit:'One C fitted on both poles pooled (pirate and assistant), measured from their joint mean.',
    why:'The stencil keeps the axis that separates the poles and squeezes the rest. It does not pick a pole on its own. Interpolating with h keeps part of every direction, unlike replacing h with Ch.'},
  contrast:{b:.8,name:'Contrastive',who:'COAST, Miao et al. 2026',eq:'h′ = μ + [(1 − β) I + β · (C₊ ∧ ¬C₋)] (h − μ)',
    fit:'C₊ fitted on successes (here, in-character turns), C₋ on failures (turns that broke character).',
    why:'Keeps the room successes use and failures do not, then blends it in softly. Built for robot policies; COAST also compares it with a random subspace that has the same spectrum.'},
  gated:{b:.9,name:'Gated (untested)',who:'Proposed here; no paper yet',eq:'h′ = (1 − g) · h + g · (m + C(h − m)),   g = β · σ((τ − s(h)) / w)',
    fit:'C fitted on pirate states around their own mean m; gate g from a persona score s(h), with τ = ½.',
    why:'The conceptor supplies the shape and the gate supplies the "when": in-character states (g ≈ 0) pass untouched, so nothing is added to them turn after turn. Nobody has tested this on personas or across turns.'}
};
const s8={v:'raw',beta:VAR8.raw.b,t:1,from:null};
function map8(v,h){
  const b=s8.beta;
  switch(v){
    case 'raw':return sc(mv(C8.raw,h),b);
    case 'global':return add(MU8G,sc(mv(C8.global,sub(h,MU8G)),b));
    case 'target':return add(MU8A,sc(mv(NOT(C8.target),sub(h,MU8A)),b));
    case 'bipolar':{const d=sub(h,MU8G);return add(MU8G,add(sc(d,1-b),sc(mv(C8.bipolar,d),b)));}
    case 'contrast':{const M=madd(msc(I2,1-b),msc(C8.contrast,b));return add(MU8G,mv(M,sub(h,MU8G)));}
    case 'gated':{const g=Math.min(1,b)*sigmoid((.5-score8(h))/.07);return lerp(h,add(MU8P,mv(C8.own,sub(h,MU8P))),g);}
  }
  return h;
}
const ALL8=()=>s8.v==='contrast'?A8.concat(P8,F8):A8.concat(P8);
function pos8(){
  const e=ease(s8.t),src=ALL8();
  return src.map((p,i)=>{const to=map8(s8.v,p);return (s8.from&&s8.t<1&&s8.from.length===src.length)?lerp(s8.from[i],to,e):to;});
}
const fig8=Fig('c8',.84,[-.35,1.75,-.4,1.45],(c,V)=>{
  const v=s8.v,r=.32;
  const anchor={raw:[0,0],global:MU8G,target:MU8A,bipolar:MU8G,contrast:MU8G,gated:MU8P}[v];
  if(v==='raw')seg(c,V,[0,0],sc(unit(MU8P),2.1),COL.pirate,1.2,[6,4],.8);
  if(v==='gated'){
    const mid=lerp(MU8A,MU8P,.5),pp=perp(U8);
    seg(c,V,add(mid,sc(pp,-1.2)),add(mid,sc(pp,1.2)),COL.muted,1.3,[5,4]);
    label(c,V,'gate = ½',add(mid,sc(pp,.62)),{dx:6,color:COL.ink});
  }
  poly(c,V,circlePts(anchor,r),{stroke:COL.muted,width:1,dash:[3,4],alpha:.8});
  if(v==='contrast'){
    poly(c,V,imgCircle(C8.plus,anchor,r),{stroke:COL.pirate,width:1.3});
    poly(c,V,imgCircle(C8.minus,anchor,r),{stroke:COL.orange,width:1.3});
    poly(c,V,imgCircle(C8.contrast,anchor,r),{stroke:COL.ink,width:2.4,fill:COL.ink,fillAlpha:.06});
  }else{
    const M={raw:C8.raw,global:C8.global,target:NOT(C8.target),bipolar:C8.bipolar,gated:C8.own}[v];
    poly(c,V,imgCircle(M,anchor,r),{stroke:COL.pirate,width:2,fill:COL.wash});
  }
  const src=ALL8(),P=pos8(),nA=A8.length,nP=P8.length;
  const colOf=i=>i<nA?COL.assist:i<nA+nP?COL.pirate:COL.orange;
  c.globalAlpha=.2;c.lineWidth=1;
  for(let i=0;i<src.length;i++){c.strokeStyle=colOf(i);c.beginPath();c.moveTo(V.X(src[i][0]),V.Y(src[i][1]));c.lineTo(V.X(P[i][0]),V.Y(P[i][1]));c.stroke();}
  c.globalAlpha=1;
  dots(c,V,src.slice(0,nA),COL.assist,2.2,.45,true);
  dots(c,V,src.slice(nA,nA+nP),COL.pirate,2.2,.45,true);
  if(v==='contrast')dots(c,V,src.slice(nA+nP),COL.orange,2.2,.45,true);
  dots(c,V,P.slice(0,nA),COL.assist,2.4,.9);
  dots(c,V,P.slice(nA,nA+nP),COL.pirate,2.4,.9);
  if(v==='contrast')dots(c,V,P.slice(nA+nP),COL.orange,2.4,.9);
  const o=[V.X(0),V.Y(0)];c.strokeStyle=COL.ink;c.lineWidth=1.4;c.beginPath();c.moveTo(o[0]-5,o[1]-5);c.lineTo(o[0]+5,o[1]+5);c.moveTo(o[0]+5,o[1]-5);c.lineTo(o[0]-5,o[1]+5);c.stroke();
  label(c,V,'zero',[0,0],{dx:9,dy:11,color:COL.ink});
  const tagA={raw:'anchor: zero',global:'anchor: joint mean',target:'anchor: assistant mean',bipolar:'anchor: joint mean',contrast:'anchor: joint mean',gated:'anchor: pirate mean'}[v];
  label(c,V,tagA,[-.3,1.38],{color:COL.ink});
});
function upd8(){
  out('p8b',s8.beta.toFixed(2));
  const d=VAR8[s8.v];
  const sA0=A8.reduce((s,p)=>s+score8(p),0)/A8.length,sA1=A8.reduce((s,p)=>s+score8(map8(s8.v,p)),0)/A8.length;
  const mvP=P8.reduce((s,p)=>s+norm(sub(map8(s8.v,p),p)),0)/P8.length;
  $('var8').innerHTML='<h3>'+d.name+'</h3><p class="src">'+d.who+'</p>'+
    '<div class="eq">'+d.eq+'</div>'+
    '<dl class="facts"><dt>Fitted on</dt><dd>'+d.fit+'</dd><dt>Why</dt><dd>'+d.why+'</dd></dl>'+
    '<div class="readout"><div><span class="k">Assistant states, persona score</span><span>'+f2(sA0)+' → '+f2(sA1)+'</span></div>'+
    '<div><span class="k">Pirate states, average move</span><span>'+mvP.toFixed(2)+'</span></div></div>'+
    '<p class="note">Persona score: 0 at the assistant mean, 1 at the pirate mean.</p>';
  fig8.draw();
}
onSeg('m8',v=>{if(v===s8.v)return;s8.from=pos8();s8.v=v;s8.beta=VAR8[v].b;$('p8b').value=s8.beta;upd8();if(s8.from.length===ALL8().length)animate(s8,fig8);else{s8.t=1;fig8.draw();}});
onRange('p8b',v=>{s8.beta=v;s8.t=1;upd8();});

/* ---------- Plate 2.9: the autoconceptor ---------- */
const s9={alpha:sliderToAlpha(460),cue:25};
const SIG9=[1,.5,.3,.15],N9=400,LAM9=.3;
function sim9(){
  const a2=1/(s9.alpha*s9.alpha);
  return SIG9.map(sig=>{
    let c=0;const v=[0];
    for(let t=1;t<=N9;t++){
      const Ez2=t<=s9.cue?sig:c*c*sig;           // cue: fit the input; after: fit its own filtered output
      c=Math.max(0,Math.min(1,c+LAM9*((1-c)*Ez2-a2*c)));
      v.push(c);
    }
    return v;
  });
}
const XS9=Array.from({length:N9+1},(_,i)=>i);
let SIM9=sim9();
const ch9=LineChart('c9',{aspect:.62,xs:()=>XS9,yRange:()=>[0,1],yTicks:()=>[0,.25,.5,.75,1],yFmt:v=>v.toFixed(2),
  xTicks:()=>[0,100,200,300,400],xTitle:'steps',mr:70,
  bands:()=>[{x0:0,x1:s9.cue,color:COL.wash,label:'cue'}],
  refs:()=>[{y:.5,label:'½'}],
  series:()=>SIG9.map((sig,i)=>({name:'σ = '+sig.toFixed(2),short:'σ '+sig.toFixed(2),color:rgba(COL.pirate,[1,.78,.56,.36][i]),values:SIM9[i]})),
  tipTitle:i=>'step '+i,vFmt:v=>v.toFixed(3)});
function upd9(){
  out('p9a',fmtA(s9.alpha));out('p9c',String(s9.cue));
  SIM9=sim9();
  const a2=1/(s9.alpha*s9.alpha);
  const r=[['α⁻²',a2.toFixed(3)],['Survival needs σ above 4α⁻²',(4*a2).toFixed(3)]];
  SIG9.forEach((sig,i)=>{const end=SIM9[i][N9];r.push(['σ = '+sig.toFixed(2),end>.02?'settles at '+end.toFixed(2):'dies (→ 0)']);});
  rows($('r9'),r);
  ch9.f.draw();
}
onRange('p9a',v=>{s9.alpha=sliderToAlpha(v);upd9();});
onRange('p9c',v=>{s9.cue=v;upd9();});

/* ================= Part 3: the method zoo on one toy space ================= */
const MUZA=[-.55,-.35],MUZP=[.5,.42];
const ZA=cloud(gaussians(90,23),MUZA,mm(rot(15*deg),diag(.16,.12)));
const ZP=cloud(gaussians(120,11),MUZP,mm(rot(-25*deg),diag(.32,.09)));
const VZ=sub(MUZP,MUZA),UZ=unit(VZ),LZ=norm(VZ),NZ=perp(UZ),MIDZ=lerp(MUZA,MUZP,.5);
const RZP=corr(ZP,MUZP),EZP=eigSym(RZP),CZP=conceptor(RZP,4);
const SZ=sc(madd(corr(ZA,MUZA),RZP),.5);
const WZ=unit(mv(minv(SZ),VZ));                       // probe normal (LDA direction)
const MUZG=mean(ZA.concat(ZP));
const MUZF=lerp(MUZP,MUZA,.42);
const ZF=cloud(gaussians(70,97),MUZF,mm(rot(Math.atan2(UZ[1],UZ[0])),diag(.2,.06)));
const CZplus=conceptor(corr(ZP,MUZP),4),CZminus=conceptor(corr(ZF,MUZP),4),CZcon=AND(CZplus,NOT(CZminus));
const TAU_CAP=quantile(ZP.map(p=>dot(p,UZ)),.25);
const projZ=p=>dot(sub(p,MUZA),UZ)/LZ;
const PROBE_SC=.08;                                    // probe logit = wᵀ(h − mid)/0.08
const pProbe=h=>sigmoid(dot(WZ,sub(h,MIDZ))/PROBE_SC);
const MAGS_B=UZ,MAGS_R=.16;                            // error direction and trigger radius (√τ)
const SZA=corr(ZA,MUZA),SZP=RZP;
const OT_W=(()=>{const a=msqrt(SZA),ai=misqrt(SZA);return sym(mm(mm(ai,msqrt(mm(mm(a,SZP),a))),ai));})();
const OT_D=[Math.sqrt(SZP[0]/SZA[0]),Math.sqrt(SZP[3]/SZA[3])];
const BZ=[-1.3,1.6,-1.25,1.45];

/* manifold toy: persona samples along a curved sheet; geodesic = shortest path on a 6-neighbour graph */
const ARC_C=[-.95,1.15];
const bez=t=>[(1-t)*(1-t)*MUZA[0]+2*(1-t)*t*ARC_C[0]+t*t*MUZP[0],(1-t)*(1-t)*MUZA[1]+2*(1-t)*t*ARC_C[1]+t*t*MUZP[1]];
const ARC=(()=>{const r=rng(311),o=[];for(let i=0;i<170;i++){const t=i/169,b=bez(t),n=[(r()-.5)*.11,(r()-.5)*.11];o.push({p:add(b,n),t});}return o;})();
const GEO=(()=>{
  const n=ARC.length,K=6,adj=ARC.map(()=>[]);
  for(let i=0;i<n;i++){
    const d=ARC.map((q,j)=>[norm(sub(q.p,ARC[i].p)),j]).filter(x=>x[1]!==i).sort((a,b)=>a[0]-b[0]).slice(0,K);
    d.forEach(([w,j])=>{adj[i].push([j,w]);adj[j].push([i,w]);});
  }
  const src=0,dst=n-1,dist=new Array(n).fill(1e9),prev=new Array(n).fill(-1),done=new Array(n).fill(false);dist[src]=0;
  for(let it=0;it<n;it++){let u=-1;for(let i=0;i<n;i++)if(!done[i]&&(u<0||dist[i]<dist[u]))u=i;if(u<0||dist[u]>=1e9)break;done[u]=true;adj[u].forEach(([v,w])=>{if(dist[u]+w<dist[v]){dist[v]=dist[u]+w;prev[v]=u;}});}
  const path=[];for(let v=dst;v>=0;v=prev[v])path.unshift(ARC[v].p);
  return path;
})();
function along(path,t){
  const L=[0];for(let i=1;i<path.length;i++)L.push(L[i-1]+norm(sub(path[i],path[i-1])));
  const target=t*L[L.length-1];
  for(let i=1;i<path.length;i++)if(L[i]>=target){const u=(target-L[i-1])/((L[i]-L[i-1])||1);return lerp(path[i-1],path[i],u);}
  return path[path.length-1];
}
const nearestArc=p=>Math.min(...ARC.map(q=>norm(sub(q.p,p))));

/* GCAD schematic: three turns of tokens */
const GC=(()=>{
  const r=rng(77),tok=[];
  for(let i=0;i<10;i++)tok.push({k:'S'});
  for(let turn=0;turn<3;turn++){
    for(let i=0;i<6;i++)tok.push({k:'U',turn});
    for(let i=0;i<14;i++){
      const pos=tok.length,spike=(i<2||r()<.18)?1:0;
      tok.push({k:'A',turn,d:Math.max(.02,.55*Math.exp(-pos/48)*(1+.9*spike)+(r()-.5)*.04)});
    }
  }
  return tok;
})();
function gcadEdits(cb){
  const A=GC.filter(t=>t.k==='A'),dbar=A.reduce((s,t)=>s+t.d,0)/A.length;
  const gate=d=>2*cb*sigmoid(14*(d-dbar));
  let resV=0,resG=0;const out=[];
  GC.forEach(t=>{
    if(t.k!=='A'){out.push(null);resV*=.92;resG*=.92;return;}
    const g=gate(t.d),eg=g*t.d;resV=.92*resV+.35;resG=.92*resG+eg*.25;
    out.push({g,eg,resV,resG});
  });
  return {out,dbar};
}

const ZOO={
  caa:{grp:'Arrow',name:'Steering vector',who:'CAA (Rimsky et al. 2024) · Persona Vectors (Chen et al. 2025)',when:'Always on',
    plain:'Add the same arrow to every activation, at every token.',
    learns:'Mean activation in persona minus mean activation out of persona, on judge-filtered answers.',
    edit:'h′ = h + α·v',acts:'Always: every token of every turn.',
    cant:'Tell a state that is already in character from one that is not. Everything moves by the same amount.',
    nums:'Over 10 turns on Qwen2.5-7B, persona-vector steering\'s coherence fell from 76.6 at turn 1 to 57.9 at turn 10 (Kang et al. 2026).',
    math:[['h′ = h + α · v,   v = mean(h | persona) − mean(h | default)','v comes from answer-token averages at one layer. α is chosen by sweeping: too large and the text falls apart.']],
    src:[['Persona Vectors','https://arxiv.org/abs/2507.21509']],
    ctl:{label:'Strength α',min:0,max:1.5,step:.01,val:.8,fmt:v=>v.toFixed(2)},
    F:(p,k)=>add(p,sc(VZ,k)),
    extra:(c,V)=>{arrow(c,V,MUZA,MUZP,COL.ink,2,10,.8);label(c,V,'v',lerp(MUZA,MUZP,.5),{dx:8,dy:-8,color:COL.ink});}},
  gate:{grp:'Arrow',name:'Gated vector',who:'DSAS (Ferrando et al.) · MERA (Hedström et al., ICML 2025)',when:'Gated per token',
    plain:'The same arrow, but a small learned gate decides, token by token, how much of it to apply.',
    learns:'The arrow, plus a probe that predicts when steering is needed. MERA calibrates when to steer and abstains otherwise.',
    edit:'(1 − g)·h + g·(h + α·v)',acts:'Per token, where the gate fires.',
    cant:'Change the shape of the edit: it is still one arrow. The gate needs its own labelled data.',
    nums:'No multi-turn persona results found.',
    math:[['h′ = (1 − g) · h + g · T(h),   g = σ(θᵀh + b),   T(h) = h + α · v','DSAS learns θ and b; with g fixed at 1 it is a plain steering vector. MERA steers only when a probe predicts an error, by a calibrated amount.']],
    src:[['DSAS','https://arxiv.org/abs/2512.03661'],['MERA','https://github.com/annahedstroem/MERA-steering']],
    ctl:{label:'Strength α',min:0,max:1.5,step:.01,val:.9,fmt:v=>v.toFixed(2)},
    gateOf:p=>sigmoid(-dot(sub(p,MIDZ),UZ)/.1),
    F:(p,k)=>add(p,sc(VZ,k*ZOO.gate.gateOf(p))),
    extra:(c,V)=>{seg(c,V,add(MIDZ,sc(NZ,-1.6)),add(MIDZ,sc(NZ,1.6)),COL.muted,1.3,[5,4]);label(c,V,'gate = ½',add(MIDZ,sc(NZ,-.95)),{dx:6,color:COL.ink});},
    more:h=>[['Gate at your activation',ZOO.gate.gateOf(h).toFixed(2)]]},
  cap:{grp:'Half-space',name:'Activation cap',who:'The Assistant Axis (Lu et al. 2026)',when:'Past a threshold',
    plain:'Leave a state alone if it is past a line on one axis; otherwise lift it just to the line.',
    learns:'One axis (a mean difference) and a floor: the 25th percentile of projections from the role rollouts.',
    edit:'h′ = h − v·min(⟨h, v⟩ − τ, 0)',acts:'Only where the projection falls below the floor; at 8 layers for Qwen3-32B and 16 for Llama-3.3-70B, at every token.',
    cant:'See any direction except its one axis.',
    nums:'Nearly 60% fewer harmful answers to persona jailbreaks with capabilities kept. It stopped harmful drift in replayed case-study conversations.',
    math:[['h′ = h − v · min(⟨h, v⟩ − τ, 0)','Clamps the projection on the unit vector v to at least τ. The paper floors the Assistant Axis to keep a model near its default persona. To hold a persona, put the floor on the persona axis instead, as our pilot does (Part 4). A ceiling swaps min for max.']],
    src:[['The Assistant Axis','https://arxiv.org/abs/2601.10387']],
    ctl:{label:'Floor τ',min:-.4,max:.9,step:.01,val:+TAU_CAP.toFixed(2),fmt:v=>f2(v)},
    F:(p,k)=>add(p,sc(UZ,Math.max(0,k-dot(p,UZ)))),
    extra:(c,V,k)=>{const b=sc(UZ,k);seg(c,V,add(b,sc(NZ,-2)),add(b,sc(NZ,2)),COL.pirate,1.8,[6,4]);label(c,V,'floor',add(b,sc(NZ,-.95)),{dx:6,color:COL.ink});}},
  liseco:{grp:'Half-space',name:'LiSeCo',who:'Cheng, Baroni &amp; Amo Alonso',when:'Past a threshold',
    plain:'Train a probe for the concept, then make the smallest move that puts its probability back inside the allowed range.',
    learns:'A logistic probe per layer, from labelled examples.',
    edit:'Smallest shift along the probe\'s normal that reaches probability p.',acts:'Only when the probe is outside the bound.',
    cant:'Promise anything the probe cannot see. The guarantee is about the probe\'s output.',
    nums:'Shown on keeping text away from concepts such as toxicity. No persona or multi-turn results found.',
    math:[['h′ = h + max(0, (logit p − b − wᵀh) / ‖w‖²) · w','The closed-form smallest L2 move that makes σ(wᵀh′ + b) ≥ p. To keep a concept out, flip the inequality. In this toy the probe normal comes from the two clouds\' shapes, so it tilts away from the mean difference.']],
    src:[['LiSeCo','https://arxiv.org/abs/2405.15454']],
    ctl:{label:'Target probability p',min:.5,max:.99,step:.01,val:.9,fmt:v=>v.toFixed(2)},
    F:(p,k)=>{const need=PROBE_SC*Math.log(k/(1-k))-dot(WZ,sub(p,MIDZ));return add(p,sc(WZ,Math.max(0,need)));},
    extra:(c,V,k)=>{const b=add(MIDZ,sc(WZ,PROBE_SC*Math.log(k/(1-k)))),pp=perp(WZ);
      seg(c,V,add(MIDZ,sc(pp,-2)),add(MIDZ,sc(pp,2)),COL.muted,1,[2,3]);
      seg(c,V,add(b,sc(pp,-2)),add(b,sc(pp,2)),COL.pirate,1.8,[6,4]);
      label(c,V,'probe = '+k.toFixed(2),add(b,sc(pp,-.9)),{dx:6,color:COL.ink});},
    more:h=>[['Probe says pirate',pProbe(h).toFixed(2)]]},
  pass:{grp:'Subspace',name:'Erase or rescale',who:'INLP (Ravfogel et al. 2020) · LEACE (Belrose et al. 2023) · PaSS (Malik &amp; ElSherief 2026)',when:'Always on',
    plain:'Find the directions that carry the persona, then delete them (erasure) or scale them up or down (PaSS).',
    learns:'A subspace, from labels: INLP by training and projecting out probes again and again, LEACE in closed form.',
    edit:'h′ = h + (β − 1)·P(h − μ)',acts:'Always.',
    cant:'Keep shared content that lives in the same directions. A hard projection treats every state alike.',
    nums:'No multi-turn persona results found.',
    math:[['h′ = h + (β − 1) · P (h − μ)','P projects onto the persona subspace. β = 0 erases (INLP, LEACE), β = 1 changes nothing, β > 1 amplifies the persona (PaSS). LEACE picks an oblique projection, so that no linear classifier can recover the concept while h changes as little as possible.']],
    src:[['LEACE','https://arxiv.org/abs/2306.03819'],['PaSS','https://arxiv.org/abs/2610.04676']],
    ctl:{label:'Scale β',min:0,max:2,step:.01,val:0,fmt:v=>v.toFixed(2)},
    F:(p,k)=>add(p,sc(UZ,(k-1)*dot(sub(p,MUZG),UZ))),
    extra:(c,V)=>{seg(c,V,add(MUZG,sc(NZ,-2)),add(MUZG,sc(NZ,2)),COL.muted,1.2,[5,4]);seg(c,V,add(MUZG,sc(UZ,-2)),add(MUZG,sc(UZ,2)),COL.pirate,1.2,[6,4],.7);}},
  sea:{grp:'Subspace',name:'Spectral editing',who:'SEA (Qiu et al., NeurIPS 2024)',when:'Always on',
    plain:'Keep the directions that move with good examples and drop the ones that move with bad examples, found with one SVD.',
    learns:'Cross-covariances between neutral activations and positive or negative demonstrations.',
    edit:'Project onto the kept directions: a hard cut.',acts:'Always.',
    cant:'Grade directions softly: each one is kept or dropped.',
    nums:'Training-free; reported on truthfulness and bias benchmarks. No persona results found.',
    math:[['Ω₊ = cov(h, h₊),  Ω₋ = cov(h, h₋):  keep the top singular directions of Ω₊, drop the top ones of Ω₋','A hard version of what a conceptor does softly. In this 2D toy, keeping one direction means projecting onto the pirate cloud\'s long axis.']],
    src:[['Spectral Editing of Activations','https://arxiv.org/abs/2405.09719']],
    ctl:null,
    F:p=>add(MUZG,sc(EZP.u1,dot(sub(p,MUZG),EZP.u1))),
    extra:(c,V)=>{seg(c,V,add(MUZG,sc(EZP.u1,-2.2)),add(MUZG,sc(EZP.u1,2.2)),COL.pirate,1.6,[6,4]);label(c,V,'kept direction',add(MUZG,sc(EZP.u1,-1.15)),{dx:6,dy:-10,color:COL.ink});}},
  mags:{grp:'Subspace',name:'Error subspace',who:'MAGS (Li et al. 2026)',when:'Past a threshold',
    plain:'Learn which directions failures drift along, and pull back only when a state has drifted too far along them.',
    learns:'Per-problem differences between correct and failed reasoning traces, in a few attention heads.',
    edit:'a′ = a − α·BᵀB(a − μ_c)',acts:'Only when ‖B(a − μ_c)‖² > τ.',
    cant:'Work without matched success and failure traces.',
    nums:'Evaluated on reasoning tasks. No persona results found.',
    math:[['fire if ‖B(a − μ_c)‖² > τ,  then  a′ = a − α · BᵀB (a − μ_c)','B holds the top directions of the per-problem differences δᵢ; μ_c is the mean of correct states. For a persona, "correct" could be in-character turns and "failed" turns that broke character.']],
    src:[['MAGS','https://arxiv.org/abs/2605.21770']],
    ctl:{label:'Pull α',min:0,max:1,step:.01,val:.8,fmt:v=>v.toFixed(2)},
    fires:p=>Math.abs(dot(sub(p,MUZP),MAGS_B))>MAGS_R,
    F:(p,k)=>{const o=dot(sub(p,MUZP),MAGS_B);return Math.abs(o)>MAGS_R?sub(p,sc(MAGS_B,k*o)):p;},
    extra:(c,V)=>{[-1,1].forEach(s=>{const b=add(MUZP,sc(MAGS_B,s*MAGS_R));seg(c,V,add(b,sc(NZ,-2)),add(b,sc(NZ,2)),COL.pirate,1.4,[6,4]);});
      label(c,V,'no-fire band',add(MUZP,sc(NZ,.9)),{dx:4,color:COL.ink});},
    more:h=>[['Fires at your activation',ZOO.mags.fires(h)?'yes':'no']]},
  conceptor:{grp:'Ellipsoid',name:'Conceptor',who:'Jaeger 2014 · Postmus &amp; Abreu 2024 · Triantafyllopoulos et al. 2026',when:'Always on',
    plain:'Around the persona\'s center, squeeze each direction by how little the persona uses it.',
    learns:'The persona cloud\'s spread, direction by direction (its correlation matrix).',
    edit:'h′ = (1 − β)·h + β·(μ + C(h − μ))',acts:'Always.',
    cant:'Leave in-character states alone, unless you add a gate. With fewer samples than dimensions it deletes whatever it has not seen (Plate 2.7).',
    nums:'No multi-turn measurements found. Our 3-conversation pilot is in Part 4.',
    math:[['C = R (R + α⁻² I)⁻¹,   h′ = (1 − β) h + β (μ + C (h − μ))','The interpolation form our pilot uses, with β = 0.3. Plate 2.8 shows five other variants.']],
    src:[['Steering LLMs using Conceptors','https://arxiv.org/abs/2410.16314']],
    ctl:{label:'Strength β',min:0,max:1,step:.01,val:.6,fmt:v=>v.toFixed(2)},
    F:(p,k)=>lerp(p,add(MUZP,mv(CZP,sub(p,MUZP))),k),
    extra:(c,V)=>{poly(c,V,circlePts(MUZP,.42),{stroke:COL.muted,width:1,dash:[3,4]});poly(c,V,imgCircle(CZP,MUZP,.42),{stroke:COL.pirate,width:2,fill:COL.wash});}},
  coast:{grp:'Ellipsoid',name:'Contrastive conceptor',who:'COAST (Miao et al. 2026)',when:'Always on',
    plain:'Fit one stencil on successes and one on failures, keep what successes use and failures do not, and blend that in softly.',
    learns:'Two clouds: success rollouts and failure rollouts.',
    edit:'h′ = μ + [(1 − β)I + β·(C₊ ∧ ¬C₋)](h − μ)',acts:'Always; per control step for robots.',
    cant:'Run without failure data.',
    nums:'Robot policies, compared with a random subspace that has the same spectrum. No persona results found.',
    math:[['C = C₊ ∧ ¬C₋ = (C₊⁻¹ + (I − C₋)⁻¹ − I)⁻¹','AND keeps only room both stencils allow, so directions the failures use get squeezed even where successes also vary. Orange: failure states.']],
    src:[['COAST','https://arxiv.org/abs/2605.17144']],
    ctl:{label:'Strength β',min:0,max:1,step:.01,val:.8,fmt:v=>v.toFixed(2)},
    F:(p,k)=>add(MUZP,mv(madd(msc(I2,1-k),msc(CZcon,k)),sub(p,MUZP))),
    extra:(c,V)=>{dots(c,V,ZF,COL.orange,2.2,.55);
      poly(c,V,imgCircle(CZplus,MUZP,.42),{stroke:COL.pirate,width:1.3});poly(c,V,imgCircle(CZminus,MUZP,.42),{stroke:COL.orange,width:1.3});
      poly(c,V,imgCircle(CZcon,MUZP,.42),{stroke:COL.ink,width:2.2,fill:COL.ink,fillAlpha:.06});}},
  ot:{grp:'Ellipsoid',name:'Distribution map',who:'MiMiC (Singh et al., ICML 2024) · AcT (Rodriguez et al. 2025) · LinEAS (2025)',when:'Always on',
    plain:'Move the whole assistant cloud onto the pirate cloud: same center, same spread.',
    learns:'The mean and covariance of each cloud. No pairing needed.',
    edit:'h′ = μ_P + W(h − μ_A)',acts:'Always.',
    cant:'Leave in-character states alone: applied to them, it pushes them past the target (watch the teal states). AcT also works one dimension at a time, so it cannot rotate a cloud; switch to AcT to see it.',
    nums:'Used for bias and toxicity. No multi-turn persona results found.',
    math:[['W = Σ_A^{−½} (Σ_A^{½} Σ_P Σ_A^{½})^{½} Σ_A^{−½}','The optimal-transport map between two Gaussians (MiMiC). AcT uses a per-dimension version, ω ⊙ h + b. LinEAS learns such maps across layers, end to end.']],
    src:[['Representation Surgery','https://arxiv.org/abs/2402.09631']],
    ctl:{label:'Strength t',min:0,max:1,step:.01,val:1,fmt:v=>v.toFixed(2)},
    opts:[['mimic','MiMiC (full)'],['act','AcT (per dimension)']],
    F:(p,k)=>{const d=sub(p,MUZA);const to=zs.opt.ot==='act'?add(MUZP,[OT_D[0]*d[0],OT_D[1]*d[1]]):add(MUZP,mv(OT_W,d));return lerp(p,to,k);},
    extra:(c,V)=>{poly(c,V,shapeEll(SZP,MUZP,2),{stroke:COL.pirate,width:1.6,dash:[5,4]});label(c,V,'pirate spread (2σ)',add(MUZP,[.05,.45]),{color:COL.ink});}},
  manifold:{grp:'Curved sheet',name:'Geodesic steering',who:'PersonaManifold (Xu, Xu &amp; Wu 2026) · Manifold Steering (Wurgaft et al. 2026)',when:'Always on',
    plain:'Travel between personas along the sheet where real persona states live, instead of in a straight line through empty space.',
    learns:'A neighbour graph over many persona samples.',
    edit:'Move along the shortest path in the graph.',acts:'Always.',
    cant:'Work with only a few personas: the graph needs dense samples.',
    nums:'About 2K personas on 7–8B models. The path query takes under 1 ms per pair, and each point along the path is tested over 20 dialogue turns.',
    math:[['path = shortest path on a k-nearest-neighbour graph of persona states','A straight line between two persona means can leave the data. PersonaManifold also weights each edge by a local metric and smooths the path with a spline. This toy builds a plain 6-neighbour graph over 170 samples and runs Dijkstra.']],
    src:[['PersonaManifold','https://arxiv.org/abs/2609.34571'],['Manifold Steering','https://arxiv.org/abs/2605.05115']],
    ctl:{label:'Progress t',min:0,max:1,step:.01,val:.5,fmt:v=>v.toFixed(2)},custom:'manifold'},
  gcad:{grp:'Attention pathway',name:'Gated attention delta',who:'GCAD, Prompt–Activation Duality (Kang et al. 2026)',when:'Gated per token',
    plain:'Copy what the persona system prompt contributes to attention, and add it back at every token: more on tokens that match the prompt, less on the rest.',
    learns:'How the attention output changes when the persona system prompt is present, keeping only the system-prompt tokens\' contribution (the "cropped" delta).',
    edit:'Add c_i × (cropped attention delta) before the MLP.',acts:'Per token, through the gate c_i.',
    cant:'Run without a contrasting pair of system prompts.',
    nums:'Qwen2.5-7B, 15 traits, 10 turns: coherence 88.6 → 86.7 (standard steering 76.6 → 57.9); trait 91.5 → 93.1.',
    math:[['a′_t = a_t + c_t · Δ,   c_t = 2 c_base · σ(s · (d_t − d̄))','Δ: the cropped attention delta. d_t: how well token t\'s query matches the persona prompt\'s keys, one dot product per head from queries and keys the model already computed; d̄: its average during extraction. The edit enters the attention output before the MLP. The paper traces residual steering\'s multi-turn decay to steered states being written into the KV cache and reused; drawing Δ from system-prompt tokens and gating it avoids most of that.']],
    src:[['Prompt–Activation Duality','https://arxiv.org/abs/2605.10664']],
    ctl:{label:'Gate strength c_base',min:0,max:1,step:.01,val:.5,fmt:v=>v.toFixed(2)},custom:'gcad'}
};
const ZORDER=['caa','gate','cap','liseco','pass','sea','mags','conceptor','coast','ot','manifold','gcad'];
const zs={id:'conceptor',k:{},opt:{ot:'mimic'},h:[-.25,-.05]};
ZORDER.forEach(id=>{if(ZOO[id].ctl)zs.k[id]=ZOO[id].ctl.val;});

(function buildZooList(){
  const groups=[];ZORDER.forEach(id=>{const g=ZOO[id].grp;let G=groups.find(x=>x.g===g);if(!G){G={g,ids:[]};groups.push(G);}G.ids.push(id);});
  $('zooList').innerHTML=groups.map(G=>'<div class="grp"><span>'+G.g+'</span><div>'+G.ids.map(id=>'<button type="button" class="chip" data-m="'+id+'" aria-pressed="false">'+ZOO[id].name+'</button>').join('')+'</div></div>').join('');
  const panel=$('zooCtl1').parentElement,seg=document.createElement('div');
  seg.className='seg';seg.id='zooOpt';seg.setAttribute('role','group');seg.setAttribute('aria-label','Variant');seg.hidden=true;
  panel.insertBefore(seg,$('zooCtl1'));
})();

const figZ=Canvas('cZoo',.86,(c,w,h,f)=>{
  const m=ZOO[zs.id],k=zs.k[zs.id];
  f.V=makeView(w,h,BZ);const V=f.V;
  if(m.custom==='gcad'){paintGcad(c,w,h,k);return;}
  grid(c,V,w,h);
  if(m.custom==='manifold'){paintManifold(c,V,k);return;}
  const F=p=>m.F(p,k);
  if(m.extra)m.extra(c,V,k);
  const before=ZA.concat(ZP),after=before.map(F),nA=ZA.length;
  c.globalAlpha=.16;c.lineWidth=1;
  before.forEach((p,i)=>{c.strokeStyle=i<nA?COL.assist:COL.pirate;c.beginPath();c.moveTo(V.X(p[0]),V.Y(p[1]));c.lineTo(V.X(after[i][0]),V.Y(after[i][1]));c.stroke();});
  c.globalAlpha=1;
  dots(c,V,ZA,COL.assist,2.1,.4,true);dots(c,V,ZP,COL.pirate,2.1,.4,true);
  dots(c,V,after.slice(0,nA),COL.assist,2.3,.85);dots(c,V,after.slice(nA),COL.pirate,2.3,.85);
  const hp=zs.h,h2=F(hp);
  arrow(c,V,hp,h2,COL.probe,2.2,9);
  dots(c,V,[hp],COL.probe,6,1,true);dots(c,V,[h2],COL.probe,5,1);
  label(c,V,'yours',hp,{dx:10,dy:-10,color:COL.ink});
  label(c,V,'assistant',add(MUZA,[-.1,-.42]),{align:'center',color:COL.ink});
  label(c,V,'pirate',add(MUZP,[.52,.18]),{align:'center',color:COL.ink});
});
function paintManifold(c,V,k){
  ARC.forEach(q=>{const col=q.t<.12?COL.assist:q.t>.88?COL.pirate:COL.other;mark(c,V.X(q.p[0]),V.Y(q.p[1]),'o',2.4,col,false,true);});
  poly(c,V,GEO,{closed:false,stroke:COL.ink,width:2});
  seg(c,V,MUZA,MUZP,COL.muted,1.6,[6,4]);
  const g=along(GEO,k),s=lerp(MUZA,MUZP,k);
  dots(c,V,[s],COL.probe,5.5,1,true);dots(c,V,[g],COL.probe,5.5,1);
  label(c,V,'geodesic',bez(.42),{dx:-12,dy:-14,align:'right',color:COL.ink});
  label(c,V,'straight line',lerp(MUZA,MUZP,.62),{dx:8,dy:12,color:COL.ink});
  label(c,V,'assistant',add(MUZA,[-.05,-.2]),{align:'center',color:COL.ink});
  label(c,V,'pirate',add(MUZP,[.2,-.18]),{align:'center',color:COL.ink});
  label(c,V,'other personas',bez(.55),{dx:12,dy:-16,color:COL.ink});
}
function paintGcad(c,w,h,cb){
  const E=gcadEdits(cb),n=GC.length,L=Math.max(100,Math.min(150,w*.26)),R=12,cw=(w-L-R)/n;
  const top=26,tokH=12,gap=12,foot=30,rh=Math.max(24,(h-top-tokH-gap-foot-3*gap)/4);
  const y0=top,y1=y0+tokH+gap,y2=y1+rh+gap,y3=y2+rh+gap,y4=y3+rh+gap;
  [['Tokens',y0+tokH/2],['Prompt match',y1+rh/2],['GCAD edit',y2+rh/2],['Vector edit',y3+rh/2],['Cache carry',y4+rh/2]]
    .forEach(([t,y])=>text(c,t,L-8,y,{size:10,align:'right',color:COL.ink,halo:false}));
  GC.forEach((t,i)=>{
    const x=L+i*cw,bw=Math.max(1,cw-1);
    if(t.k==='A'){c.strokeStyle=COL.muted;c.lineWidth=1;c.strokeRect(x+1,y0+.5,Math.max(1,bw-1),tokH-1);}else{c.fillStyle=t.k==='S'?COL.pirate:COL.other;c.fillRect(x+.5,y0,bw,tokH);}
    const e=E.out[i];if(!e)return;
    const b1=Math.min(1,t.d/.9)*rh;c.fillStyle=COL.muted;c.fillRect(x+.5,y1+rh-b1,bw,b1);
    const b2=Math.min(1,e.eg/.9)*rh;c.fillStyle=COL.pirate;c.fillRect(x+.5,y2+rh-b2,bw,b2);
    const b3=.35/.9*rh;c.fillStyle=COL.orange;c.fillRect(x+.5,y3+rh-b3,bw,b3);
  });
  c.strokeStyle=COL.hair;c.lineWidth=1;
  [y1+rh,y2+rh,y3+rh,y4+rh].forEach(y=>{c.beginPath();c.moveTo(L,Math.round(y)+.5);c.lineTo(w-R,Math.round(y)+.5);c.stroke();});
  const yb=y4+rh,maxR=Math.max(...E.out.filter(Boolean).map(e=>e.resV));
  [['resV',COL.orange],['resG',COL.pirate]].forEach(([key,col])=>{
    c.strokeStyle=col;c.lineWidth=2;c.beginPath();let on=false;
    E.out.forEach((e,i)=>{if(!e)return;const x=L+(i+.5)*cw,y=yb-e[key]/maxR*rh;on?c.lineTo(x,y):c.moveTo(x,y);on=true;});c.stroke();
  });
  text(c,'system',L,y0-9,{size:10,halo:false});
  [0,1,2].forEach(turn=>text(c,'turn '+(turn+1),L+(10+turn*20+6)*cw,y0-9,{size:10,halo:false}));
  const lastV=E.out[n-1].resV,lastG=E.out[n-1].resG;
  text(c,'vector',w-R,yb-lastV/maxR*rh+11,{size:10,align:'right',color:COL.ink});
  text(c,'GCAD',w-R,yb-lastG/maxR*rh-9,{size:10,align:'right',color:COL.ink});
}
function updZoo(){
  const m=ZOO[zs.id],k=zs.k[zs.id];
  document.querySelectorAll('.chip[data-m]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.m===zs.id)));
  const ctl=$('zooCtl1'),inp=$('zs1');
  ctl.hidden=!m.ctl;
  if(m.ctl){$('zs1l').textContent=m.ctl.label;inp.min=m.ctl.min;inp.max=m.ctl.max;inp.step=m.ctl.step;inp.value=k;out('zs1',m.ctl.fmt(k));}
  const opt=$('zooOpt');
  if(m.opts){
    opt.hidden=false;
    if(opt.dataset.for!==zs.id){opt.dataset.for=zs.id;opt.innerHTML=m.opts.map(o=>'<button type="button" data-v="'+o[0]+'">'+o[1]+'</button>').join('');}
    opt.querySelectorAll('button[data-v]').forEach(b=>b.setAttribute('aria-pressed',String(zs.opt[zs.id]===b.dataset.v)));
  }else{opt.hidden=true;opt.innerHTML='';opt.dataset.for='';}
  const r=[];
  if(m.custom==='manifold'){
    const g=along(GEO,k),s=lerp(MUZA,MUZP,k);
    r.push(['Distance to nearest persona sample',''],['  geodesic',nearestArc(g).toFixed(2)],['  straight line',nearestArc(s).toFixed(2)]);
    $('zooCap').textContent='Toy: persona samples lie on a curved sheet. Solid pink: geodesic position at t. Hollow pink: straight-line position at t.';
  }else if(m.custom==='gcad'){
    const E=gcadEdits(k).out.filter(Boolean),mg=E.reduce((s,e)=>s+e.eg,0)/E.length;
    r.push(['Average edit per answer token',''],['  GCAD',mg.toFixed(2)],['  residual vector','0.35'],['Carried in the cache at the last token',''],['  GCAD',E[E.length-1].resG.toFixed(2)],['  residual vector',E[E.length-1].resV.toFixed(2)]);
    $('zooCap').textContent='Schematic, not data: three turns of tokens. Teal: the persona system prompt; grey: user; outlined: the model\'s answers. Bars show how much each method edits each answer token; the bottom row shows how much of those edits is still carried forward.';
  }else{
    const h=zs.h,h2=m.F(h,k);
    r.push(['Your persona score',f2(projZ(h))+' → '+f2(projZ(h2))],['Moved by',norm(sub(h2,h)).toFixed(2)]);
    if(m.more)m.more(h).forEach(x=>r.push(x));
    $('zooCap').textContent='Hollow: states before. Solid: after. Pink: your activation (drag it, or use the arrow keys). Persona score: 0 at the assistant mean, 1 at the pirate mean.';
  }
  rows($('zooRead'),r);
  $('zooCard').innerHTML='<div class="eyebrow">'+m.grp+' · '+m.when+'</div><h3>'+m.name+'</h3><p class="src">'+m.who+'</p>'+
    '<dl class="facts"><dt>Plain words</dt><dd>'+m.plain+'</dd><dt>Learns from</dt><dd>'+m.learns+'</dd><dt>The edit</dt><dd>'+m.edit+'</dd><dt>When</dt><dd>'+m.acts+'</dd><dt>Can\'t do</dt><dd>'+m.cant+'</dd><dt>Numbers</dt><dd>'+m.nums+'</dd></dl>'+
    '<details class="math"><summary>The math</summary>'+m.math.map(x=>'<div class="eq">'+x[0]+'</div><span class="eqnote">'+x[1]+'</span>').join('')+'</details>'+
    '<p class="src">'+m.src.map(s=>'<a href="'+s[1]+'">'+s[0]+'</a>').join(' · ')+'</p>';
  figZ.draw();
}
function selectZoo(id,scroll){
  if(!ZOO[id])return;zs.id=id;updZoo();
  if(scroll){const el=$('zoo-explorer');if(el)el.scrollIntoView({behavior:reduce?'auto':'smooth',block:'start'});}
}
document.addEventListener('click',e=>{const b=e.target.closest('.chip[data-m]');if(!b)return;selectZoo(b.dataset.m,!!b.closest('#shapegrid'));});
$('zooOpt').addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(!b)return;zs.opt[zs.id]=b.dataset.v;updZoo();});
onRange('zs1',v=>{zs.k[zs.id]=v;updZoo();});
draggable(figZ,()=>zs.h,p=>{if(ZOO[zs.id].custom)return;zs.h=p;updZoo();},BZ);

/* ================= Plate 4.1: toy turn-by-turn dynamics ================= */
const s41={pull:.35,mem:.7};
const TURNS41=20,ATT41=[4,8,12,16];
const NOISE41=(()=>{const r=rng(2024);return Array.from({length:TURNS41},()=>(r()-.5)*.1);})();
const XS41=Array.from({length:TURNS41},(_,i)=>i+1);
function sim41(kind){
  let s=1,res=0;const S=[],C=[];
  for(let t=1;t<=TURNS41;t++){
    const a=1/(1+t/6),k=ATT41.includes(t)?1:.15;
    s=s+.25*(1-s)*a-s41.pull*k+NOISE41[t-1];
    if(s>1)s-=.15*(s-1);
    let e=0,side=0;
    if(kind==='vec')e=.15;
    else if(kind==='con'){e=.4*(1-s);side=.08;}
    else if(kind==='cap')e=Math.max(0,.75-s);
    s=Math.max(-1.2,Math.min(1.8,s+e));
    res=s41.mem*res+Math.abs(e)+side;
    S.push(s);C.push(Math.max(0,1-.35*res));
  }
  return {S,C};
}
const KINDS41=[['none','prompt only','prompt','none',[5,4]],['vec','always-on vector','vector','orange'],['con','always-on conceptor','conceptor','pirate'],['cap','cap (acts when needed)','cap','assist']];
const SIM41={};
KINDS41.forEach(k=>{SIM41[k[0]]=sim41(k[0]);});
const ser41=key=>()=>KINDS41.map(k=>({name:k[1],short:k[2],color:COL[k[3]],dash:k[4],values:SIM41[k[0]][key]}));
const bands41=()=>ATT41.map(t=>({x0:t-.45,x1:t+.45,color:COL.grid,alpha:1}));
const chSimP=LineChart('cSimP',{aspect:.72,xs:()=>XS41,yRange:()=>[-1,1.6],yTicks:()=>[-1,-.5,0,.5,1,1.5],yFmt:v=>v.toFixed(1),
  xTicks:()=>[1,4,8,12,16,20],xFmt:v=>String(v),xTitle:'turn (shaded: attack)',mr:88,bands:bands41,
  refs:()=>[{y:1,label:'in character'},{y:0,label:'assistant'},{y:.75,label:'cap floor'}],series:ser41('S'),tipTitle:i=>'turn '+(i+1)+(ATT41.includes(i+1)?' · attack':'')});
const chSimC=LineChart('cSimC',{aspect:.72,xs:()=>XS41,yRange:()=>[0,1.05],yTicks:()=>[0,.25,.5,.75,1],yFmt:v=>v.toFixed(2),
  xTicks:()=>[1,4,8,12,16,20],xFmt:v=>String(v),xTitle:'turn',mr:88,bands:bands41,series:ser41('C'),tipTitle:i=>'turn '+(i+1)});
const tab41=dataTable('cSimC','Show the toy\'s numbers');
function upd41(){
  out('simPull',s41.pull.toFixed(2));out('simMem',s41.mem.toFixed(2));
  KINDS41.forEach(k=>{SIM41[k[0]]=sim41(k[0]);});
  chSimP.f.draw();chSimC.f.draw();
  tab41.set(['turn'].concat(KINDS41.map(k=>k[2]+' score'),KINDS41.map(k=>k[2]+' coherence')),
    XS41.map((t,i)=>[String(t)].concat(KINDS41.map(k=>SIM41[k[0]].S[i].toFixed(2)),KINDS41.map(k=>SIM41[k[0]].C[i].toFixed(2)))));
}
onRange('simPull',v=>{s41.pull=v;upd41();});
onRange('simMem',v=>{s41.mem=v;upd41();});
