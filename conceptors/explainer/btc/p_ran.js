/* ================= what we ran: vector and cap ================= */
(function(){let mode='cap',pc=25,c=1,F='hold';const probe={p:[0.2,-0.9]};
  const projP=W2.P.map(h=>dot(h,W2.vhat)).sort((a,b)=>a-b);
  const tau=()=>quantile(projP,pc/100);
  const edit=h=>{if(mode==='none')return{h,on:false};if(mode==='vector')return{h:add(h,scl(W2.d,c)),on:true};
    const p=dot(h,W2.vhat),t=tau();return p<t?{h:add(h,scl(W2.vhat,t-p)),on:true}:{h,on:false}};
  const pl=new Plot2D($('#c1'),BOX2,draw);pl.handles.push({get:()=>probe.p,set:w=>probe.p=w});
  function draw(){if(!pl.ready())return;loadCols();pl.clear();drawRegions(pl);
    if(mode==='cap'){pl.halfplane(W2.vhat,tau(),A(COL.navy,.035));pl.boundary(W2.vhat,tau(),COL.navy,1.8,null,'floor τ')}
    drawAxis(pl);drawSteer(pl,edit,F);
    const e=edit(probe.p);pl.ring(probe.p);if(e.on)pl.arrow(probe.p,e.h,COL.edit,2.4,9);
    if(F==='set'){const s=setMetrics(edit);reads('#m1',[[f2(s.pos),'persona-axis position after (0 = assistant, 1 = persona)'],[pct(s.landed),'tokens landing in the persona region'],[pct(s.edited),'tokens edited']]);return}
    const m=metrics(edit);reads('#m1',[[pct(m.restore),'restore (drifted)'],[f2(m.dist),'distortion (in character)'],[pct(m.caught),'drifted tokens edited'],[pct(m.ftr),'in-character tokens edited']])}
  seg('#s1',[['none','No steering'],['vector','Vector'],['cap','Cap']],mode,v=>{mode=v;draw()});
  slider('r1t',v=>v+'th',v=>{pc=v;draw()});slider('r1c',v=>(v/100).toFixed(2),v=>{c=v/100;draw()});
  seg('#s_capf',FRAMES,F,v=>{F=v;$('#f_cap').textContent=FRAME_TXT[F];draw()});$('#f_cap').textContent=FRAME_TXT[F];
})();

/* ================= what we ran: the aperture, on real SmolLM3 activations ================= */
(function(){const S=DATA.spectrum,L=S.lam_over_median,Wv=S.vhat_weight,E=S.gap_energy,n=L.length;
  const sw=Wv.reduce((a,b)=>a+b,0),se=E.reduce((a,b)=>a+b,0);
  const kOf=v=>Math.pow(10,v/100);
  const keepAxis=k=>{let s=0;for(let i=0;i<n;i++)s+=Wv[i]*L[i]/(L[i]+k);return s/sw};
  const gapKept=k=>{let s=0;for(let i=0;i<n;i++){const g=L[i]/(L[i]+k);s+=E[i]*g*g}return s/se};
  const nKept=k=>{let c=0;for(let i=0;i<n;i++)if(L[i]>k)c++;return c};
  const NB=44,lgn=Math.log10(n);const bin=i=>Math.min(NB-1,Math.floor(Math.log10(i+1)/lgn*NB));
  const wb=Array(NB).fill(0),eb=Array(NB).fill(0);for(let i=0;i<n;i++){wb[bin(i)]+=Wv[i]/sw;eb[bin(i)]+=E[i]/se}
  const KS=[1,3,10,30,100];
  const SEL=[['smol','SmolLM3, 5 personas'],['smol:pirate','pirate'],['smol:ghost','ghost'],['smol:bard','bard'],['smol:genie','genie'],['smol:alien','alien'],
    ['olmo','OLMo, 2 personas'],['olmo:pirate','OLMo pirate'],['olmo:genie','OLMo genie']];
  let k=kOf(100),sel='smol';
  function view(id){const[model,p]=id.split(':');const ap=DATA.aperture[model];const ps=p?[p]:Object.keys(ap);
    const at=(kk,key)=>{const v=ps.map(q=>ap[q][String(kk)]).filter(Boolean).map(c=>c[key]);return v.length===ps.length?avg(v):null};
    const judges=kk=>Math.min(...ps.map(q=>ap[q][String(kk)]?ap[q][String(kk)].judges:0));
    // the all-persona view reads the same rounded means as the comparison table, so the two never disagree
    const ref=(m,key)=>{if(!p&&DATA.means[model][m])return DATA.means[model][m][key];const v=ps.map(q=>ap[q].ref&&ap[q].ref[m]?ap[q].ref[m][key]:null).filter(x=>x!=null);return v.length===ps.length?avg(v):null};
    return{at,judges,ref,model}}
  const c1=$('#c2s'),g1=c1.getContext('2d'),c2=$('#c2h');watch(c1,()=>draw());watch(c2,()=>draw());
  function drawSpectrum(){const r=fitCanvas(c1,g1);if(!r)return;const W=r.width,H=r.height,font=FONT();
    const pad={l:44,r:14,t:26,b:58},stripH=12;const yTop=pad.t,yBot=H-pad.b;
    const X=i=>pad.l+Math.log10(i)/lgn*(W-pad.l-pad.r),Y=v=>yBot-v*(yBot-yTop);
    g1.font=`500 11px ${font}`;g1.lineWidth=1;
    [0,.25,.5,.75,1].forEach(v=>{g1.strokeStyle=COL.grid;g1.beginPath();g1.moveTo(pad.l,Y(v));g1.lineTo(W-pad.r,Y(v));g1.stroke();g1.fillStyle=COL.muted;g1.textAlign='right';g1.textBaseline='middle';g1.fillText(String(v),pad.l-6,Y(v))});
    [1,10,100,1000,n].forEach(i=>{g1.strokeStyle=COL.grid;g1.beginPath();g1.moveTo(X(i),yTop);g1.lineTo(X(i),yBot);g1.stroke()});
    const kept=nKept(k);if(kept>=1){g1.strokeStyle=A(COL.navy,.7);g1.setLineDash([4,3]);g1.beginPath();g1.moveTo(X(Math.max(1,kept)),yTop);g1.lineTo(X(Math.max(1,kept)),yBot);g1.stroke();g1.setLineDash([]);
      g1.fillStyle=COL.navy;g1.textAlign='left';g1.textBaseline='top';g1.fillText('λ = a',X(Math.max(1,kept))+4,yTop+2)}
    const curve=(kk,col,w,dash)=>{g1.save();g1.strokeStyle=col;g1.lineWidth=w;if(dash)g1.setLineDash(dash);g1.beginPath();for(let i=0;i<n;i++){const x=X(i+1),y=Y(L[i]/(L[i]+kk));i?g1.lineTo(x,y):g1.moveTo(x,y)}g1.stroke();g1.restore()};
    curve(1,A(COL.muted,.9),1.2,[4,3]);curve(k,COL.navy,2.2);
    g1.fillStyle=COL.ink;g1.font=`600 12px ${font}`;g1.textAlign='left';g1.textBaseline='top';g1.fillText(`Keep share λ/(λ + a) of each of the ${n.toLocaleString('en-US')} directions, widest first`,pad.l,5);
    g1.font=`500 11px ${font}`;g1.fillStyle=COL.muted;g1.textAlign='right';g1.fillText('dashed: k = 1',W-pad.r,6);
    const sy1=yBot+8,sy2=sy1+stripH+4;const bx=b=>pad.l+b/NB*(W-pad.l-pad.r),bw=(W-pad.l-pad.r)/NB;const wmax=Math.max(...wb),emax=Math.max(...eb);
    for(let b=0;b<NB;b++){g1.fillStyle=A(COL.persona,.05+.9*wb[b]/wmax);g1.fillRect(bx(b),sy1,bw+.6,stripH);g1.fillStyle=A(COL.assist,.05+.9*eb[b]/emax);g1.fillRect(bx(b),sy2,bw+.6,stripH)}
    g1.font=`600 11px ${font}`;g1.textAlign='right';g1.textBaseline='middle';g1.fillStyle=COL.persona;g1.fillText('axis',pad.l-6,sy1+stripH/2);g1.fillStyle=COL.assist;g1.fillText('gap',pad.l-6,sy2+stripH/2);
    g1.font=`500 11px ${font}`;g1.fillStyle=COL.muted;g1.textAlign='center';g1.textBaseline='top';
    [1,10,100,1000,n].forEach(i=>g1.fillText(i.toLocaleString('en-US'),X(i),sy2+stripH+3));
    g1.textAlign='center';g1.fillText('direction rank (log)',(pad.l+W-pad.r)/2,sy2+stripH+16)}
  function drawHolds(){loadCols();const v=view(sel);const series=[['ord',COL.persona,null],['push',COL.drift,[6,4]],['coh',COL.navy,[2,3]]].map(([key,col,dash])=>({
      pts:KS.map(kk=>{const x=v.at(kk,key);return[kk,x==null?null:(key==='coh'?x:100*x)]}),col,dash,w:2,marker:3.5}));
    lineChart(c2,{title:'Measured 20-turn holds with the persona prompt',x:{min:1,max:100,log:true,ticks:KS,label:'aperture k'},y:{min:0,max:100,ticks:[0,25,50,75,100]},series,vline:clamp(k,1,100)});
    const nearest=KS.reduce((a,b)=>Math.abs(Math.log(b/k))<Math.abs(Math.log(a/k))?b:a);
    const rv=(m)=>{const o=v.ref(m,'ord'),p=v.ref(m,'push'),c=v.ref(m,'coh');return o==null?'n/a':`${pct(o)} / ${pct(p)} / ${f0(c)}`};
    $('#c2note').textContent=`Same holds, same personas: prompt alone ${rv('none')}; vector ${rv('add')}; cap ${rv('cap')}; mean-shift control ${rv('meanshift_tok')} (in character / at pushes / coherence). Points are measured at k = 1, 3, 10, 30 and 100; the lines between them only guide the eye.`;
    return{nearest,v}}
  function draw(){loadCols();drawSpectrum();const h=drawHolds();if(!h)return;const{nearest,v}=h;const j=v.judges(nearest);
    reads('#m2',[[pct(1-keepAxis(k)),'pull along the persona axis per pass'],[nKept(k).toLocaleString('en-US'),'directions keeping more than half'],[pct(gapKept(k)),'share of the gap’s energy kept'],
      [`${pct(v.at(nearest,'ord'))} / ${pct(v.at(nearest,'push'))} / ${f0(v.at(nearest,'coh'))}`,`measured at k = ${nearest}: in character / pushes / coherence`+(j<4?` (${j} judges)`:'')]])}
  slider('r2k',x=>{const kk=kOf(x);return kk<10?kk.toFixed(1):kk.toFixed(0)},x=>{k=kOf(x);draw()});
  seg('#s2p',SEL,sel,v=>{sel=v;draw()});
})();

/* ================= what we ran: a better trigger ================= */
(function(){let gate='fisher',metric='mahal',pc=25;const probe={p:[0.8,-0.2]};
  const SW=W2.SigW,SWi=inv(SW),wF=mv(SWi,W2.d),wFu=unit(wF);
  const raw=h=>dot(h,W2.vhat),fis=h=>dot(h,wF);
  const rawP=W2.P.map(raw).sort((a,b)=>a-b),fisP=W2.P.map(fis).sort((a,b)=>a-b);
  const aucRaw=auroc(W2.Dr.map(raw),W2.P.map(raw)),aucFis=auroc(W2.Dr.map(fis),W2.P.map(fis));
  const pl=new Plot2D($('#c3'),BOX2,draw);pl.handles.push({get:()=>probe.p,set:w=>probe.p=w});
  const hc=$('#c3h'),hg=hc.getContext('2d');watch(hc,()=>draw());
  function hist(scoreP,scoreD,tau){const r=fitCanvas(hc,hg);if(!r)return;const W=r.width,H=r.height,pad={l:12,r:12,t:22,b:22};
    const all=scoreP.concat(scoreD);const lo=Math.min(...all),hi=Math.max(...all);const B=34,bw=(hi-lo)/B;
    const bins=a=>{const h=Array(B).fill(0);a.forEach(x=>h[Math.min(B-1,Math.floor((x-lo)/bw))]++);return h};
    const hp=bins(scoreP),hd=bins(scoreD);const mx=Math.max(...hp,...hd);
    const X=x=>pad.l+(x-lo)/(hi-lo)*(W-pad.l-pad.r),Y=v=>H-pad.b-v/mx*(H-pad.t-pad.b);
    [[hp,COL.persona],[hd,COL.drift]].forEach(([h,col])=>{hg.fillStyle=A(col,.35);hg.strokeStyle=col;hg.lineWidth=1.2;hg.beginPath();hg.moveTo(X(lo),Y(0));
      h.forEach((v,i)=>{hg.lineTo(X(lo+i*bw),Y(v));hg.lineTo(X(lo+(i+1)*bw),Y(v))});hg.lineTo(X(hi),Y(0));hg.closePath();hg.fill();hg.stroke()});
    hg.strokeStyle=gate==='fisher'?COL.violet:COL.navy;hg.lineWidth=2;hg.beginPath();hg.moveTo(X(tau),pad.t-6);hg.lineTo(X(tau),H-pad.b);hg.stroke();
    hg.font=`600 12px ${FONT()}`;hg.fillStyle=COL.ink;hg.textBaseline='top';hg.textAlign='left';
    hg.fillText((gate==='fisher'?'Fisher score w·h':'raw score v̂·h')+'   ← assistant-like   persona-like →',pad.l,4);
    hg.fillStyle=COL.muted;hg.textAlign='right';hg.fillText('floor',X(tau)-4,pad.t);hg.strokeStyle=COL.hair;hg.lineWidth=1;hg.beginPath();hg.moveTo(pad.l,H-pad.b);hg.lineTo(W-pad.r,H-pad.b);hg.stroke()}
  function draw(){if(!pl.ready())return;loadCols();pl.clear();drawRegions(pl);
    const tr=quantile(rawP,pc/100),tf=quantile(fisP,pc/100);
    pl.halfplane(gate==='fisher'?wF:W2.vhat,gate==='fisher'?tf:tr,A(gate==='fisher'?COL.violet:COL.navy,.06));
    pl.boundary(W2.vhat,tr,COL.navy,1.6,[6,4],'raw floor');pl.boundary(wF,tf,COL.violet,2,null,'Fisher floor');
    drawAxis(pl);
    const trig=h=>gate==='fisher'?fis(h)<tf:raw(h)<tr;
    TOK.hold.forEach(h=>trig(h)?pl.dot(h,3.3,A(COL.ink,.88)):pl.dot(h,3.1,A(COL.paper,.9),A(COL.ink,.8),1.2));
    pl.arrow(W2.mP,add(W2.mP,scl(wFu,1.4)),COL.violet,2,8);pl.text(add(W2.mP,scl(wFu,1.65)),'w',COL.violet,{font:'700 13px',align:'center'});
    axisLabels(pl);const fp=fis(probe.p);pl.ring(probe.p);
    if(fp<tf){const delta=metric==='mahal'?scl(mv(SW,wF),(tf-fp)/dot(wF,mv(SW,wF))):scl(wF,(tf-fp)/dot(wF,wF));pl.arrow(probe.p,add(probe.p,delta),COL.edit,2.6,9)}
    const dM=h=>{if(fis(h)>=tf)return 0;const de=scl(mv(SW,wF),(tf-fis(h))/dot(wF,mv(SW,wF)));return Math.sqrt(dot(de,mv(SWi,de)))};
    const dE=h=>{if(fis(h)>=tf)return 0;const de=scl(wF,(tf-fis(h))/dot(wF,wF));return Math.sqrt(dot(de,mv(SWi,de)))};
    const cR=avg(W2.Dr.map(h=>raw(h)<tr?1:0)),cF=avg(W2.Dr.map(h=>fis(h)<tf?1:0));
    reads('#m3',[[pct(cR),'drifted caught, raw floor'],[pct(cF),'drifted caught, Fisher floor'],[f2(aucRaw)+' / '+f2(aucFis),'AUROC raw / Fisher'],
      [f2(dM(probe.p))+' / '+f2(dE(probe.p)),'probe edit in data units: data-metric / Euclidean']]);
    hist(gate==='fisher'?W2.P.map(fis):W2.P.map(raw),gate==='fisher'?W2.Dr.map(fis):W2.Dr.map(raw),gate==='fisher'?tf:tr)}
  seg('#s3g',[['raw','Gate: raw axis'],['fisher','Gate: Fisher']],gate,v=>{gate=v;draw()});
  seg('#s3m',[['mahal','Correction: data metric'],['euclid','Correction: Euclidean']],metric,v=>{metric=v;draw()});
  slider('r3p',v=>v+'%',v=>{pc=v;draw()});
})();

/* ================= what we ran: a region, in 3D ================= */
(function(){const r=mulberry32(5);const muP=[2,0,0],muA=[-2,0,0];const th=32*Math.PI/180;const u1=[Math.cos(th),Math.sin(th),0],u2=[-Math.sin(th),Math.cos(th),0];
  const tok=mu=>add(mu,add(add(scl(u1,1.15*randn(r)),scl(u2,.25*randn(r))),[0,0,.24*randn(r)]));
  const N=120,P=[],As=[],DA=[],DB=[];for(let i=0;i<N;i++){P.push(tok(muP));As.push(tok(muA))}
  for(let i=0;i<60;i++){DA.push(sub(tok(muP),[4*(.3+.5*r()),0,0]));DB.push(add(tok(muP),[0,0,.9+.9*r()]))}
  const mP=meanOf(P),mA3=meanOf(As),d3=sub(mP,mA3),SigP4=covOf(P),SigA4=covOf(As),T4=P.slice(0,24).concat(DA.slice(0,10),DB.slice(0,10));const xs=P.map(p=>p[0]).sort((a,b)=>a-b);const tau=quantile(xs,.25);
  const SW3=mscl(madd(covOf(P),covOf(As)),.5),w3=mv(inv(SW3),d3),wd=dot(w3,d3);
  const coords=h=>[dot(w3,h),h[2]];const Q=P.map(coords);const qm=meanOf(Q);const S=covOf(Q),Si=inv(S),Sh=symFn(S,v=>Math.sqrt(Math.max(v,0)));
  const free=unit([-w3[1],w3[0],0]);
  const place=(base,q)=>{const h=add(base,scl(d3,(q[0]-dot(w3,base))/wd));h[2]=q[1];return h};
  let mode='region',rad=2.5;
  const edit=h=>{if(mode==='none')return{h,on:false};if(mode==='cap')return h[0]<tau?{h:[tau,h[1],h[2]],on:true}:{h,on:false};
    const q=sub(coords(h),qm);const m=Math.sqrt(dot(q,mv(Si,q)));if(m<=rad)return{h,on:false};return{h:place(h,add(qm,scl(q,rad/m))),on:true}};
  const pl=new Plot3D($('#c4'),draw,{yaw:-.38,pitch:.55,span:5});
  const ALL4=P.concat(As,DA,DB);
  function draw(){if(!pl.ready())return;loadCols();pl.clear();pl.fit(ALL4);pl.axes3([-4.6,-3.6,-1.6],1.2,['persona','content','identity']);
    if(mode==='cap'){const y0=-4,y1=4,z0=-1.6,z1=3.2;pl.poly([[tau,y0,z0],[tau,y1,z0],[tau,y1,z1],[tau,y0,z1]],A(COL.navy,.7),1.2,{close:true,fill:A(COL.navy,.07)})}
    if(mode==='region'){const ringAt=(s,t)=>place(add(mP,scl(free,s)),add(qm,mv(Sh,[rad*Math.cos(t),rad*Math.sin(t)])));
      for(const s of[-3.4,-1.7,0,1.7,3.4]){const ring=[];for(let i=0;i<=60;i++)ring.push(ringAt(s,i/60*2*Math.PI));pl.poly(ring,A(COL.navy,.55),1.1)}
      for(const t of[0,Math.PI/2,Math.PI,3*Math.PI/2])pl.poly([ringAt(-3.4,t),ringAt(3.4,t)],A(COL.navy,.4),1);
      pl.label(12,18,'navy cylinder: the region, unbounded along content',COL.navy)}
    if(mode==='cap')pl.label(12,18,'navy plane: the cap floor on the raw persona axis',COL.navy);
    pl.blob(mA3,SigA4,2,A(COL.assist,.8),{fill:A(COL.assist,.08)});pl.blob(mP,SigP4,2,A(COL.persona,.85),{fill:A(COL.persona,.11)});
    const E4=T4.map(edit),pts=[];T4.forEach((h,i)=>{if(E4[i].on)pts.push({p:h,r:2.6,stroke:A(COL.muted,.8),lw:1})});E4.forEach(e=>pts.push({p:e.h,r:3.1,fill:A(COL.ink,.88)}));pl.points(pts);
    T4.forEach((h,i)=>{if(E4[i].on)pl.arrow(h,E4[i].h,A(COL.edit,.8),1.4,6)});
    let cA=0,cB=0,fp=0,dd=0;DA.forEach(h=>{if(edit(h).on)cA++});DB.forEach(h=>{if(edit(h).on)cB++});
    P.forEach(h=>{const e=edit(h);if(e.on){fp++;dd+=dot(sub(e.h,h),sub(e.h,h))}});const spread=Math.sqrt(avg(P.map(h=>dot(sub(h,mP),sub(h,mP)))));
    pl.text(add(muA,[0,0,-1.2]),'assistant (the default)',COL.assist);pl.text(add(mP,[0,0,-1.25]),'persona',COL.persona);
    reads('#m4',[[pct(cA/DA.length),'tokens slid toward the assistant, edited'],[pct(cB/DB.length),'tokens slipped off along identity, edited'],[pct(fp/P.length),'in-character tokens edited'],[f2(Math.sqrt(dd/P.length)/spread),'distortion (in character)']])}
  seg('#s4',[['none','No steering'],['cap','Cap'],['region','Region']],mode,v=>{mode=v;draw()});
  slider('r4r',v=>(v/100).toFixed(2),v=>{rad=v/100;draw()});
  $('#b4v').addEventListener('click',()=>{pl.yaw=Math.atan2(free[0],free[1]);pl.pitch=0;draw()});
})();
