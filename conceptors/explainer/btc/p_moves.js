/* ================= shared fits for the toy world ================= */
// Ridge logistic regression by Newton steps. Y = 1 for the class the classifier should flag.
function logistic(X,Y,lam=1){let th=[0,0,0];
  for(let it=0;it<30;it++){const g=[lam*th[0],lam*th[1],0];let H=[[lam,0,0],[0,lam,0],[0,0,1e-6]];
    X.forEach((x,i)=>{const xt=[x[0],x[1],1];const p=sigm(dot(th,xt));const e=p-Y[i],w=p*(1-p);
      for(let a=0;a<3;a++){g[a]+=e*xt[a];for(let b=0;b<3;b++)H[a][b]+=w*xt[a]*xt[b]}});
    th=sub(th,mv(inv(H),g))}
  return{w:[th[0],th[1]],b:th[2],p:h=>sigm(th[0]*h[0]+th[1]*h[1]+th[2])}}
const LOG_A=logistic(W2.P.concat(W2.A),W2.P.map(()=>0).concat(W2.A.map(()=>1)));   // flags assistant-like tokens
const FIS_W=mv(inv(W2.SigW),W2.d);

/* ================= how to move a token: the gallery ================= */
const MOVES=(function(){const d=W2.d,vh=W2.vhat,wFu=unit(FIS_W);
  const projP=W2.P.map(h=>dot(h,vh)).sort((a,b)=>a-b),fisP=W2.P.map(h=>dot(h,wFu)).sort((a,b)=>a-b);
  const SaH=symFn(W2.SigA,v=>Math.sqrt(Math.max(v,0))),SaHi=inv(SaH);
  const Amap=mm(mm(SaHi,symFn(mm(mm(SaH,W2.SigP),SaH),v=>Math.sqrt(Math.max(v,0)))),SaHi);
  const T=h=>add(W2.mP,mv(Amap,sub(h,W2.mA)));
  const target=unit(sub(W2.mP,ZERO));const I2=eye(2);let Ck=null,kC=-1;const getC=k=>{if(k!==kC){Ck=conceptor(W2.SigP,k);kC=k}return Ck};
  const pctP={label:'Floor: persona percentile',min:2,max:80,val:25,fmt:v=>v+'th'};
  const beta={label:'Strength β',min:0,max:100,val:100,fmt:v=>(v/100).toFixed(2)};
  const tpar={label:'Strength t',min:0,max:100,val:100,fmt:v=>(v/100).toFixed(2)};
  return[
    {id:'none',label:'None',param:null,eq:'h′ = h',desc:'No steering: the drifted tokens stay where they are.',src:'',f:h=>({h,on:false})},
    {id:'add',label:'Add',one:'v',param:{label:'Strength c',min:0,max:150,val:100,fmt:v=>(v/100).toFixed(2)},eq:'h′ = h + c·d',
      desc:'Adds the same vector to every token. The whole cloud shifts and keeps its shape, and tokens already in character get pushed past the persona.',
      src:`${cite('actadd','ActAdd')} (Turner et al. 2023), ${cite('caa','CAA')} (Panickssery et al. 2024), ${cite('pv','persona vectors')} (Chen et al. 2025). Our steering vector.`,
      f:(h,p)=>({h:add(h,scl(d,p/100)),on:p>0})},
    {id:'cap',label:'Clamp',one:'v',param:pctP,eq:'h′ = h + v̂ · max(0, τ − v̂·h)',
      desc:'Lifts only the tokens below a floor on one axis, to the floor. Everything above the floor is untouched.',
      src:`${cite('axis','Assistant Axis')} capping (Lu et al. 2026). Our cap.`,
      f:(h,p)=>{const t=quantile(projP,p/100),s=dot(h,vh);return s<t?{h:add(h,scl(vh,t-s)),on:true}:{h,on:false}}},
    {id:'band',label:'Band',one:'w',param:{label:'Band floor: persona percentile',min:2,max:60,val:25,fmt:v=>v+'th'},eq:'s = ŵ·h,   h′ = h + ŵ · (clip(s, τ_lo, τ_hi) − s)',
      desc:'Keeps a probe’s score inside an allowed band with the smallest possible move, so a token outside lands on the band’s edge. Here the probe is the Fisher direction and the band runs to the 99th percentile, so the chart below uses that axis.',
      src:`${cite('liseco','LiSeCo')} (Cheng & Amo Alonso 2024), which proves the probe score lands in the band at that layer.`,
      f:(h,p)=>{const lo=quantile(fisP,p/100),hi=quantile(fisP,.99),s=dot(h,wFu),c=clamp(s,lo,hi);return c!==s?{h:add(h,scl(wFu,c-s)),on:true}:{h,on:false}}},
    {id:'pin',label:'Pin',one:'v',param:beta,eq:'h′ = h − β · v̂ v̂ᵀ (h − μ_p)',
      desc:'Sets every token’s persona coordinate to the persona mean’s (the dashed teal line), from above and below. The cap is the one-sided version: it only lifts tokens below its floor.',
      src:'Shown for comparison; a two-sided clamp.',
      f:(h,p)=>{const s=dot(sub(h,W2.mP),vh);return{h:sub(h,scl(vh,p/100*s)),on:p>0}}},
    {id:'erase',label:'Erase',one:'v',param:beta,eq:'h′ = h − β · v̂ v̂ᵀ h        (h measured from the zero vector)',
      desc:'Sets the persona coordinate to zero, measured from the zero vector, so every token lands on the dashed navy line. Persona and assistant tokens then look the same along the axis: the move deletes the direction, so it can’t hold a persona along it. At strength β it multiplies the coordinate by 1 − β, which makes it Scale with the factor 1 − β.',
      src:`Directional ablation (${cite('arditi','Arditi et al. 2024')}). ${cite('leace','LEACE')} (Belrose et al. 2023) erases around the mean instead of zero, in closed form, so that no linear classifier can recover the concept.`,
      f:(h,p)=>{const s=dot(sub(h,ZERO),vh);return{h:sub(h,scl(vh,p/100*s)),on:p>0}}},
    {id:'scale',label:'Scale',one:'v',param:{label:'Scale β',min:0,max:300,val:200,fmt:v=>(v/100).toFixed(2)},eq:'h′ = h + (β − 1) · P_persona h        P_persona = v̂ v̂ᵀ here',
      desc:'Multiplies the persona coordinate by β, measured from the zero vector. β = 1 does nothing and β = 0 is Erase. With β above 1, tokens move away from the dashed navy line: tokens with some persona get more, and tokens on the assistant side get pushed further toward the assistant, so it only helps when the persona is already there.',
      src:`${cite('pass','PaSS')} (Malik & ElSherief 2026), with the persona subspace found by ${cite('inlp','INLP')} (Ravfogel et al. 2020).`,
      f:(h,p)=>{const s=dot(sub(h,ZERO),vh);return{h:add(h,scl(vh,(p/100-1)*s)),on:p!==100}}},
    {id:'shrink',label:'Shrink',param:{label:'Aperture k (× median)',min:0,max:200,val:100,fmt:v=>{const k=Math.pow(10,v/100);return k<10?k.toFixed(1):k.toFixed(0)}},eq:'h′ = h − (I − C)(h − μ_p)        C = R (R + aI)⁻¹,  a = k × median eigenvalue',
      desc:'Pulls each token toward the persona mean, by a/(λ + a) in each direction: hard where persona tokens barely vary, weakly where they vary a lot. The teal ellipse is what C keeps.',
      src:`Conceptors (${cite('jaeger','Jaeger 2014')}; ${cite('postmus','Postmus & Abreu 2024')}; ${cite('triant','Triantafyllopoulos et al. 2026')}). Our conceptor.`,
      f:(h,p)=>{const C=getC(Math.pow(10,p/100));return{h:sub(h,mv(msub(I2,C),sub(h,W2.mP))),on:true}}},
    {id:'mean',label:'Mean-shift',param:beta,eq:'h′ = h + β · (μ_p − h)',
      desc:'Pulls every token toward one point, so the cloud collapses. It is the conceptor’s limit at a very large aperture, and our control for it.',
      src:`Our matrix-removed control (after ${cite('razdan','Razdan & Balamurugan 2026')}).`,
      f:(h,p)=>({h:add(h,scl(sub(W2.mP,h),p/100)),on:p>0})},
    {id:'ot',label:'Transport',param:tpar,eq:'h′ = (1 − t)·h + t·(μ_p + A (h − μ_a))',
      desc:'Maps the assistant distribution onto the persona’s: it moves the mean and reshapes the spread. In-character tokens get moved too, unless a gate stops it.',
      src:`${cite('mimic','MiMiC')}, Gaussian optimal transport (Singh et al. 2024).`,
      f:(h,p)=>{const t=p/100;return{h:add(scl(h,1-t),scl(T(h),t)),on:t>0}}},
    {id:'rotate',label:'Rotate',param:tpar,eq:'h′ = |h| · slerp(ĥ, p̂; t)        p̂ = persona direction from the zero vector',
      desc:'Turns each token toward the persona direction by a fraction t of the angle and keeps its length. In 2D the fixed-plane and great-circle versions are the same turn. This toy flatters rotation: both clouds sit at about the same distance from the zero vector, so a turned token lands inside the persona cloud.',
      src:`${cite('spherical','Spherical steering')} (You et al. 2026), ${cite('angular','angular steering')} (Vu & Nguyen 2025), ${cite('hpr','Householder pseudo-rotation')} (Pham & Nguyen 2024).`,
      f:(h,p)=>{const t=p/100;const v=sub(h,ZERO),r=nrm(v);const a=Math.atan2(v[1],v[0]),b=Math.atan2(target[1],target[0]);let da=b-a;while(da>Math.PI)da-=2*Math.PI;while(da<-Math.PI)da+=2*Math.PI;
        const a2=a+t*da;return{h:add(ZERO,[r*Math.cos(a2),r*Math.sin(a2)]),on:t>0}}},
  ].map(o=>Object.assign(o,{getC,projP,fisP,wFu}))})();

/* ================= the same tokens in 3D ================= */
// Every 2D token keeps its x and y and gains a third coordinate z, a stand-in for the thousands of other directions:
// both clouds centre on z = 0, and the assistant spreads along it about twice as much as the persona.
const W3=(function(){const r=mulberry32(97);const P=W2.P.map(h=>[h[0],h[1],.65*randn(r)]),Ap=W2.A.map(h=>[h[0],h[1],1.3*randn(r)]),Dr=W2.Dr.map(h=>[h[0],h[1],.65*randn(r)]);
  const mP=meanOf(P),mA=meanOf(Ap),d=sub(mP,mA),SigP=covOf(P),SigA=covOf(Ap);
  return{P,A:Ap,Dr,mP,mA,d,vhat:unit(d),SigP,SigA,SigW:mscl(madd(SigP,SigA),.5),ZERO:[ZERO[0],ZERO[1],0]}})();
// one function per move, the 3D versions of MOVES; 'plane' rotation keeps z, as angular steering keeps everything off its plane
const MOVES3=(function(){const W=W3,Z3=W.ZERO,vh=W.vhat;const projP=W.P.map(h=>dot(h,vh)).sort((a,b)=>a-b);
  const wu=unit(mv(inv(W.SigW),W.d)),fisP=W.P.map(h=>dot(h,wu)).sort((a,b)=>a-b);
  const sq=M=>symFn(M,v=>Math.sqrt(Math.max(v,0))),SaH=sq(W.SigA),SaHi=inv(SaH),Amap=mm(mm(SaHi,sq(mm(mm(SaH,W.SigP),SaH))),SaHi);
  const T=h=>add(W.mP,mv(Amap,sub(h,W.mA)));const I3=eye(3);let Ck=null,kC=-1;const getC=k=>{if(k!==kC){Ck=conceptor(W.SigP,k);kC=k}return Ck};
  const pdir=unit(sub(W.mP,Z3));const st={rot:'sphere'};
  const F={none:h=>({h,on:false}),
    add:(h,p)=>({h:add(h,scl(W.d,p/100)),on:p>0}),
    cap:(h,p)=>{const t=quantile(projP,p/100),s=dot(h,vh);return s<t?{h:add(h,scl(vh,t-s)),on:true}:{h,on:false}},
    band:(h,p)=>{const lo=quantile(fisP,p/100),hi=quantile(fisP,.99),s=dot(h,wu),c=clamp(s,lo,hi);return c!==s?{h:add(h,scl(wu,c-s)),on:true}:{h,on:false}},
    pin:(h,p)=>{const s=dot(sub(h,W.mP),vh);return{h:sub(h,scl(vh,p/100*s)),on:p>0}},
    erase:(h,p)=>{const s=dot(sub(h,Z3),vh);return{h:sub(h,scl(vh,p/100*s)),on:p>0}},
    scale:(h,p)=>{const s=dot(sub(h,Z3),vh);return{h:add(h,scl(vh,(p/100-1)*s)),on:p!==100}},
    shrink:(h,p)=>{const C=getC(Math.pow(10,p/100));return{h:sub(h,mv(msub(I3,C),sub(h,W.mP))),on:true}},
    mean:(h,p)=>({h:add(h,scl(sub(W.mP,h),p/100)),on:p>0}),
    ot:(h,p)=>{const t=p/100;return{h:add(scl(h,1-t),scl(T(h),t)),on:t>0}},
    rotate:(h,p)=>{const t=p/100,v=sub(h,Z3),r=nrm(v);
      if(st.rot==='sphere')return{h:add(Z3,scl(slerpU(scl(v,1/r),pdir,t),r)),on:t>0};
      const a=Math.atan2(v[1],v[0]),b=Math.atan2(pdir[1],pdir[0]);let da=b-a;while(da>Math.PI)da-=2*Math.PI;while(da<-Math.PI)da+=2*Math.PI;
      const rr=Math.hypot(v[0],v[1]),a2=a+t*da;return{h:[Z3[0]+rr*Math.cos(a2),Z3[1]+rr*Math.sin(a2),h[2]],on:t>0}}};
  return{F,st,projP,fisP,wu,getC}})();

(function(){let op='add';const vals={};MOVES.forEach(o=>{if(o.param)vals[o.id]=o.param.val});
  const pl=new Plot2D($('#c_mv'),BOX2,draw);const rng=$('#r_mvp'),out=$('#o_mvp'),lbl=$('#lbl_mvp'),wrap=rng.closest('label');
  const cur=()=>MOVES.find(o=>o.id===op);
  // 3D twin: same move, same slider value, drawn on W3
  const pl3=new Plot3D($('#c_mv3'),()=>draw3(),{yaw:-.5,pitch:.42,span:5.4});const ALL3=W3.P.concat(W3.A,W3.Dr);
  const plane3=(n,tau,c0,col,label)=>{const o0=add(c0,scl(n,tau-dot(n,c0)));let e1=cross(n,[0,0,1]);if(nrm(e1)<1e-6)e1=[1,0,0];e1=unit(e1);const e2=cross(n,e1);
    const q=(a,b)=>add(o0,add(scl(e1,a),scl(e2,b)));pl3.poly([q(-3.6,-2.6),q(3.6,-2.6),q(3.6,2.6),q(-3.6,2.6)],A(col,.75),1.2,{close:true,fill:A(col,.06)});if(label)pl3.text(q(0,2.85),label,col,'600 11px')};
  function draw3(){if(!pl3.ready())return;loadCols();pl3.clear();const o=cur(),p=vals[o.id];const f=h=>MOVES3.F[o.id](h,p);const Z3=W3.ZERO,vh=W3.vhat;
    pl3.fit(ALL3);pl3.axes3([-5.4,-3.9,-2.6],1.1,['x','y','z']);const cen=scl(add(W3.mP,W3.mA),.5);
    if(o.id==='cap')plane3(vh,quantile(MOVES3.projP,p/100),cen,COL.navy,'floor');
    if(o.id==='band'){plane3(MOVES3.wu,quantile(MOVES3.fisP,p/100),cen,COL.violet,'band floor');plane3(MOVES3.wu,quantile(MOVES3.fisP,.99),cen,COL.violet,null)}
    if(o.id==='pin')plane3(vh,dot(vh,W3.mP),W3.mP,COL.persona,'persona mean’s coordinate');
    if(o.id==='erase'||o.id==='scale')plane3(vh,dot(vh,Z3),cen,COL.navy,'persona coordinate = 0');
    if(o.id==='shrink'){const C=MOVES3.getC(Math.pow(10,p/100));const{vals:ev,vecs}=symEig(C);pl3.ellipsoid(W3.mP,vecs,ev.map(v=>Math.max(v,.02)*2.1),A(COL.persona,.55))}
    if(o.id==='ot'){const{vals:ev,vecs}=symEig(W3.SigP);pl3.ellipsoid(W3.mP,vecs,ev.map(v=>2*Math.sqrt(Math.max(v,0))),A(COL.persona,.45))}
    if(o.id==='rotate'){const R=nrm(sub(W3.mP,Z3));[[1,0,0],[0,1,0],[0,0,1]].forEach((ax,i)=>{const u=[[0,1,0],[0,0,1],[1,0,0]][i],w=cross(ax,u);const ring=[];for(let j=0;j<=72;j++){const a=j/72*2*Math.PI;ring.push(add(Z3,add(scl(u,R*Math.cos(a)),scl(w,R*Math.sin(a)))))}pl3.poly(ring,A(COL.navy,.25),1,{dash:[3,4]})})}
    pl3.poly([sub(Z3,[.2,0,0]),add(Z3,[.2,0,0])],COL.ink,1.6);pl3.poly([sub(Z3,[0,.2,0]),add(Z3,[0,.2,0])],COL.ink,1.6);pl3.poly([sub(Z3,[0,0,.2]),add(Z3,[0,0,.2])],COL.ink,1.6);pl3.text(add(Z3,[0,0,-.4]),'zero vector',COL.muted,'500 11px');
    const eP=W3.P.map(f),eD=W3.Dr.map(f);const pts=[];W3.A.forEach(h=>pts.push({p:h,r:1.9,fill:A(COL.assist,.28)}));
    W3.P.forEach(h=>pts.push({p:h,r:2.3,fill:A(COL.persona,.75)}));W3.Dr.forEach(h=>pts.push({p:h,r:2.3,fill:A(COL.drift,.75)}));
    eD.forEach(e=>{if(e.on)pts.push({p:e.h,r:2.2,stroke:A(COL.drift,.85),lw:1})});eP.forEach(e=>{if(e.on)pts.push({p:e.h,r:2.2,stroke:A(COL.persona,.85),lw:1})});pl3.points(pts);
    eD.forEach((e,i)=>{if(e.on)pl3.arrow(W3.Dr[i],e.h,A(COL.edit,.45),1,4)});eP.forEach((e,i)=>{if(e.on)pl3.arrow(W3.P[i],e.h,A(COL.edit,.35),1,4)});
    pl3.text(add(W3.mA,[0,0,-1.25]),'assistant',COL.assist);pl3.text(add(W3.mP,[0,0,-1.25]),'persona',COL.persona);
    const sdz=a=>{const m=avg(a);return Math.sqrt(avg(a.map(x=>(x-m)**2)))};
    const zb=sdz(W3.Dr.map(h=>h[2])),za=sdz(eD.map(e=>e.h[2])),pb=sdz(W3.P.map(h=>h[2])),pa=sdz(eP.map(e=>e.h[2]));
    const lc=avg(eD.map((e,i)=>nrm(sub(e.h,Z3))/nrm(sub(W3.Dr[i],Z3))-1));
    reads('#m_mv3',[[f2(zb)+' → '+f2(za),'drifted tokens: spread along z'],[f2(pb)+' → '+f2(pa),'in-character tokens: spread along z'],[(lc>=0?'+':'')+(100*lc).toFixed(0)+'%','length change in 3D (drifted, from zero)']])}
  $('#b_mv3').addEventListener('click',()=>{pl3.yaw=0;pl3.pitch=-1.45;draw3()});
  seg('#s_mv3r',[['sphere','Great circle (spherical steering)'],['plane','Fixed plane (angular steering)']],MOVES3.st.rot,v=>{MOVES3.st.rot=v;draw3()});
  function setOp(id){op=id;const o=cur();$('#s_mv3r').hidden=id!=='rotate';$('#d_mv').textContent=o.desc;$('#eq_mv').textContent=o.eq;$('#src_mv').innerHTML=o.src;
    if(o.param){wrap.hidden=false;lbl.textContent=o.param.label;rng.min=o.param.min;rng.max=o.param.max;rng.value=vals[o.id];out.textContent=o.param.fmt(vals[o.id])}else wrap.hidden=true;draw()}
  rng.addEventListener('input',()=>{const o=cur();if(!o.param)return;vals[o.id]=+rng.value;out.textContent=o.param.fmt(+rng.value);draw()});
  $('#k_mvf').addEventListener('change',()=>draw());
  // before/after chart: each token's coordinate along the move's axis, measured from the zero vector.
  // A one-axis move puts every token on one curve (its rule); a move that touches other directions scatters them.
  const rc=$('#c_mvr'),rg=rc.getContext('2d');watch(rc,()=>drawRule());
  function drawRule(){const r=fitCanvas(rc,rg);if(!r)return;loadCols();const o=cur(),p=vals[o.id];const W=r.width,H=r.height,font=FONT();
    const ax=o.one==='w'?unit(FIS_W):W2.vhat;const sOf=h=>dot(sub(h,ZERO),ax);
    const groups=[[W2.A,COL.assist,.3,1.8],[W2.P,COL.persona,.65,2.1],[W2.Dr,COL.drift,.75,2.1]];const pts=[];
    groups.forEach(([G,col,al,rad])=>G.forEach(h=>pts.push([sOf(h),sOf(o.f(h,p).h),A(col,al),rad])));
    // x spans the tokens' coordinates before; y spans before and after, so "no change" and the move both fit
    const span=(v,f=.06)=>{let a=Math.min(...v),b=Math.max(...v);const d=(b-a)*f||1;return[a-d,b+d]};
    const[x0,x1]=span(pts.map(q=>q[0])),[y0,y1]=span(pts.map(q=>q[0]).concat(pts.map(q=>q[1])));
    const pad={l:40,r:12,t:24,b:32};const X=v=>pad.l+(v-x0)/(x1-x0)*(W-pad.l-pad.r),Y=v=>H-pad.b-(v-y0)/(y1-y0)*(H-pad.t-pad.b);
    const step=s=>{const raw=s/6,e=Math.pow(10,Math.floor(Math.log10(raw))),m=raw/e;return(m<1.5?1:m<3.5?2:m<7.5?5:10)*e};
    rg.font=`500 11px ${font}`;rg.lineWidth=1;const sx=step(x1-x0),sy=step(y1-y0);
    for(let t=Math.ceil(x0/sx)*sx;t<=x1;t+=sx){rg.strokeStyle=COL.grid;rg.beginPath();rg.moveTo(X(t),pad.t);rg.lineTo(X(t),H-pad.b);rg.stroke();rg.fillStyle=COL.muted;rg.textAlign='center';rg.textBaseline='top';rg.fillText(String(+t.toFixed(2)),X(t),H-pad.b+4)}
    for(let t=Math.ceil(y0/sy)*sy;t<=y1;t+=sy){rg.strokeStyle=COL.grid;rg.beginPath();rg.moveTo(pad.l,Y(t));rg.lineTo(W-pad.r,Y(t));rg.stroke();rg.fillStyle=COL.muted;rg.textAlign='right';rg.textBaseline='middle';rg.fillText(String(+t.toFixed(2)),pad.l-6,Y(t))}
    rg.save();rg.beginPath();rg.rect(pad.l,pad.t,W-pad.l-pad.r,H-pad.t-pad.b);rg.clip();
    rg.save();rg.strokeStyle=A(COL.muted,.8);rg.setLineDash([4,4]);rg.beginPath();rg.moveTo(X(x0),Y(x0));rg.lineTo(X(x1),Y(x1));rg.stroke();rg.restore();
    if(x0<0&&x1>0){rg.save();rg.strokeStyle=A(COL.navy,.6);rg.setLineDash([2,3]);rg.beginPath();rg.moveTo(X(0),pad.t);rg.lineTo(X(0),H-pad.b);rg.stroke();rg.restore()}
    pts.forEach(q=>{rg.beginPath();rg.arc(X(q[0]),Y(q[1]),q[3],0,2*Math.PI);rg.fillStyle=q[2];rg.fill()});
    if(o.one){rg.strokeStyle=COL.edit;rg.lineWidth=2.2;rg.beginPath();for(let i=0;i<=160;i++){const s=x0+(x1-x0)*i/160;const s2=sOf(o.f(add(ZERO,scl(ax,s)),p).h);i?rg.lineTo(X(s),Y(s2)):rg.moveTo(X(s),Y(s2))}rg.stroke()}
    rg.restore();
    rg.fillStyle=COL.ink;rg.font=`600 12px ${font}`;rg.textAlign='left';rg.textBaseline='top';
    rg.fillText(o.one?`${o.label}: a one-axis rule`:`${o.label}: moves other coordinates too`,pad.l,5);
    rg.font=`500 11px ${font}`;rg.fillStyle=COL.muted;rg.textAlign='center';rg.textBaseline='bottom';
    rg.fillText((o.one==='w'?'Fisher score':'persona coordinate')+' before',pad.l+(W-pad.l-pad.r)/2,H-1);
    rg.save();rg.translate(12,pad.t+(H-pad.t-pad.b)/2);rg.rotate(-Math.PI/2);rg.textAlign='center';rg.textBaseline='middle';rg.fillText('after',0,0);rg.restore()}
  function draw(){drawRule();draw3();if(!pl.ready())return;loadCols();pl.clear();const o=cur();const p=vals[o.id];const edit=h=>o.f(h,p);
    if($('#k_mvf').checked){const st=.8,G=[];for(let x=BOX2.x0+.4;x<=BOX2.x1;x+=st)for(let y=BOX2.y0+.4;y<=BOX2.y1;y+=st)G.push([x,y]);
      const D=G.map(g=>sub(edit(g).h,g));const L=D.map(nrm).filter(x=>x>1e-6).sort((a,b)=>a-b);
      if(L.length){const q=quantile(L,.9),s=Math.min(1,.85*st/q);let clipped=0;
        G.forEach((g,i)=>{const n=nrm(D[i]);if(n<=1e-6)return;let len=s*n;if(len>1.1*st){len=1.1*st;clipped++}pl.arrow(g,add(g,scl(D[i],len/n)),A(COL.navy,.45),1,4)});
        pl.text([BOX2.x1-.15,BOX2.y1-.3],(s<1?`grid arrows at ${(100*s).toFixed(0)}% length`:'grid arrows at full length')+(clipped?', longest ones clipped':''),COL.muted,{align:'right'})}}
    if(o.id==='cap'){const t=quantile(o.projP,p/100);pl.halfplane(W2.vhat,t,A(COL.navy,.06));pl.boundary(W2.vhat,t,COL.navy,1.8,null,'floor')}
    if(o.id==='band'){const lo=quantile(o.fisP,p/100),hi=quantile(o.fisP,.99);pl.boundary(o.wFu,lo,COL.violet,1.8,null,'band floor');pl.boundary(o.wFu,hi,COL.violet,1.2,[5,4],'band ceiling')}
    if(o.id==='shrink'){const C=o.getC(Math.pow(10,p/100));const{vals:ev,vecs}=symEig(C);pl.ellipse(W2.mP,vecs,ev.map(v=>Math.max(v,.02)*2.1),COL.persona,{w:2,fill:A(COL.persona,.07)})}
    if(o.id==='rotate'){const R=nrm(sub(W2.mP,ZERO));pl.ellipse(ZERO,[[1,0],[0,1]],[R,R],A(COL.navy,.35),{w:1,dash:[3,4]})}
    if(o.id==='erase'||o.id==='scale')pl.boundary(W2.vhat,dot(W2.vhat,ZERO),COL.navy,1.6,[6,4],'persona coordinate = 0');
    if(o.id==='pin')pl.boundary(W2.vhat,dot(W2.vhat,W2.mP),COL.persona,1.6,[6,4],'persona mean’s coordinate');
    pl.line(sub(ZERO,[.18,0]),add(ZERO,[.18,0]),COL.ink,1.6);pl.line(sub(ZERO,[0,.18]),add(ZERO,[0,.18]),COL.ink,1.6);pl.text(add(ZERO,[.28,.02]),'zero vector',COL.muted);
    drawAxis(pl);drawClouds(pl);const m=metrics(edit);drawEdits(pl,m,.5);
    m.De.forEach(e=>{if(e.on)pl.dot(e.h,2.3,null,A(COL.drift,.8),1)});m.Pe.forEach(e=>{if(e.on)pl.dot(e.h,2.3,null,A(COL.persona,.8),1)});
    reads('#m_mv',[[pct(m.restore),'restore (drifted)'],[f2(m.dist),'distortion (in character)'],[(m.normCh>=0?'+':'')+(100*m.normCh).toFixed(0)+'%','length change (drifted, from zero)'],[pct(m.landed),'drifted tokens landing in the persona’s 2σ ellipse']])}
  seg('#s_mv',MOVES.map(o=>[o.id,o.label]),op,v=>setOp(v));
  const tb=$('#t_ops');tb.innerHTML='<thead><tr><th>Move</th><th>Formula</th><th>What it does</th><th>Source</th></tr></thead><tbody>'+
    MOVES.filter(o=>o.id!=='none').map(o=>`<tr><td><b>${o.label}</b></td><td><code>${o.eq.replace(/</g,'&lt;')}</code></td><td>${o.desc}</td><td>${o.src}</td></tr>`).join('')+'</tbody>';
  setOp(op);
})();

/* ================= when to act: gates ================= */
(function(){const d=W2.d,vh=W2.vhat;
  const rawP=W2.P.map(h=>dot(h,vh)).sort((a,b)=>a-b),fisP=W2.P.map(h=>dot(h,FIS_W)).sort((a,b)=>a-b);
  // GAPS: a neuron basis that is not aligned with d, per-neuron AUROC and 1D Gaussians for each class
  const ang=35*Math.PI/180,NB=[[Math.cos(ang),Math.sin(ang)],[-Math.sin(ang),Math.cos(ang)]];
  const nstat=NB.map(e=>{const a=W2.A.map(h=>dot(h,e)),p=W2.P.map(h=>dot(h,e));const ma=avg(a),mp=avg(p);
    const va=avg(a.map(x=>(x-ma)**2)),vp=avg(p.map(x=>(x-mp)**2));return{e,auc:auroc(p,a),ma,mp,va,vp}});
  const postA=(s,x)=>{const la=-.5*(x-s.ma)**2/s.va-.5*Math.log(s.va),lp=-.5*(x-s.mp)**2/s.vp-.5*Math.log(s.vp);return sigm(la-lp)};
  let gate='fisher',c=+$('#r_gtc').value/100;const tv={raw:25,fisher:25,dsas:100,gaps:30};
  const G=[
    {id:'always',label:'Always on',param:null,eq:'g(h) = 1',desc:'Every token gets the full move. This is the steering vector.',src:`${cite('actadd','ActAdd')}, ${cite('caa','CAA')}.`},
    {id:'raw',label:'Raw floor',param:{label:'Floor: persona percentile',min:2,max:60,fmt:v=>v+'th'},eq:'g(h) = 1  if  v̂·h < τ',desc:'The cap’s trigger: a token below a percentile of in-character tokens on the raw axis gets the move.',src:`${cite('axis','Assistant Axis')} capping (Lu et al. 2026).`},
    {id:'fisher',label:'Fisher floor',param:{label:'Floor: persona percentile',min:2,max:60,fmt:v=>v+'th'},eq:'g(h) = 1  if  w·h < τ_w,   w = Σ_W⁻¹ d',desc:'The same floor on the Fisher score, which discounts content. This is fisher_cap’s trigger.',src:'Our proposal (fisher_cap, Appendix B).'},
    {id:'near',label:'Nearer the assistant',param:null,eq:'g(h) = 1  if  |h − μ_a| < |h − μ_p|',desc:'Moves only tokens closer to the assistant mean than to the persona mean. MiMiC applies its transport map with this rule.',src:`${cite('mimic','MiMiC')} (Singh et al. 2024).`},
    {id:'dsas',label:'Learned soft gate',param:{label:'Gate temperature',min:20,max:300,fmt:v=>(v/100).toFixed(2)},eq:'g(h) = σ((wᵀh + b) / T),   h′ = (1 − g)·h + g·(h + c·d)',desc:'A logistic classifier, trained to tell assistant tokens from persona tokens, gives each token a weight between 0 and 1, and the edit blends the original and steered states by it.',src:`${cite('dsas','DSAS')} (Ferrando et al. 2025), which trains a per-layer logistic gate and wraps any steering map.`},
    {id:'gaps',label:'Per-neuron masks',param:{label:'Separability threshold |AUROC − ½|',min:0,max:48,fmt:v=>(v/100).toFixed(2)},eq:'h′ = h + c · (m_sep ⊙ m_post(h)) ⊙ d',desc:'Keeps neurons whose values separate the two classes, and per token steers only the neurons whose current value looks like the assistant’s under a per-neuron Gaussian. The gray dashed lines are the two neuron axes.',src:`${cite('gaps','GAPS')} (Fereidouni et al. 2026), which adds these masks to a ${cite('cast','CAST')} or DSAS gate.`},
  ];
  const cur=()=>G.find(o=>o.id===gate);
  function gateOf(h){const o=cur();if(o.id==='always')return{g:1,v:d};
    if(o.id==='raw'){const t=quantile(rawP,tv.raw/100);return{g:dot(h,vh)<t?1:0,v:d}}
    if(o.id==='fisher'){const t=quantile(fisP,tv.fisher/100);return{g:dot(h,FIS_W)<t?1:0,v:d}}
    if(o.id==='near'){return{g:nrm(sub(h,W2.mA))<nrm(sub(h,W2.mP))?1:0,v:d}}
    if(o.id==='dsas'){const T=tv.dsas/100;const z=LOG_A.w[0]*h[0]+LOG_A.w[1]*h[1]+LOG_A.b;return{g:sigm(z/T),v:d}}
    const thr=tv.gaps/100;let v=[0,0],k=0;nstat.forEach(s=>{if(Math.abs(s.auc-.5)>thr&&postA(s,dot(h,s.e))>.5){v=add(v,scl(s.e,dot(d,s.e)));k++}});
    return{g:nrm(v)/nrm(d),v:nrm(v)>0?scl(unit(v),nrm(d)):d}}
  const pl=new Plot2D($('#c_gt'),BOX2,draw);const rng=$('#r_gtt'),out=$('#o_gtt'),lbl=$('#lbl_gtt'),wrap=$('#w_gtt');
  function setGate(id){gate=id;const o=cur();$('#d_gt').textContent=o.desc;$('#eq_gt').textContent=o.eq;$('#src_gt').innerHTML=o.src;
    if(o.param){wrap.hidden=false;lbl.textContent=o.param.label;rng.min=o.param.min;rng.max=o.param.max;rng.value=tv[o.id];out.textContent=o.param.fmt(tv[o.id])}else wrap.hidden=true;draw()}
  rng.addEventListener('input',()=>{const o=cur();if(!o.param)return;tv[o.id]=+rng.value;out.textContent=o.param.fmt(+rng.value);draw()});
  slider('r_gtc',v=>(v/100).toFixed(2),v=>{c=v/100;draw()});
  function edit(h){const{g,v}=gateOf(h);if(gate==='gaps'){const vv=scl(v,c*g);return{h:add(h,vv),on:g>.02,g}}return{h:add(h,scl(v,c*g)),on:g>.02,g}}
  function draw(){if(!pl.ready())return;loadCols();pl.clear();const o=cur();
    if(o.id==='raw'){const t=quantile(rawP,tv.raw/100);pl.halfplane(vh,t,A(COL.navy,.06));pl.boundary(vh,t,COL.navy,1.8,null,'raw floor')}
    if(o.id==='fisher'){const t=quantile(fisP,tv.fisher/100);pl.halfplane(FIS_W,t,A(COL.violet,.06));pl.boundary(FIS_W,t,COL.violet,1.8,null,'Fisher floor')}
    if(o.id==='near'){const n=sub(W2.mP,W2.mA),t=dot(n,scl(add(W2.mP,W2.mA),.5));pl.halfplane(n,t,A(COL.navy,.06));pl.boundary(n,t,COL.navy,1.6,[5,4],'equal distance')}
    if(o.id==='dsas'){const n=scl(LOG_A.w,-1);pl.boundary(n,LOG_A.b,COL.navy,1.6,[5,4],'g = ½')}
    if(o.id==='gaps'){nstat.forEach((s,i)=>{pl.line(scl(s.e,-7),scl(s.e,7),A(COL.muted,.6),1,[4,4]);pl.text(scl(s.e,i?2.9:4.6),`neuron ${i+1}`+(Math.abs(s.auc-.5)>tv.gaps/100?'':' (masked)'),COL.muted)})}
    drawAxis(pl);
    W2.A.forEach(p=>pl.dot(p,2.2,A(COL.assist,.22)));
    const eP=W2.P.map(edit),eD=W2.Dr.map(edit);
    W2.P.forEach((p,i)=>pl.dot(p,2.6,A(COL.persona,.8),eP[i].g>.02?A(COL.ink,.3+.7*eP[i].g):null,1.1));
    W2.Dr.forEach((p,i)=>pl.dot(p,2.7,A(COL.drift,.15+.8*eD[i].g),COL.drift,1));
    eD.forEach((e,i)=>{if(e.on)pl.arrow(W2.Dr[i],e.h,A(COL.edit,.5),1,5)});eP.forEach((e,i)=>{if(e.on)pl.arrow(W2.P[i],e.h,A(COL.edit,.4),1,5)});
    const m=metrics(h=>edit(h));
    reads('#m_gt',[[pct(avg(eD.map(e=>e.g))),'mean gate on drifted tokens'],[pct(avg(eP.map(e=>e.g))),'mean gate on in-character tokens'],[pct(m.restore),'restore (drifted)'],[f2(m.dist),'distortion (in character)']])}
  seg('#s_gt',G.map(o=>[o.id,o.label]),gate,v=>setGate(v));
  setGate(gate);
})();

/* ================= which directions to fit ================= */
(function(){// estimators are fit on a larger draw from the same toy world, so sampling noise doesn't steer the picture
  const WB=makeWorld2D(11,1200),d=WB.d,mG=scl(add(W2.mP,W2.mA),.5),mGB=scl(add(WB.mP,WB.mA),.5);let est='mean',rr=1;
  const orient=v=>dot(v,d)<0?scl(v,-1):v;
  const LOG_P=logistic(WB.P.concat(WB.A),WB.P.map(()=>1).concat(WB.A.map(()=>0)));
  const FIS_B=mv(inv(WB.SigW),d),PAIR_B=symEig(momentAbout(WB.D,[0,0])).vecs[0],PCA_B=symEig(covOf(WB.P.concat(WB.A))).vecs[0];
  const SpB=momentAbout(WB.P,mGB),SaB=momentAbout(WB.A,mGB);
  const ccs=r=>{const{vecs}=symEig(msub(SpB,mscl(SaB,r)));const u=vecs[0];const v=scl(u,dot(u,sub(WB.mP,mGB)));return orient(nrm(v)>1e-9?v:u)};
  const E=[
    {id:'mean',label:'Mean difference',col:'navy',v:()=>d,eq:'d = μ_p − μ_a',desc:'The arrow from the assistant’s average token to the persona’s average token. Our vector, cap and region use it, and so do persona vectors. Any content that differs between the two clouds tilts it.',src:`${cite('caa','CAA')} (Panickssery et al. 2024); ${cite('pv','persona vectors')} (Chen et al. 2025).`},
    {id:'fisher',label:'Fisher',col:'violet',v:()=>FIS_B,eq:'w = Σ_W⁻¹ d',desc:'The mean difference, corrected for how much tokens spread in each direction. Directions where tokens vary a lot anyway, like content, count for less, so the arrow turns away from content. If both clouds are Gaussian with the same spread, this is the best possible line for telling them apart.',src:'Fisher’s linear discriminant, a textbook method. Our fisher_cap uses it as its trigger.'},
    {id:'probe',label:'Probe (INLP step)',col:'ochre',v:()=>LOG_P.w,eq:'w = logistic regression weights (persona vs assistant)',desc:'Train a classifier to tell persona tokens from assistant tokens and use its weights as the direction. INLP repeats this: remove the direction, train again, and stop when no classifier works. PaSS treats all the removed directions as the persona subspace.',src:`${cite('inlp','INLP')} (Ravfogel et al. 2020); ${cite('pass','PaSS')} (Malik & ElSherief 2026).`},
    {id:'paired',label:'Paired differences',col:'edit',v:()=>PAIR_B,eq:'top eigenvector of  E[(p_i − a_i)(p_i − a_i)ᵀ]',desc:'Pair each persona token with an assistant token about the same content and subtract. The content cancels in each difference, so what’s left is the persona. Our diffpull is fit on these differences.',src:'Our diffpull (per-question differences).'},
    {id:'ccs',label:'Covariance contrast',col:'persona',v:()=>ccs(rr),eq:'leading eigenvector u of  S_p − r·S_a   (scatter about the grand mean)\nv = u uᵀ (μ_p − μ_G)',desc:'Compare how the two clouds spread instead of where their centres sit: keep the directions where persona tokens spread more than assistant tokens do. With r near 1, the content both clouds share cancels and the persona’s style direction wins. In real models its strength is removing many nuisance classes at once, which 2D can’t show.',src:`${cite('ccs','Contrastive covariance subspaces')} (Xenos et al. 2026).`},
    {id:'pca',label:'PCA, no labels',col:'muted',v:()=>PCA_B,eq:'top eigenvector of the pooled covariance',desc:'The direction along which all tokens together vary most, found without knowing which token is which. Here it still finds the persona offset, but it leans toward content.',src:'Baseline, no labels.'},
  ];
  const pl=new Plot2D($('#c_dir'),BOX2,draw);const wrap=$('#w_dirr');
  slider('r_dirr',v=>(v/100).toFixed(2),v=>{rr=v/100;draw()});
  function setEst(id){est=id;const o=E.find(x=>x.id===id);$('#d_dir').textContent=o.desc;$('#eq_dir').textContent=o.eq;$('#src_dir').innerHTML=o.src;wrap.hidden=id!=='ccs';draw()}
  function draw(){if(!pl.ready())return;loadCols();pl.clear();
    pl.line(sub(mG,scl(W2.u1,6)),add(mG,scl(W2.u1,6)),A(COL.muted,.7),1.2,[2,4]);pl.text(add(mG,scl(W2.u1,3.6)),'content',COL.muted);
    pl.line(sub(W2.mP,scl(W2.w1,1.6)),add(W2.mP,scl(W2.w1,1.6)),A(COL.persona,.7),1.2,[2,4]);pl.text(add(W2.mP,scl(W2.w1,1.95)),'persona style',COL.persona);
    drawClouds(pl,{drift:false,af:.3});pl.dot(W2.mA,4.5,COL.assist,COL.paper,1.5);pl.dot(W2.mP,4.5,COL.persona,COL.paper,1.5);
    E.forEach(o=>{const v=unit(orient(o.v()));const on=o.id===est;pl.arrow(mG,add(mG,scl(v,on?2.6:2.1)),on?COL[o.col]:A(COL[o.col],.35),on?2.8:1.4,on?10:6);
      if(on)pl.text(add(mG,scl(v,2.95)),o.label,COL[o.col],{align:'center',font:'600 12px'})});
    const o=E.find(x=>x.id===est);const v=unit(orient(o.v()));
    const pp=W2.P.map(h=>dot(h,v)),pa=W2.A.map(h=>dot(h,v));const ang=Math.acos(clamp(dot(v,unit(d)),-1,1))*180/Math.PI;
    const totP=W2.SigP[0][0]+W2.SigP[1][1];const along=dot(v,mv(W2.SigP,v))/totP;
    reads('#m_dir',[[ang.toFixed(0)+'°','angle to the mean difference'],[f2(Math.abs(dot(v,W2.u1))),'|cos| with the content direction'],[f2(auroc(pa,pp)),'AUROC, persona vs assistant'],[pct(along),'share of persona-token spread along it']])}
  seg('#s_dir',E.map(o=>[o.id,o.label]),est,v=>setEst(v));
  setEst(est);
})();
