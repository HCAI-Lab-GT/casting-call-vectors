/* ================= moving distributions: transport ================= */
(function(){let mode='ot',t=1,sharp=1;const SaH=symFn(W2.SigA,v=>Math.sqrt(Math.max(v,0))),SaHi=inv(SaH);
  const Amap=mm(mm(SaHi,symFn(mm(mm(SaH,W2.SigP),SaH),v=>Math.sqrt(Math.max(v,0)))),SaHi);
  const T=h=>add(W2.mP,mv(Amap,sub(h,W2.mA)));
  const SPi=inv(W2.SigP),SAi=inv(W2.SigA);const ldP=Math.log(W2.SigP[0][0]*W2.SigP[1][1]-W2.SigP[0][1]**2),ldA=Math.log(W2.SigA[0][0]*W2.SigA[1][1]-W2.SigA[0][1]**2);
  const llr=h=>{const a=sub(h,W2.mP),b=sub(h,W2.mA);return -.5*dot(a,mv(SPi,a))-.5*ldP+.5*dot(b,mv(SAi,b))+.5*ldA};
  const piA=h=>sigm(-llr(h)*sharp/4);
  const sqrtm=M=>symFn(M,v=>Math.sqrt(Math.max(v,0)));
  const w2=(m1,S1,m2,S2)=>{const S2h=sqrtm(S2);const X=sqrtm(mm(mm(S2h,S1),S2h));const tr=M=>M[0][0]+M[1][1];return Math.sqrt(Math.max(0,dot(sub(m1,m2),sub(m1,m2))+tr(S1)+tr(S2)-2*tr(X)))};
  const pl=new Plot2D($('#c5'),BOX2,draw);let anim=null;
  function draw(){if(!pl.ready())return;loadCols();pl.clear();drawRegions(pl);drawAxis(pl);
    // gated: the tokens of a persona conversation (in character and drifted) get the map in proportion to how
    // assistant-like they look; the assistant cloud is only the reference the map was fit on, so it stays still
    if(mode==='gated'){const ed=h=>{const g=piA(h);return{h:add(h,scl(sub(T(h),h),g)),on:g>.02,g}};
      const m=metrics(ed);drawSteer(pl,ed,'hold');
      reads('#m5',[[pct(m.restore),'restore (drifted)'],[f2(m.dist),'distortion (in character)'],[pct(avg(m.De.map(e=>e.g))),'mean gate on drifted tokens'],[pct(avg(m.Pe.map(e=>e.g))),'mean gate on in-character tokens']])}
    else{const go=h=>mode==='ot'?add(scl(h,1-t),scl(T(h),t)):add(h,scl(W2.d,t));
      const moved=W2.A.map(go),mm_=meanOf(moved),Sm=covOf(moved);drawSteer(pl,h=>({h:go(h),on:t>0}),'set');
      const shapeErr=w2([0,0],Sm,[0,0],W2.SigP);
      reads('#m5',[[f2(w2(mm_,Sm,W2.mP,W2.SigP)),'distance to persona cloud (W₂)'],[f2(shapeErr),'shape mismatch alone (W₂, means aligned)'],[f2(t),'progress t'],[mode==='ot'?'shift + reshape':'shift only','what moves']])}}
  seg('#s5',[['vector','Shift (vector)'],['ot','Transport'],['gated','Gated transport']],mode,v=>{mode=v;draw()});
  slider('r5t',v=>(v/100).toFixed(2),v=>{t=v/100;draw()});slider('r5g',v=>(v/100).toFixed(2),v=>{sharp=v/100;draw()});
  $('#b5p').addEventListener('click',()=>{if(anim){cancelAnimationFrame(anim);anim=null;return}const el=$('#r5t');const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
    if(reduce){el.value=100;t=1;$('#o5t').textContent='1.00';draw();return}let v=0;const step=()=>{v=Math.min(100,v+1.4);el.value=v;t=v/100;$('#o5t').textContent=t.toFixed(2);draw();anim=v<100?requestAnimationFrame(step):null};step()});
})();

/* ================= curved geometry: flat and curved, in general ================= */
// A sheet bent into part of a cylinder. u runs along the bend, v along the straight edge. The geodesic between two
// points is a straight line in (u, v), wrapped onto the sheet, so its length never changes as the sheet bends.
(function(){let bend=.8,t=.5;const Lu=6,Ly=3,N=280,sig=.05;const rnd=mulberry32(61);
  const UV=[];for(let i=0;i<N;i++)UV.push([2*rnd()-1,2*rnd()-1,randn(rnd)]);const Aq=[-.85,-.55],Bq=[.85,.55];
  const geom=()=>{const ph=Math.max(1e-4,bend*Math.PI),R=Lu/ph,zc=R*(1-Math.cos(ph/2))/2;
    const at=(u,v,n=0)=>{const th=u*Lu/2/R;return[R*Math.sin(th)+n*Math.sin(th),v*Ly/2,R*(1-Math.cos(th))-zc-n*Math.cos(th)]};return{at,R}};
  const pl3=new Plot3D($('#c_fc3'),draw,{yaw:-.55,pitch:.38,span:3.4});const pl2=new Plot2D($('#c_fc2'),{x0:-3.4,x1:3.4,y0:-1.7,y1:1.7},draw);
  // the box every bend fits in (flat: x ±3; fully bent: z ±1), so the 3D zoom holds while the bend changes
  const FCB=[];for(const x of[-3,3])for(const y of[-1.5,1.5])for(const z of[-1,1])FCB.push([x,y,z]);
  function draw(){loadCols();const G=geom();const a=G.at(...Aq),b=G.at(...Bq);
    const chord=Array.from({length:41},(_,i)=>add(a,scl(sub(b,a),i/40)));const geo=Array.from({length:41},(_,i)=>{const s=i/40;return G.at(Aq[0]+(Bq[0]-Aq[0])*s,Aq[1]+(Bq[1]-Aq[1])*s)});
    const grid=[];for(let i=0;i<=120;i++)for(let j=0;j<=24;j++)grid.push(G.at(-1+i/60,-1+j/12));
    const gapOf=q=>Math.min(...grid.map(g=>nrm(sub(g,q))));const gaps=chord.map(gapOf);
    const dS=nrm(sub(b,a)),dG=Math.hypot((Bq[0]-Aq[0])*Lu/2,(Bq[1]-Aq[1])*Ly/2);
    const ms=add(a,scl(sub(b,a),t)),mg=G.at(Aq[0]+(Bq[0]-Aq[0])*t,Aq[1]+(Bq[1]-Aq[1])*t);
    if(pl3.ready()){pl3.clear();pl3.fit(FCB);pl3.axes3([-3.1,-2.4,-1.9],.8,['x','y','z']);
      for(let i=0;i<=12;i++){const u=-1+i/6;pl3.poly(Array.from({length:13},(_,j)=>G.at(u,-1+j/6)),A(COL.navy,.18),1)}
      for(let j=0;j<=4;j++){const v=-1+j/2;pl3.poly(Array.from({length:61},(_,i)=>G.at(-1+i/30,v)),A(COL.navy,.18),1)}
      pl3.points(UV.map(([u,v,n])=>({p:G.at(u,v,sig*n),r:1.9,fill:A(COL.persona,.55)})));
      pl3.poly(chord,COL.edit,2.2,{dash:[6,4]});pl3.poly(geo,COL.navy,2.4);
      pl3.points([{p:a,r:6,fill:COL.ochre,stroke:COL.paper,lw:1.5},{p:b,r:6,fill:COL.violet,stroke:COL.paper,lw:1.5},{p:ms,r:5,fill:COL.paper,stroke:COL.edit,lw:2.2},{p:mg,r:5,fill:COL.paper,stroke:COL.navy,lw:2.2}])}
    if(pl2.ready()){pl2.clear();const P2=q=>[q[0],q[2]];
      UV.forEach(([u,v,n])=>pl2.dot(P2(G.at(u,v,sig*n)),2.1,A(COL.persona,.45)));
      pl2.path(Array.from({length:61},(_,i)=>P2(G.at(-1+i/30,0))),A(COL.navy,.35),1);
      pl2.path(chord.map(P2),COL.edit,2,[6,4]);pl2.path(geo.map(P2),COL.navy,2.4);
      pl2.dot(P2(a),6,COL.ochre,COL.paper,1.5);pl2.dot(P2(b),6,COL.violet,COL.paper,1.5);pl2.dot(P2(ms),5,COL.paper,COL.edit,2.2);pl2.dot(P2(mg),5,COL.paper,COL.navy,2.2);
      pl2.text([-3.25,1.5],'end-on: x across, z up',COL.muted)}
    reads('#m_fc',[[f2(dS),'straight-line distance'],[f2(dG),'distance along the sheet (never changes)'],[f2(dG/dS),'ratio: along the sheet / straight'],[f2(Math.max(...gaps)),'largest gap between the straight path and the sheet']])}
  slider('r_fcb',v=>(v/100).toFixed(2),v=>{bend=v/100;draw()});slider('r_fct',v=>(v/100).toFixed(2),v=>{t=v/100;draw()});
})();

/* ================= curved geometry: straight lines leave curved data ================= */
(function(){let bend=.7,t=.5;const N=170,sig=.12;const BOXM={x0:-5,x1:5,y0:-3.6,y1:3.6};
  const pl=new Plot2D($('#c_mf'),BOXM,draw);let S=null;
  function build(){const r=mulberry32(31);const phi=Math.max(.03,bend*1.5*Math.PI),R=8.2/phi;
    const raw=[];for(let i=0;i<N;i++){const th=-phi/2+phi*(i+.5)/N+(r()-.5)*phi/N;const rr=R+sig*randn(r);raw.push([rr*Math.sin(th),rr*Math.cos(th)])}
    const ends=[[R*Math.sin(-phi/2),R*Math.cos(-phi/2)],[R*Math.sin(phi/2),R*Math.cos(phi/2)]];
    const all=raw.concat(ends);const x0=Math.min(...all.map(p=>p[0])),x1=Math.max(...all.map(p=>p[0])),y0=Math.min(...all.map(p=>p[1])),y1=Math.max(...all.map(p=>p[1]));
    const c=[(x0+x1)/2,(y0+y1)/2];const pts=raw.map(p=>sub(p,c)),a=sub(ends[0],c),b=sub(ends[1],c);
    const V=pts.concat([a,b]);const n=V.length,K=8;const adj=V.map(()=>[]);
    for(let i=0;i<n;i++){V.map((q,j)=>[j,nrm(sub(q,V[i]))]).filter(x=>x[0]!==i).sort((u,v)=>u[1]-v[1]).slice(0,K).forEach(([j,w])=>{adj[i].push([j,w]);adj[j].push([i,w])})}
    const dist=Array(n).fill(Infinity),prev=Array(n).fill(-1),done=Array(n).fill(false);dist[n-2]=0;
    for(let it=0;it<n;it++){let u=-1;for(let i=0;i<n;i++)if(!done[i]&&(u<0||dist[i]<dist[u]))u=i;if(u<0||dist[u]===Infinity)break;done[u]=true;if(u===n-1)break;
      adj[u].forEach(([j,w])=>{if(dist[u]+w<dist[j]){dist[j]=dist[u]+w;prev[j]=u}})}
    let geo=[];if(dist[n-1]<Infinity){for(let v=n-1;v>=0;v=prev[v]){geo.push(V[v]);if(v===n-2)break}geo.reverse()}else geo=[a,b];
    S={pts,a,b,geo,dg:dist[n-1]<Infinity?dist[n-1]:nrm(sub(b,a)),de:nrm(sub(b,a))}}
  const lenOf=path=>{let L=0;for(let i=1;i<path.length;i++)L+=nrm(sub(path[i],path[i-1]));return L};
  const at=(path,f)=>{const L=lenOf(path)*f;let acc=0;for(let i=1;i<path.length;i++){const s=nrm(sub(path[i],path[i-1]));if(acc+s>=L)return add(path[i-1],scl(sub(path[i],path[i-1]),(L-acc)/Math.max(s,1e-9)));acc+=s}return path[path.length-1]};
  const gap=p=>Math.min(...S.pts.map(q=>nrm(sub(q,p))))/sig;
  function draw(){if(!pl.ready())return;loadCols();if(!S)build();pl.clear();
    S.pts.forEach(p=>pl.dot(p,2.4,A(COL.persona,.55)));
    const straight=[S.a,S.b];pl.path(straight,COL.edit,2,[6,4]);pl.path(S.geo,COL.navy,2.4);
    const ms=at(straight,t),mg=at(S.geo,t);pl.dot(ms,6,A(COL.paper,.9),COL.edit,2.2);pl.dot(mg,6,A(COL.paper,.9),COL.navy,2.2);
    pl.dot(S.a,6,COL.ochre,COL.paper,1.5);pl.dot(S.b,6,COL.violet,COL.paper,1.5);
    pl.text(add(S.a,[0,-.42]),'persona A',COL.ochre,{align:'center',font:'600 12px'});pl.text(add(S.b,[0,-.42]),'persona B',COL.violet,{align:'center',font:'600 12px'});
    pl.text([BOXM.x0+.2,BOXM.y1-.3],'orange dashed: straight blend · navy: geodesic through the data',COL.ink,{font:'600 12px'});
    const samp=Array.from({length:61},(_,i)=>gap(at(straight,i/60)));
    reads('#m_mf',[[f2(S.dg/S.de),'d_geodesic / d_straight'],[samp.reduce((a,b)=>Math.max(a,b),0).toFixed(1)+'σ','straight path: largest gap from the data'],[pct(avg(samp.map(x=>x>3?1:0))),'straight path more than 3σ from any token'],
      [gap(ms).toFixed(1)+'σ / '+gap(mg).toFixed(1)+'σ','gap at t: straight / geodesic']])}
  slider('r_mfb',v=>(v/100).toFixed(2),v=>{bend=v/100;S=null;draw()});slider('r_mft',v=>(v/100).toFixed(2),v=>{t=v/100;draw()});
  const c2=$('#c_mf2');const drawRep=()=>{loadCols();lineChart(c2,{x:{min:0,max:1,ticks:[0,.25,.5,.75,1],label:'blend position t'},y:{min:.4,max:.75,ticks:[.4,.5,.6,.7],fmt:v=>v.toFixed(1)},
    series:[{pts:[[0,.72],[.25,.54],[.5,.43],[.75,.57],[1,.72]],col:COL.edit,dash:[6,4],marker:3.5,label:'straight blend',labelDy:14,labelBase:'top'},
            {pts:[[0,.72],[.25,.69],[.5,.66],[.75,.68],[1,.72]],col:COL.navy,marker:3.5,label:'geodesic blend'}]})};
  watch(c2,drawRep);
})();

/* ================= curved geometry: keep the length, turn the direction ================= */
(function(){let mode='both',t=.7;const r=mulberry32(77);
  const aDir=unit([-.6,.3,.74]),pDir=unit([.62,.32,.72]);
  const jit=(u,s)=>unit(add(u,[s*randn(r),s*randn(r),s*randn(r)]));
  const CA=[],CP=[];for(let i=0;i<46;i++){CA.push(jit(aDir,.14));CP.push(jit(pDir,.14))}
  const h0=jit(aDir,.1),pm=unit(meanOf(CP)),d=sub(meanOf(CP),meanOf(CA));
  const slerp=(a,b,s)=>{const th=Math.acos(clamp(dot(a,b),-1,1));if(th<1e-6)return a;return add(scl(a,Math.sin((1-s)*th)/Math.sin(th)),scl(b,Math.sin(s*th)/Math.sin(th)))};
  const angDeg=(a,b)=>Math.acos(clamp(dot(unit(a),unit(b)),-1,1))*180/Math.PI;
  const pl=new Plot3D($('#c_sp'),draw,{yaw:-.45,pitch:.3,span:1.5});
  // points just outside the unit sphere, where the labels sit, so the globe and its labels fill the canvas
  const SPB=[];for(let i=0;i<40;i++){const z=-1+2*(i+.5)/40,rr=Math.sqrt(1-z*z),a=i*2.4;SPB.push(scl([rr*Math.cos(a),rr*Math.sin(a),z],1.2))}
  function draw(){if(!pl.ready())return;loadCols();pl.clear();pl.fit(SPB);
    for(let lat=-60;lat<=60;lat+=30){const z=Math.sin(lat*Math.PI/180),rr=Math.cos(lat*Math.PI/180),ring=[];for(let i=0;i<=72;i++){const a=i/72*2*Math.PI;ring.push([rr*Math.cos(a),rr*Math.sin(a),z])}pl.poly(ring,A(COL.muted,.22),1)}
    for(let lon=0;lon<180;lon+=30){const a=lon*Math.PI/180,mer=[];for(let i=0;i<=72;i++){const b=i/72*2*Math.PI;mer.push([Math.cos(b)*Math.cos(a),Math.cos(b)*Math.sin(a),Math.sin(b)])}pl.poly(mer,A(COL.muted,.22),1)}
    pl.poly([[0,0,0],aDir],A(COL.assist,.6),1.2,{dash:[3,3]});pl.poly([[0,0,0],pm],A(COL.persona,.6),1.2,{dash:[3,3]});
    const pts=[];CA.forEach(p=>pts.push({p,r:2.4,fill:A(COL.assist,.55)}));CP.forEach(p=>pts.push({p,r:2.4,fill:A(COL.persona,.7)}));pl.points(pts);
    pl.text([0,0,-.08],'zero vector',COL.muted,'500 11px');pl.points([{p:[0,0,0],r:3,fill:COL.ink}]);
    const steps=40,addP=[],rotP=[];for(let i=0;i<=steps;i++){const s=t*i/steps;addP.push(add(h0,scl(d,s)));rotP.push(slerp(h0,pm,s))}
    if(mode!=='rotate'){pl.poly(addP,COL.edit,2.4);pl.points([{p:addP[steps],r:5,fill:COL.edit,stroke:COL.paper,lw:1.5}])}
    if(mode!=='add'){pl.poly(rotP,COL.navy,2.4);pl.points([{p:rotP[steps],r:5,fill:COL.navy,stroke:COL.paper,lw:1.5}])}
    pl.points([{p:h0,r:5.5,fill:COL.paper,stroke:COL.drift,lw:2}]);
    pl.text(add(aDir,scl(aDir,.16)),'assistant',COL.assist);pl.text(add(pm,scl(pm,.16)),'persona',COL.persona);
    const ea=addP[steps],er=rotP[steps];const minAdd=Math.min(...addP.map(nrm));
    reads('#m_sp',[[f2(nrm(ea)),'length after adding (start = 1)'],[f2(minAdd),'shortest length along the added path'],[f2(nrm(er)),'length after rotating'],
      [angDeg(ea,pm).toFixed(0)+'° / '+angDeg(er,pm).toFixed(0)+'°','angle to the persona: added / rotated']])}
  seg('#s_sp',[['add','Add'],['rotate','Rotate'],['both','Both']],mode,v=>{mode=v;draw()});
  slider('r_spt',v=>(v/100).toFixed(2),v=>{t=v/100;draw()});
})();

/* ================= curved geometry: sharp directions of the behaviour loss ================= */
(function(){let share=.03,ratio=79;const af=35*Math.PI/180,f=[Math.cos(af),Math.sin(af)],s=[-Math.sin(af),Math.cos(af)];
  const pl=new Plot2D($('#c_sh'),{x0:-1,x1:4.2,y0:-.7,y1:3.1},draw);
  const ratioOf=v=>5*Math.pow(10,2*v/100);
  function draw(){if(!pl.ready())return;loadCols();pl.clear();
    const v=scl(add(scl(f,Math.sqrt(1-share)),scl(s,Math.sqrt(share))),3),vc=scl(f,3*Math.sqrt(1-share));
    pl.line(sub(v,scl(s,1.3)),add(v,scl(s,1.3)),A(COL.violet,.7),1.2,[4,4]);pl.text(add(v,scl(s,1.45)),'sharp direction',COL.violet,{align:'center'});
    pl.line(sub(v,scl(f,1.2)),add(v,scl(f,.9)),A(COL.muted,.7),1.2,[4,4]);pl.text(add(v,scl(f,1.05)),'flat',COL.muted);
    [.02,.08,.18,.32].forEach(c=>pl.ellipse(v,[f,s],[Math.sqrt(2*c),Math.sqrt(2*c/ratio)],A(COL.navy,.45),{w:1}));
    pl.arrow([0,0],v,COL.edit,2.6,10);pl.arrow([0,0],vc,COL.navy,2.4,9);pl.line(vc,v,COL.edit,1.4,[3,3]);
    pl.dot([0,0],3.5,COL.ink);pl.text([.08,-.25],'origin of steering-vector space',COL.muted);
    pl.text(add(scl(v,.5),scl(s,.25)),'original v',COL.edit,{align:'center',font:'600 12px'});pl.text(add(scl(vc,.62),scl(s,-.28)),'sharp part removed',COL.navy,{align:'center',font:'600 12px'});
    const before=(1-share)+share*ratio;
    reads('#m_sh',[[(Math.sqrt(1-share)).toFixed(3),'cosine, original vs cleaned'],['−'+(100*(1-Math.sqrt(1-share))).toFixed(1)+'%','length change'],
      [before.toFixed(1)+' → 1.0','directional curvature vᵀHv/|v|² (flat = 1)'],[(before).toFixed(1)+'×','curvature reduction']])}
  slider('r_shs',v=>(v/10).toFixed(1)+'%',v=>{share=v/1000;draw()});slider('r_shk',v=>ratioOf(v).toFixed(0),v=>{ratio=ratioOf(v);draw()});
})();

/* ================= over turns: holding is control ================= */
(function(){const TURNS=20,TPT=12,NT=TURNS*TPT;const PUSH=[5,13],PROBE=[8,16];let seed=3;
  let P=.8,mem=.6,gamma=.3,tau=.6,ck=10;const PULL={1:.11,3:.20,10:.33,30:.49,100:.68};
  const CTRL=[{id:'none',label:'Prompt alone',c:'muted',on:true},{id:'vector',label:'Vector',c:'edit',on:true},{id:'cap',label:'Cap',c:'assist',on:true},
    {id:'partial',label:'Partial correction',c:'navy',on:false},{id:'ctok',label:'Conceptor pull',c:'persona',on:true},{id:'antic',label:'Cap + anticipation',c:'drift',on:false},
    {id:'integral',label:'Integral action',c:'violet',on:false},{id:'attractor',label:'Attractor with memory',c:'ochre',on:false}];
  loadCols();const box=$('#k6');CTRL.forEach(c=>{const l=document.createElement('label');const inp=document.createElement('input');inp.type='checkbox';inp.id='k6-'+c.id;inp.checked=c.on;
    const sw=document.createElement('i');sw.className='sw ln';sw.style.background=`var(--${c.c})`;l.append(inp,sw,document.createTextNode(c.label));
    inp.addEventListener('change',()=>{c.on=inp.checked;draw()});box.appendChild(l)});
  function sim(id){const r=mulberry32(seed);let s=1,b=1,I=0,z=0,acc=0;const xs=[],us=[];
    for(let i=0;i<NT;i++){const turn=Math.floor(i/TPT)+1;const hit=PUSH.includes(turn)||PROBE.includes(turn);let m=b;if(PUSH.includes(turn))m-=P;if(PROBE.includes(turn))m-=.6*P;
      s=s+.22*(m-s)+.04*randn(r);let u=0;const tEff=id==='antic'&&hit?tau+.25:tau;
      if(id==='vector')u=.09;
      else if((id==='cap'||id==='antic')&&s<tEff)u=tEff-s;
      else if(id==='partial'&&s<tau)u=gamma*(tau-s);
      else if(id==='ctok')u=PULL[ck]*(1-s);
      else if(id==='integral'){const e=Math.max(0,tau-s);I=.96*I+e;u=.25*e+.06*I}
      else if(id==='attractor'){z=.9*z+.1*(1-s);u=.45*z}
      s+=u;xs.push(s);us.push(Math.abs(u));acc+=s;
      if((i+1)%TPT===0){const turnAvg=acc/TPT;acc=0;if(hit&&turnAvg<b)b+=mem*(turnAvg-b)}}
    const late=xs.slice(16*TPT);
    return{xs,inChar:avg(xs.map(x=>x>=.5?1:0)),late:avg(late.map(x=>x>=.5?1:0)),edit:us.reduce((a,c)=>a+c,0),over:avg(xs.map(x=>Math.max(0,x-1.15)))}}
  const cv=$('#c6'),g=cv.getContext('2d');watch(cv,()=>draw());
  function draw(){const rc=fitCanvas(cv,g);if(!rc)return;loadCols();
    const W=rc.width,H=rc.height,pad={l:40,r:12,t:26,b:26},y0=-.4,y1=1.7;const X=i=>pad.l+i/(NT-1)*(W-pad.l-pad.r),Y=v=>pad.t+(y1-v)/(y1-y0)*(H-pad.t-pad.b);
    const font=FONT();g.font=`500 11px ${font}`;
    g.fillStyle=A(COL.persona,.08);g.fillRect(pad.l,Y(1.15),W-pad.l-pad.r,Y(.5)-Y(1.15));
    for(let tn=1;tn<=TURNS;tn++){const x=X((tn-1)*TPT);g.strokeStyle=COL.grid;g.lineWidth=1;g.beginPath();g.moveTo(x,pad.t);g.lineTo(x,H-pad.b);g.stroke();
      if(PUSH.includes(tn)||PROBE.includes(tn)){g.fillStyle=A(PUSH.includes(tn)?COL.drift:COL.ochre,.1);g.fillRect(x,pad.t,X(TPT)-X(0),H-pad.t-pad.b)}
      if(tn%2===1||tn===20){g.fillStyle=COL.muted;g.textAlign='center';g.textBaseline='top';g.fillText(String(tn),x+(X(TPT)-X(0))/2,H-pad.b+5)}}
    g.textAlign='left';g.textBaseline='bottom';g.fillStyle=COL.drift;g.fillText('drop the act: turns 5, 13',pad.l+4,pad.t-6);g.fillStyle=COL.ochre;g.fillText('are you an AI?: turns 8, 16',pad.l+190,pad.t-6);
    g.textAlign='right';g.textBaseline='middle';g.fillStyle=COL.muted;[0,.5,1,1.5].forEach(v=>{g.fillText(v.toFixed(1),pad.l-6,Y(v));g.strokeStyle=COL.hair;g.beginPath();g.moveTo(pad.l,Y(v));g.lineTo(W-pad.r,Y(v));g.stroke()});
    g.strokeStyle=A(COL.navy,.6);g.setLineDash([5,4]);g.beginPath();g.moveTo(pad.l,Y(tau));g.lineTo(W-pad.r,Y(tau));g.stroke();g.setLineDash([]);
    g.save();g.translate(12,pad.t+(H-pad.t-pad.b)/2);g.rotate(-Math.PI/2);g.textAlign='center';g.fillStyle=COL.muted;g.fillText('persona coordinate (1 = persona)',0,0);g.restore();
    const rows=[];CTRL.forEach(c=>{const s=sim(c.id);rows.push([c,s]);if(!c.on)return;g.strokeStyle=COL[c.c];g.lineWidth=c.id==='none'?1.4:1.9;g.beginPath();s.xs.forEach((v,i)=>{const x=X(i),y=Y(clamp(v,y0,y1));i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()});
    $('#t6').innerHTML='<thead><tr><th>Toy controller</th><th>tokens in character (≥ 0.5)</th><th>turns 17-20 in character</th><th>total edit</th><th>overshoot past 1.15</th></tr></thead><tbody>'+
      rows.map(([c,s])=>`<tr><td><i class="sw ln" style="background:var(--${c.c})"></i> ${c.label}${c.id==='ctok'?` (k = ${ck})`:''}</td><td class="num">${pct(s.inChar)}</td><td class="num">${pct(s.late)}</td><td class="num">${s.edit.toFixed(1)}</td><td class="num">${s.over.toFixed(3)}</td></tr>`).join('')+'</tbody>'}
  slider('r6p',v=>(v/100).toFixed(2),v=>{P=v/100;draw()});slider('r6m',v=>(v/100).toFixed(2),v=>{mem=v/100;draw()});
  slider('r6g',v=>(v/100).toFixed(2),v=>{gamma=v/100;draw()});slider('r6t',v=>(v/100).toFixed(2),v=>{tau=v/100;draw()});
  seg('#s6k',[[1,'pull at k = 1'],[3,'k = 3'],[10,'k = 10'],[30,'k = 30'],[100,'k = 100']],ck,v=>{ck=v;draw()});
  $('#b6r').addEventListener('click',()=>{seed=(seed*7919+13)%100000;draw()});
  const NP=DATA.nopush;const cell=x=>x&&x.std!=null?`${pct(x.std)} → ${pct(x.nopush)}`:'n/a';
  $('#t6r').innerHTML='<thead><tr><th>Real, SmolLM3: turns 17-20 in character, with pushes → without</th><th>prompt alone</th><th>vector</th><th>cap</th><th>conceptor, k = 10</th></tr></thead><tbody>'+
    Object.keys(NP).map(p=>`<tr><td>${p}</td><td class="num">${cell(NP[p].none)}</td><td class="num">${cell(NP[p].add)}</td><td class="num">${cell(NP[p].cap)}</td><td class="num">${cell(NP[p].ctok_tight)}</td></tr>`).join('')+'</tbody>';
})();

/* ================= persona space (3D) ================= */
(function(){const r=mulberry32(21);const muA=[-1.8,0,0],core=[3.6,0,0];
  const G=[{n:'assistant',c:'assist',mu:muA,sp:[0,0,0]},
    {n:'pirate',c:'persona',f:1,sp:[0,1.4,.5]},{n:'ghost',c:'violet',f:.95,sp:[0,-.7,1.4]},{n:'bard',c:'olive',f:.9,sp:[0,1.1,-1]},
    {n:'genie',c:'drift',f:1.05,sp:[0,-1.3,-.6]},{n:'consultant',c:'ochre',f:.35,sp:[0,-.3,-.4]}];
  G.forEach(gr=>{if(gr.n!=='assistant')gr.mu=add(add(muA,scl(core,gr.f)),gr.sp)});
  const cu=unit([.45,.75,.48]);let o1=unit([-cu[1],cu[0],0]);const o2=[cu[1]*o1[2]-cu[2]*o1[1],cu[2]*o1[0]-cu[0]*o1[2],cu[0]*o1[1]-cu[1]*o1[0]];
  const NPER=45;G.forEach(gr=>{gr.pts=[];for(let i=0;i<NPER;i++)gr.pts.push(add(gr.mu,add(add(scl(cu,.9*randn(r)),scl(o1,.22*randn(r))),scl(o2,.22*randn(r)))))});
  const mA=meanOf(G[0].pts);const means=G.map(gr=>meanOf(gr.pts));const gm=meanOf(means);
  const SigB=covOf(means);let SigW=[[0,0,0],[0,0,0],[0,0,0]];G.forEach(gr=>{SigW=madd(SigW,covOf(gr.pts))});SigW=mscl(SigW,1/G.length);
  const pca=symEig(SigB);const Wih=symFn(SigW,v=>1/Math.sqrt(Math.max(v,1e-9)));const lda=symEig(mm(mm(Wih,SigB),Wih));const ldaDirs=lda.vecs.map(v=>unit(mv(Wih,v)));
  const Rs=G.map(gr=>momentAbout(gr.pts,mA));const personas=[1,2,3,4,5];
  let view='clouds',sel=1,k=1;const pl=new Plot3D($('#c8'),draw,{yaw:-.75,pitch:.35,span:4.8});
  const I3=eye(3);const quota=C=>(C[0][0]+C[1][1]+C[2][2])/3;
  function andNot(Ci,Co){return inv(msub(madd(inv(Ci),inv(msub(I3,Co))),I3))}
  function drawC(C,col){const{vals,vecs}=symEig(C);pl.ellipsoid(mA,vecs,vals.map(v=>Math.max(v,.03)*3.6),A(col,.55))}
  const ALL8=[].concat(...G.map(gr=>gr.pts));
  function draw(){if(!pl.ready())return;loadCols();pl.clear();pl.fit(ALL8);pl.axes3([-4,-3.6,-2.2],1,['x','y','z']);
    const pts=[];G.forEach(gr=>gr.pts.forEach(p=>pts.push({p,r:2.2,fill:A(COL[gr.c],view==='clouds'?.75:.4)})));pl.points(pts);
    G.forEach((gr,i)=>pl.text(add(means[i],[0,0,gr.n==='consultant'||gr.n==='bard'?-1.05:1.05]),gr.n,COL[gr.c]));
    let rd=[];
    if(view==='pca'){pca.vecs.forEach((v,i)=>{const L=1.6*Math.sqrt(Math.max(pca.vals[i],0))+.3;pl.arrow(gm,add(gm,scl(v,L)),COL.navy,2.4-i*.5);pl.text(add(gm,scl(v,L+.35)),'PC'+(i+1),COL.navy)});
      const tot=pca.vals.reduce((a,b)=>a+b,0);rd=[[pca.vals.map(v=>pct(v/tot)).join(' / '),'variance of means per PC'],[f2(Math.abs(dot(pca.vecs[0],unit(core)))),'cos(PC1, shared core)']]}
    else if(view==='lda'){ldaDirs.forEach((v,i)=>{if(i>1&&lda.vals[i]<1e-6)return;const L=2.3-i*.55;pl.arrow(gm,add(gm,scl(v,L)),COL.violet,2.4-i*.5);pl.text(add(gm,scl(v,L+.35)),'LD'+(i+1),COL.violet)});
      const tot=lda.vals.reduce((a,b)=>a+Math.max(b,0),0);rd=[[lda.vals.map(v=>pct(Math.max(v,0)/tot)).join(' / '),'separation per discriminant axis'],[f2(Math.abs(dot(ldaDirs[0],unit(core)))),'cos(LD1, shared core)']]}
    else if(view==='or'){let R=[[0,0,0],[0,0,0],[0,0,0]];personas.forEach(i=>{R=madd(R,Rs[i])});R=mscl(R,1/personas.length);const C=conceptor(R,k);drawC(C,COL.navy);
      const{vals,vecs}=symEig(C);pl.arrow(mA,add(mA,scl(vecs[0],vals[0]*3.6*Math.sign(dot(vecs[0],core)||1))),COL.navy,2.4);
      rd=[[f2(quota(C)),'quota of persona space (OR)'],[f2(Math.abs(dot(vecs[0],unit(core)))),'cos(top direction, shared core)']]}
    else if(view==='andnot'){const rest=personas.filter(i=>i!==sel);let Ro=[[0,0,0],[0,0,0],[0,0,0]];rest.forEach(i=>{Ro=madd(Ro,Rs[i])});Ro=mscl(Ro,1/rest.length);
      const Ci=conceptor(Rs[sel],k),Co=conceptor(Ro,k),Cx=andNot(Ci,Co);drawC(Ci,COL[G[sel].c]);drawC(Cx,COL.ink);
      const{vals,vecs}=symEig(Cx);const off=G[sel].sp;let v0=vecs[0];if(dot(v0,sub(means[sel],mA))<0)v0=scl(v0,-1);pl.arrow(mA,add(mA,scl(v0,Math.max(vals[0],.08)*3.6*1.4)),COL.ink,2.6);
      rd=[[f2(quota(Ci))+' → '+f2(quota(Cx)),'quota: '+G[sel].n+' → AND NOT rest'],[f2(Math.abs(dot(v0,unit(off)))),'cos(kept direction, true '+G[sel].n+' offset)'],[f2(Math.abs(dot(v0,unit(core)))),'cos(kept direction, shared core)']]}
    else{rd=[['6','groups'],[String(NPER),'tokens each']]}
    pl.arrow(mA,add(mA,core),A(COL.gold,.95),2.2);pl.text(add(mA,scl(core,.5)),'shared core',COL.ochre,'600 11px');
    reads('#m8',rd)}
  // the deck's main slide: the same clouds, the shared core and each persona's own offset from the core line
  const cm=$('#c8m'),plm=cm&&new Plot3D(cm,drawMain,{yaw:-.75,pitch:.35,span:4.8});
  function drawMain(){if(!plm||!plm.ready())return;loadCols();plm.clear();plm.fit(ALL8);plm.axes3([-4,-3.6,-2.2],1,['x','y','z']);
    const pts=[];G.forEach(gr=>gr.pts.forEach(p=>pts.push({p,r:2.2,fill:A(COL[gr.c],.75)})));plm.points(pts);
    G.forEach((gr,i)=>{if(i)plm.arrow(add(mA,scl(core,gr.f)),means[i],A(COL[gr.c],.9),1.4)});
    plm.arrow(mA,add(mA,core),A(COL.gold,.95),2.4);plm.text(add(mA,scl(core,.5)),'shared core',COL.ochre,'600 11px');
    G.forEach((gr,i)=>plm.text(add(means[i],[0,0,gr.n==='consultant'||gr.n==='bard'?-1.05:1.05]),gr.n,COL[gr.c]));
    reads('#m8m',[['6','groups'],[String(NPER),'tokens each'],['1 + 5','shared core + own offsets']])}
  seg('#s8v',[['clouds','Clouds'],['pca','PCA of means'],['lda','Discriminant axes'],['or','Persona space (OR)'],['andnot','AND NOT the rest']],view,v=>{view=v;draw()});
  seg('#s8p',personas.map(i=>[i,G[i].n]),sel,v=>{sel=v;draw()});
  slider('r8a',v=>Math.pow(10,v/100).toFixed(2),v=>{k=Math.pow(10,v/100);draw()});
})();

/* ================= the comparison table and the sources ================= */
(function(){const M=DATA.means;
  const hold=key=>{const s=M.smol[key],o=M.olmo[key];const f=x=>x?`${pct(x.ord)} / ${pct(x.push)} / ${f0(x.coh)}`:null;
    const parts=[];if(s)parts.push(`SmolLM3 ${f(s)}`);if(o)parts.push(`OLMo ${f(o)}`);return parts.join('<br>')};
  const NR='<span class="none">not run</span>';
  const R=[
    ['What we ran'],
    ['Steering vector (ActAdd, CAA, persona vectors)','every token','add c·d','mean difference of two prompt conditions','single-turn behaviour and trait steering on Llama 2 and later models; persona vectors also monitor traits during training',hold('add')],
    ['Activation cap, one layer','token below a floor','lift to the floor along one axis','persona axis plus a percentile floor','Assistant Axis: capping the assistant axis cut harmful jailbreak responses by about 60% (single-turn)',hold('cap')],
    ['Activation cap, Assistant Axis style','below a floor, six layers','lift to the floor at each layer','an axis and floor per layer','as above',hold('cap_aa')],
    ['Conceptor, median aperture (ours)','every token','shrink toward the persona mean, per direction','per-token persona covariance','function steering on GPT-J 81.6% against 32.0% for addition (Postmus & Abreu); semantic steering win rates 0.70-0.82 against 0.50-0.62 (Triantafyllopoulos et al.); single-turn',hold('conceptor_tok')],
    ['Conceptor, aperture 10x (ours)','every token','same, with a = 10 × median','same','no published multi-turn test',hold('ctok_tight')+'<br>AI admissions 38% on SmolLM3'],
    ['Conceptor + cap (ours)','every token, then below a floor','shrink, then clamp','both fits','none',hold('ctok_cap')],
    ['Vector + cap (ours)','every token, then below a floor','add, then clamp','both fits','none',hold('add_cap')],
    ['Mean-shift control (ours)','every token','pull to the persona mean','persona mean only','the matrix-removed control of Razdan & Balamurugan',hold('meanshift_tok')],
    ['Difference conceptor, diffpull (ours)','every token','pull along the paired-difference subspace','persona minus assistant answers to the same question','none',hold('diffpull')],
    ['Fisher-gated cap, fisher_cap (ours)','Fisher score below a floor','lift along d (data-metric edit)','Fisher direction, percentile floor','none; our proposal',hold('fisher_cap')+'<br>at a lower strength than the cap'],
    ['Region (ours)','outside an ellipse in discriminative coordinates','pull back to the ellipse along d and identity','Fisher score and an identity probe','none; our proposal',hold('region')+'<br>AI admissions 25% on SmolLM3'],
    ['How to move a token'],
    ['LiSeCo','probe score outside a band','minimum-norm move onto the band','a linear probe per layer','English to Spanish, P(Spanish) 0.91-0.99 across three models (single-turn)',NR],
    ['Directional ablation','every token','remove one direction','mean difference','refusal is carried by one direction in 13 chat models up to 72B',NR],
    ['LEACE','every token','closed-form least-squares erasure','labelled representations','no linear classifier can recover the concept afterwards',NR],
    ['PaSS on an INLP subspace','prompt tokens','scale the persona component by β','INLP on activations with and without the persona instruction','emoji persona score −54.0 / +71.4 against −16.7 / +30 for a mean-difference vector (Llama-3-8B, single-turn)',NR],
    ['MiMiC (Gaussian transport)','states nearer the source mean','affine map matching mean and covariance','class means and covariances','GPT-2 large toxicity 0.39 to 0.29, perplexity 24.7 to 30.7 (single-turn)',NR],
    ['Spherical steering','gated by confidence','rotate along a great circle, keep the length','a target direction','about +10 points over addition on TruthfulQA, COPA, StoryCloze',NR],
    ['Angular steering','not stated in the abstract','rotate within a fixed 2D plane','a feature direction','refusal and emotion steering (NeurIPS 2025)',NR],
    ['Householder pseudo-rotation','every token','reflection-based rotation, keeps the length','a target direction','safety benchmarks',NR],
    ['When to act'],
    ['CAST','once per prompt','add the vector to every token of the answer, or none','condition and behaviour vectors (PCA of contrastive states)','refusal on harmless prompts 96.4% to 2.2% while harmful stays 90.7% (Qwen1.5-1.8B, single-turn)',NR],
    ['DSAS','soft logistic gate per token','blend the original and steered state','a per-layer classifier','toxicity 13.67 to 8.64 for CAA on Qwen2.5-1.5B (single-turn)',NR],
    ['GAPS','per neuron and per token','steer only separable, source-like neurons','per-neuron AUROC and Gaussians','Gemma-3 4B toxicity 3.52% to 0.48% at a fixed capability budget (single-turn)',NR],
    ['Which directions to fit'],
    ['Contrastive covariance subspaces (CCS)','every token','add a projected mean difference','scatter of the target class minus nuisance classes','Qwen3-8B, 20-turn dialogues: concept-and-fluent rate 93% to 80%, against 36% to 10% for CAA and 91% to 71% for a system prompt',NR],
    ['SEA (spectral editing)','every token','project onto positive-covarying directions, off negative ones','cross-covariance with positive and negative demonstrations','TruthfulQA MC1 36.96 to 39.41 (Llama-2-7B-chat, single-turn)',NR],
    ['SAE trait routing','a routing module picks facet vectors per context','add the selected facet vectors to the residual stream','a contrastive sparse autoencoder on a 15,000-sample Big Five facet corpus','beat CAA and prompt-only baselines on character fidelity; SAE plus prompt did best',NR],
    ['Moving distributions'],
    ['Influence-weighted transport','every token','optimal transport toward influential concept examples','influence scores of the concept examples','beats activation-transport baselines on toxicity, concept induction and truthfulness',NR],
    ['Persona Dosing','every position','a learned flow, strength = flow time','a learned velocity field conditioned on the trait text','core-trait expression at coherence ≥ 75: 75-86 against 42-68 for CAA (single-turn)',NR],
    ['Cobras','query-adaptive','Schrödinger bridge on the sphere of directions','the two activation distributions','beats prior steering in nearly all model-task settings while largely keeping OOD performance',NR],
    ['Curved geometry'],
    ['Manifold steering','not stated','move along a fitted activation manifold','a manifold fit to activations','linear steering cut off-manifold and gave unnatural outputs; manifold steering followed natural behaviour paths',NR],
    ['PersonaManifold geodesic steering','not stated','interpolate personas along graph geodesics','about 2,000 persona means','coherence at the blend midpoint 0.66 against 0.43 for a straight-line blend',NR],
    ['Sharp-direction removal','every token','drop the steering Hessian’s top directions from v','the steering Hessian','cosine to the original above 0.974; beat CAA in all 25 open-ended conditions',NR],
    ['Over turns'],
    ['Attractor control (CAL)','every token','a latent with memory pulls toward the persona centre','reframing and stress corpora','100-turn dialogues with a shock at turn 50: human recovery rating 4.07 against 3.42 for ActAdd and 2.77 for a prompt',NR],
    ['PID steering','every layer','PID combination of per-layer mean differences','source and target sets','jailbreak success 94.85 against 93.26 for difference-in-means (Llama-3.1-8B); the feedback runs across layers',NR],
    ['Persona space'],
    ['BILLY','not stated','blend several persona vectors','several persona vectors','creative generation; merge rule in the paper',NR],
  ];
  // each method's paper, as links under its name
  const NAMEREF={'Steering vector (ActAdd, CAA, persona vectors)':['actadd','caa','pv'],'Activation cap, one layer':['axis'],'Activation cap, Assistant Axis style':['axis'],
    'Conceptor, median aperture (ours)':['jaeger','postmus','triant'],'Mean-shift control (ours)':['razdan'],'LiSeCo':['liseco'],'Directional ablation':['arditi'],'LEACE':['leace'],
    'PaSS on an INLP subspace':['pass','inlp'],'MiMiC (Gaussian transport)':['mimic'],'Spherical steering':['spherical'],'Angular steering':['angular'],'Householder pseudo-rotation':['hpr'],
    'CAST':['cast'],'DSAS':['dsas'],'GAPS':['gaps'],'Contrastive covariance subspaces (CCS)':['ccs'],'SEA (spectral editing)':['sea'],'SAE trait routing':['sae'],
    'Influence-weighted transport':['influence'],'Persona Dosing':['dosing'],'Cobras':['cobras'],'Manifold steering':['manifold'],'PersonaManifold geodesic steering':['pm'],
    'Sharp-direction removal':['azizi'],'Attractor control (CAL)':['cal'],'PID steering':['pid'],'BILLY':['billy']};
  const nameCell=n=>`<b>${n}</b>`+(NAMEREF[n]?`<br><span class="src">${NAMEREF[n].map(k=>cite(k)).join('; ')}</span>`:'');
  $('#t_all').innerHTML='<thead><tr><th>Method</th><th>When it acts</th><th>How it moves h</th><th>Fit on</th><th>Published evidence and setting</th><th>Our holds (ordinary / pushes / coherence)</th></tr></thead><tbody>'+
    R.map(row=>row.length===1?`<tr class="grp"><td colspan="6">${row[0]}</td></tr>`:`<tr><td>${nameCell(row[0])}</td><td>${row[1]}</td><td>${row[2]}</td><td>${row[3]}</td><td>${row[4]}</td><td class="${row[5]===NR?'':'ours num'}">${row[5]}</td></tr>`).join('')+'</tbody>';
  // the reference list, built from REF so every entry opens the paper itself in a new tab
  const ol=$('#refs');Object.values(REF).sort((a,b)=>a.a.localeCompare(b.a)).forEach(r=>{const li=document.createElement('li');
    li.innerHTML=`${escH(r.a)}. <a href="${r.url}" target="_blank" rel="noopener">${escH(r.t)}</a>.${r.v?' '+escH(r.v)+'.':''}${r.x?' arXiv '+r.x+'.':' Conference PDF.'}`;ol.appendChild(li)});
})();
