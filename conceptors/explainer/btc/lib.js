"use strict";
/* ================= math ================= */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function randn(r){let u=0;while(u===0)u=r();const v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
const add=(a,b)=>a.map((x,i)=>x+b[i]);
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
const scl=(a,s)=>a.map(x=>x*s);
const nrm=a=>Math.sqrt(dot(a,a));
const unit=a=>scl(a,1/nrm(a));
const mv=(A,x)=>A.map(r=>dot(r,x));
const mm=(A,B)=>A.map(r=>B[0].map((_,j)=>{let s=0;for(let k=0;k<r.length;k++)s+=r[k]*B[k][j];return s}));
const eye=n=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));
const madd=(A,B)=>A.map((r,i)=>r.map((x,j)=>x+B[i][j]));
const msub=(A,B)=>A.map((r,i)=>r.map((x,j)=>x-B[i][j]));
const mscl=(A,s)=>A.map(r=>r.map(x=>x*s));
const outer=(a,b)=>a.map(x=>b.map(y=>x*y));
const avg=a=>a.reduce((s,x)=>s+x,0)/a.length;
const clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));
const sigm=z=>1/(1+Math.exp(-clamp(z,-60,60)));
function inv(A0){const n=A0.length;const A=A0.map((r,i)=>r.concat(eye(n)[i]));
  for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;[A[c],A[p]]=[A[p],A[c]];
    const d=A[c][c];for(let j=0;j<2*n;j++)A[c][j]/=d;
    for(let r=0;r<n;r++)if(r!==c){const f=A[r][c];if(f!==0)for(let j=0;j<2*n;j++)A[r][j]-=f*A[c][j]}}
  return A.map(r=>r.slice(n))}
function symEig(A0){const n=A0.length;const A=A0.map(r=>r.slice());const V=eye(n);
  for(let sw=0;sw<80;sw++){let off=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)off+=A[i][j]*A[i][j];if(off<1e-20)break;
    for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){if(Math.abs(A[p][q])<1e-15)continue;
      const th=(A[q][q]-A[p][p])/(2*A[p][q]);const t=(th>=0?1:-1)/(Math.abs(th)+Math.sqrt(th*th+1));const c=1/Math.sqrt(t*t+1),s=t*c;
      for(let k=0;k<n;k++){const x=A[k][p],y=A[k][q];A[k][p]=c*x-s*y;A[k][q]=s*x+c*y}
      for(let k=0;k<n;k++){const x=A[p][k],y=A[q][k];A[p][k]=c*x-s*y;A[q][k]=s*x+c*y}
      for(let k=0;k<n;k++){const x=V[k][p],y=V[k][q];V[k][p]=c*x-s*y;V[k][q]=s*x+c*y}}}
  const idx=[...Array(n).keys()].sort((i,j)=>A[j][j]-A[i][i]);
  return{vals:idx.map(i=>A[i][i]),vecs:idx.map(i=>V.map(r=>r[i]))}}
function symFn(A,f){const{vals,vecs}=symEig(A);let R=A.map(r=>r.map(()=>0));vals.forEach((v,k)=>{R=madd(R,mscl(outer(vecs[k],vecs[k]),f(v)))});return R}
function meanOf(P){const m=P[0].map(()=>0);P.forEach(p=>p.forEach((x,i)=>m[i]+=x));return m.map(x=>x/P.length)}
function momentAbout(P,c){const n=c.length;let S=Array.from({length:n},()=>Array(n).fill(0));
  P.forEach(p=>{const d=sub(p,c);for(let i=0;i<n;i++)for(let j=0;j<n;j++)S[i][j]+=d[i]*d[j]});return mscl(S,1/P.length)}
const covOf=P=>momentAbout(P,meanOf(P));
const median=a=>{const s=a.slice().sort((x,y)=>x-y);const m=s.length>>1;return s.length%2?s[m]:(s[m-1]+s[m])/2};
function conceptor(R,k){const{vals}=symEig(R);const a=k*median(vals.map(v=>Math.max(v,0)));return symFn(R,l=>Math.max(l,0)/(Math.max(l,0)+a))}
function quantile(sorted,q){const i=(sorted.length-1)*q;const lo=Math.floor(i),hi=Math.ceil(i);return sorted[lo]+(sorted[hi]-sorted[lo])*(i-lo)}
function auroc(neg,pos){let w=0;for(const p of pos)for(const n of neg)w+=p>n?1:p===n?.5:0;return w/(pos.length*neg.length)}
const rot2=(v,a)=>[v[0]*Math.cos(a)-v[1]*Math.sin(a),v[0]*Math.sin(a)+v[1]*Math.cos(a)];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const slerpU=(a,b,t)=>{const th=Math.acos(clamp(dot(a,b),-1,1));if(th<1e-6)return a;return add(scl(a,Math.sin((1-t)*th)/Math.sin(th)),scl(b,Math.sin(t*th)/Math.sin(th)))};
const pct=x=>x==null?'n/a':(100*x).toFixed(0)+'%';
const f2=x=>x.toFixed(2);
const f0=x=>x==null?'n/a':x.toFixed(0);

/* ================= references: one registry for inline links, the reference list and the table ================= */
// Every arXiv ID here was checked against the arXiv API on 6 Nov 2026 (title and first author).
const REF={
  actadd:{a:'Turner et al. 2023',t:'Steering Language Models With Activation Engineering',x:'2308.10248'},
  arditi:{a:'Arditi et al. 2024',t:'Refusal in Language Models Is Mediated by a Single Direction',x:'2406.11717'},
  axis:{a:'Lu et al. 2026',t:'The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models',v:'ICML 2026',x:'2601.10387'},
  azizi:{a:'Azizi et al. 2026',t:'The Sharp Directions Are Against You: Curvature Analysis of Activation Steering',x:'2610.16084'},
  billy:{a:'Pai et al. 2025',t:'BILLY: Steering Large Language Models via Merging Persona Vectors for Creative Generation',x:'2510.10157'},
  caa:{a:'Panickssery et al. 2024',t:'Steering Llama 2 via Contrastive Activation Addition',v:'ACL 2024',x:'2312.06681'},
  cal:{a:'Chang, Chiu & Kao 2026',t:'Persona as a Dynamical System: Attractor-Based Character Control for Neural Text Generation',v:'INLG 2026',url:'https://2026.inlgmeeting.org/static/proceedings/long/watermarked_pdfs/43.pdf'},
  cast:{a:'Lee et al. 2024',t:'Programming Refusal with Conditional Activation Steering',x:'2409.05907'},
  casting:{a:'Song et al. 2026',t:'Role Steering of Language Models for Social Simulations (Casting Call)',x:'2608.00023'},
  ccs:{a:'Xenos et al. 2026',t:'Contrastive Covariance Subspaces for Activation Steering',x:'2610.14025'},
  cobras:{a:'Dalili et al. 2026',t:'Conditional Optimal Bridge for Riemannian Activation Steering (Cobras)',x:'2607.10517'},
  dosing:{a:'Jin et al. 2026',t:'Persona Dosing: Calibrated Activation Steering for Graded Trait Control',x:'2609.36388'},
  dsas:{a:'Ferrando et al. 2025',t:'Dynamically Scaled Activation Steering (DSAS)',x:'2512.03661'},
  gaps:{a:'Fereidouni et al. 2026',t:'GAPS: Dimension-Level Gates for Conditional Activation Steering',x:'2609.01878'},
  hpr:{a:'Pham & Nguyen 2024',t:'Householder Pseudo-Rotation: A Novel Approach to Activation Editing in LLMs with Direction-Magnitude Perspective',x:'2409.10053'},
  influence:{a:'Dixon, Roberts & Quinzan 2026',t:'Steering by Influence: Curvature Aware Data Weighting for Activation Steering',x:'2610.06383'},
  inlp:{a:'Ravfogel et al. 2020',t:'Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (INLP)',x:'2004.07667'},
  jaeger:{a:'Jaeger 2014',t:'Controlling Recurrent Neural Networks by Conceptors',x:'1403.3369'},
  leace:{a:'Belrose et al. 2023',t:'LEACE: Perfect linear concept erasure in closed form',x:'2306.03819'},
  liseco:{a:'Cheng & Amo Alonso 2024',t:'LiSeCo: Linear Semantic Control for Language Generation',x:'2405.15454'},
  manifold:{a:'Wurgaft et al. 2026',t:'Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior',x:'2605.05115'},
  mimic:{a:'Singh et al. 2024',t:'Representation Surgery: Theory and Practice of Affine Steering (MiMiC)',v:'ICML 2024',x:'2402.09631'},
  pass:{a:'Malik & ElSherief 2026',t:'Extracting Persona Subspaces Through Iterative Nullspace Projection For Modulation (PaSS)',x:'2610.04676'},
  pid:{a:'Nguyen et al. 2025',t:'Activation Steering with a Feedback Controller (PID steering)',x:'2510.04309'},
  pm:{a:'Xu, Xu & Wu 2026',t:'PersonaManifold: Revealing and Exploiting Curved Geometry in LLM Persona Representations',x:'2609.34571'},
  postmus:{a:'Postmus & Abreu 2024',t:'Steering Large Language Models using Conceptors: Improving Addition-Based Activation Engineering',x:'2410.16314'},
  pv:{a:'Chen et al. 2025',t:'Persona Vectors: Monitoring and Controlling Character Traits in Language Models',x:'2507.21509'},
  razdan:{a:'Razdan & Balamurugan 2026',t:'When a Conceptor Complement Becomes Gain: Operator Geometry and Behavioral Attribution in Language Models',x:'2610.16325'},
  sae:{a:'Tang et al. 2026',t:'Facet-Level Persona Control by Trait-Activated Routing with Contrastive SAE for Role-Playing LLMs',x:'2602.19157'},
  sea:{a:'Qiu et al. 2024',t:'Spectral Editing of Activations for Large Language Model Alignment (SEA)',x:'2405.09719'},
  spherical:{a:'You, Deng & Chen 2026',t:'Spherical Steering: Geometry-Aware Activation Rotation for Language Models',x:'2602.08169'},
  triant:{a:'Triantafyllopoulos et al. 2026',t:'Conceptors for Semantic Steering',x:'2605.04980'},
  angular:{a:'Vu & Nguyen 2025',t:'Angular Steering: Behavior Control via Rotation in Activation Space',x:'2510.26243'},
};
Object.values(REF).forEach(r=>{if(!r.url)r.url='https://arxiv.org/abs/'+r.x});
const escH=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
// cite('mimic') -> a link to the paper itself, opening in a new tab; cite('mimic','MiMiC') uses that link text
function cite(k,txt){const r=REF[k];if(!r)throw new Error('unknown reference '+k);return`<a href="${r.url}" target="_blank" rel="noopener" title="${escH(r.t)}">${txt||r.a}</a>`}
// body links are written <a data-ref="key">text</a>; this fills in the address, so every link comes from REF
function linkRefs(root=document){root.querySelectorAll('a[data-ref]').forEach(a=>{const r=REF[a.dataset.ref];if(!r)throw new Error('unknown reference '+a.dataset.ref);
  a.href=r.url;a.target='_blank';a.rel='noopener';a.title=r.t;if(!a.textContent.trim())a.textContent=r.a})}

/* ================= colors ================= */
const COL={};
function loadCols(){const cs=getComputedStyle(document.documentElement);['persona','assist','drift','edit','navy','gold','ink','muted','hair','grid','violet','ochre','paper','panel','sunk','olive'].forEach(k=>COL[k]=cs.getPropertyValue('--'+k).trim())}
function A(hex,a){const h=hex.replace('#','');const n=parseInt(h,16);return`rgba(${n>>16&255},${n>>8&255},${n&255},${a})`}
const FONT=()=>getComputedStyle(document.body).fontFamily;

/* ================= UI helpers ================= */
const $=s=>document.querySelector(s);
function seg(id,opts,val,on){const el=$(id);el.innerHTML='';opts.forEach(([v,l])=>{const b=document.createElement('button');b.type='button';b.textContent=l;b.setAttribute('aria-pressed',String(v===val));
  b.addEventListener('click',()=>{[...el.children].forEach(c=>c.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');on(v)});el.appendChild(b)})}
function slider(id,fmt,on){const el=$('#'+id),out=$('#'+id.replace(/^r/,'o'));const upd=()=>{out.textContent=fmt(+el.value);on(+el.value)};el.addEventListener('input',upd);out.textContent=fmt(+el.value);return()=>+el.value}
function reads(id,items){$(id).innerHTML=items.map(([v,l])=>`<div><b>${v}</b><span>${l}</span></div>`).join('')}
// redraw on resize, and once right after setup even if no resize callback ever arrives (hidden pages, thumbnails)
function watch(el,fn){const ro=new ResizeObserver(()=>fn());ro.observe(el);setTimeout(fn,0);return ro}
function fitCanvas(c,g){const r=c.getBoundingClientRect();if(!r.width)return null;const dpr=window.devicePixelRatio||1;c.width=Math.round(r.width*dpr);c.height=Math.round(r.height*dpr);g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,r.width,r.height);return r}
function arrowPx(g,x0,y0,x1,y1,col,w=1.4,head=6){const L=Math.hypot(x1-x0,y1-y0);if(L<1.5)return;g.save();g.strokeStyle=col;g.fillStyle=col;g.lineWidth=w;g.beginPath();g.moveTo(x0,y0);g.lineTo(x1,y1);g.stroke();
  const an=Math.atan2(y1-y0,x1-x0);const h=Math.min(head,L*.6);g.beginPath();g.moveTo(x1,y1);g.lineTo(x1-h*Math.cos(an-.42),y1-h*Math.sin(an-.42));g.lineTo(x1-h*Math.cos(an+.42),y1-h*Math.sin(an+.42));g.closePath();g.fill();g.restore()}

/* ================= 2D plot ================= */
class Plot2D{
  constructor(canvas,box,draw){this.c=canvas;this.g=canvas.getContext('2d');this.box=box;this.draw=draw;this.handles=[];this.drag=null;
    const pt=e=>{const r=canvas.getBoundingClientRect();return[e.clientX-r.left,e.clientY-r.top]};
    canvas.addEventListener('pointerdown',e=>{const p=pt(e);for(const h of this.handles){const s=this.S(h.get());if(Math.hypot(s[0]-p[0],s[1]-p[1])<20){this.drag=h;canvas.setPointerCapture(e.pointerId);e.preventDefault();return}}});
    canvas.addEventListener('pointermove',e=>{if(!this.drag)return;const p=pt(e);const w=this.Wd(p[0],p[1]);const b=this.box;
      w[0]=clamp(w[0],b.x0,b.x1);w[1]=clamp(w[1],b.y0,b.y1);this.drag.set(w);this.draw()});
    const end=()=>{this.drag=null};canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);
    watch(canvas,()=>{this.resize();this.draw()})}
  resize(){const r=this.c.getBoundingClientRect();if(!r.width)return;const dpr=window.devicePixelRatio||1;this.W=r.width;this.H=r.height;
    this.c.width=Math.round(this.W*dpr);this.c.height=Math.round(this.H*dpr);this.g.setTransform(dpr,0,0,dpr,0,0);
    const b=this.box;const s=Math.min(this.W/(b.x1-b.x0),this.H/(b.y1-b.y0));this.s=s;this.ox=this.W/2-s*(b.x0+b.x1)/2;this.oy=this.H/2+s*(b.y0+b.y1)/2}
  // measure on first use too: a ResizeObserver callback can be late or missing in a hidden or background page
  ready(){if(!(this.W>0))this.resize();return this.W>0}
  S(p){return[this.ox+this.s*p[0],this.oy-this.s*p[1]]}
  Wd(x,y){return[(x-this.ox)/this.s,(this.oy-y)/this.s]}
  clear(){this.g.clearRect(0,0,this.W,this.H);const g=this.g;g.strokeStyle=COL.grid;g.lineWidth=1;const b=this.box;
    for(let x=Math.ceil(b.x0);x<=b.x1;x++){const a=this.S([x,b.y0]),c=this.S([x,b.y1]);g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(c[0],c[1]);g.stroke()}
    for(let y=Math.ceil(b.y0);y<=b.y1;y++){const a=this.S([b.x0,y]),c=this.S([b.x1,y]);g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(c[0],c[1]);g.stroke()}}
  dot(p,r,fill,stroke,lw=1.5){const s=this.S(p),g=this.g;g.beginPath();g.arc(s[0],s[1],r,0,2*Math.PI);if(fill){g.fillStyle=fill;g.fill()}if(stroke){g.strokeStyle=stroke;g.lineWidth=lw;g.stroke()}}
  line(a,b,col,w=1,dash){const g=this.g,s=this.S(a),t=this.S(b);g.save();g.strokeStyle=col;g.lineWidth=w;if(dash)g.setLineDash(dash);g.beginPath();g.moveTo(s[0],s[1]);g.lineTo(t[0],t[1]);g.stroke();g.restore()}
  path(pts,col,w=1.6,dash){const g=this.g;g.save();g.strokeStyle=col;g.lineWidth=w;if(dash)g.setLineDash(dash);g.beginPath();pts.forEach((p,i)=>{const s=this.S(p);i?g.lineTo(s[0],s[1]):g.moveTo(s[0],s[1])});g.stroke();g.restore()}
  arrow(a,b,col,w=1.4,head=6){const s=this.S(a),t=this.S(b);arrowPx(this.g,s[0],s[1],t[0],t[1],col,w,head)}
  ellipse(c,vecs,axes,col,{w=1.6,dash=null,fill=null}={}){const g=this.g;g.save();g.beginPath();
    for(let i=0;i<=72;i++){const t=i/72*2*Math.PI;const p=add(c,add(scl(vecs[0],axes[0]*Math.cos(t)),scl(vecs[1],axes[1]*Math.sin(t))));const s=this.S(p);i?g.lineTo(s[0],s[1]):g.moveTo(s[0],s[1])}
    g.closePath();if(fill){g.fillStyle=fill;g.fill()}g.strokeStyle=col;g.lineWidth=w;if(dash)g.setLineDash(dash);g.stroke();g.restore()}
  covEllipse(c,S,k,col,opt){const{vals,vecs}=symEig(S);this.ellipse(c,vecs,vals.map(v=>k*Math.sqrt(Math.max(v,0))),col,opt)}
  halfplane(n,tau,fill){const b=this.box;let poly=[[b.x0,b.y0],[b.x1,b.y0],[b.x1,b.y1],[b.x0,b.y1]];const out=[];
    for(let i=0;i<poly.length;i++){const a=poly[i],c=poly[(i+1)%poly.length];const fa=dot(n,a)-tau,fc=dot(n,c)-tau;
      if(fa<=0)out.push(a);if((fa<0&&fc>0)||(fa>0&&fc<0)){const t=fa/(fa-fc);out.push(add(a,scl(sub(c,a),t)))}}
    if(out.length<3)return;const g=this.g;g.beginPath();out.forEach((p,i)=>{const s=this.S(p);i?g.lineTo(s[0],s[1]):g.moveTo(s[0],s[1])});g.closePath();g.fillStyle=fill;g.fill()}
  boundary(n,tau,col,w=1.6,dash,label){const u=unit(n);const p0=scl(u,tau/nrm(n));const t=[-u[1],u[0]];this.line(add(p0,scl(t,-30)),add(p0,scl(t,30)),col,w,dash);
    if(label){const b=this.box;let best=null;for(let k=-30;k<=30;k+=.25){const p=add(p0,scl(t,k));if(p[1]<=b.y1-.25&&p[1]>=b.y0&&p[0]>=b.x0&&p[0]<=b.x1){if(!best||p[1]>best[1])best=p}}if(best)this.text(add(best,[.12,-.05]),label,col,{font:'600 12px'})}}
  text(p,str,col,{align='left',font='500 12px',base='middle',halo=false}={}){const g=this.g,s=this.S(p);g.font=`${font} ${FONT()}`;g.textAlign=align;g.textBaseline=base;
    if(halo){g.save();g.lineWidth=4;g.lineJoin='round';g.strokeStyle=A(COL.paper,.92);g.strokeText(str,s[0],s[1]);g.restore()}g.fillStyle=col;g.fillText(str,s[0],s[1])}
  ring(p){this.dot(p,9,A(COL.paper,.85),COL.ink,2);this.dot(p,3,COL.ink)}
}

/* ================= 3D plot ================= */
class Plot3D{
  constructor(canvas,draw,{yaw=-.62,pitch=.38,span=5.2}={}){this.c=canvas;this.g=canvas.getContext('2d');this.draw=draw;this.yaw=yaw;this.pitch=pitch;this.span=span;
    let last=null;canvas.addEventListener('pointerdown',e=>{last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});
    canvas.addEventListener('pointermove',e=>{if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];last=[e.clientX,e.clientY];
      this.yaw+=dx*.009;this.pitch=clamp(this.pitch+dy*.009,-1.45,1.45);this.draw()});
    const end=()=>{last=null};canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);
    canvas.addEventListener('keydown',e=>{const k={ArrowLeft:[-.12,0],ArrowRight:[.12,0],ArrowUp:[0,-.12],ArrowDown:[0,.12]}[e.key];if(k){this.yaw+=k[0];this.pitch=clamp(this.pitch+k[1],-1.45,1.45);this.draw();e.preventDefault()}});
    watch(canvas,()=>{this.resize();this.draw()})}
  resize(){const r=this.c.getBoundingClientRect();if(!r.width)return;const dpr=window.devicePixelRatio||1;this.W=r.width;this.H=r.height;
    this.c.width=Math.round(this.W*dpr);this.c.height=Math.round(this.H*dpr);this.g.setTransform(dpr,0,0,dpr,0,0);this.s=Math.min(this.W,this.H)/(2*this.span);this.cx=this.W/2;this.cy=this.H/2;this._fk=null}
  ready(){if(!(this.W>0))this.resize();return this.W>0}
  P(p){const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw);const X=cy*p[0]-sy*p[1],Y=sy*p[0]+cy*p[1],Z=p[2];
    const cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);const Y2=cp*Y+sp*Z,Z2=-sp*Y+cp*Z;return[this.cx+this.s*X,this.cy-this.s*Z2,Y2]}
  clear(){this.g.clearRect(0,0,this.W,this.H)}
  poly(pts,col,w=1,{close=false,dash=null,fill=null}={}){const g=this.g;g.save();g.beginPath();pts.forEach((p,i)=>{const s=this.P(p);i?g.lineTo(s[0],s[1]):g.moveTo(s[0],s[1])});if(close)g.closePath();
    if(fill){g.fillStyle=fill;g.fill()}if(col){g.strokeStyle=col;g.lineWidth=w;if(dash)g.setLineDash(dash);g.stroke()}g.restore()}
  arrow(a,b,col,w=2,head=9){const s=this.P(a),t=this.P(b);arrowPx(this.g,s[0],s[1],t[0],t[1],col,w,head)}
  bbox(pts){let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;pts.forEach(p=>{const s=this.P(p);x0=Math.min(x0,s[0]);x1=Math.max(x1,s[0]);y0=Math.min(y0,s[1]);y1=Math.max(y1,s[1])});return{x0,x1,y0,y1}}
  // centre the points; once per canvas size, also zoom so they fill the canvas at every yaw, so turning never clips them
  fit(pts){this.cx=0;this.cy=0;const key=this.W+'x'+this.H;
    if(this._fk!==key){this._fk=key;const y0=this.yaw;let k=Infinity;
      for(let i=0;i<8;i++){this.yaw=y0+i*Math.PI/4;const b=this.bbox(pts);k=Math.min(k,this.W*.9/Math.max(b.x1-b.x0,1e-6),this.H*.84/Math.max(b.y1-b.y0,1e-6))}
      this.yaw=y0;this.s*=k}
    const b=this.bbox(pts);this.cx=this.W/2-(b.x0+b.x1)/2;this.cy=this.H/2-(b.y0+b.y1)/2}
  label(x,y,str,col){const g=this.g;g.fillStyle=col;g.font=`600 12px ${FONT()}`;g.textAlign='left';g.textBaseline='middle';g.fillText(str,x,y)}
  text(p,str,col,font='600 12px'){const s=this.P(p),g=this.g;g.font=`${font} ${FONT()}`;g.textAlign='center';g.textBaseline='middle';
    g.save();g.lineWidth=4;g.lineJoin='round';g.strokeStyle=A(COL.paper,.92);g.strokeText(str,s[0],s[1]);g.restore();g.fillStyle=col;g.fillText(str,s[0],s[1])}
  points(list){const pr=list.map(o=>({o,s:this.P(o.p)}));pr.sort((a,b)=>b.s[2]-a.s[2]);const g=this.g;
    for(const{o,s}of pr){g.beginPath();g.arc(s[0],s[1],o.r||2.6,0,2*Math.PI);if(o.fill){g.fillStyle=o.fill;g.fill()}if(o.stroke){g.strokeStyle=o.stroke;g.lineWidth=o.lw||1.2;g.stroke()}}}
  ellipsoid(c,vecs,axes,col){const pt=(u,v)=>add(c,add(add(scl(vecs[0],axes[0]*Math.cos(v)*Math.cos(u)),scl(vecs[1],axes[1]*Math.cos(v)*Math.sin(u))),scl(vecs[2],axes[2]*Math.sin(v))));
    for(const v of[-1,-.5,0,.5,1]){const ring=[];for(let i=0;i<=48;i++)ring.push(pt(i/48*2*Math.PI,v));this.poly(ring,col,1)}
    for(let m=0;m<6;m++){const u=m/6*Math.PI;const mer=[];for(let i=0;i<=48;i++)mer.push(pt(u,-Math.PI/2+i/48*2*Math.PI));this.poly(mer,col,1)}}
  // the kσ region of a 3D Gaussian as this view shows it: under this projection an ellipsoid's outline is the ellipse
  // whose covariance is J S Jᵀ, J being the screen offsets of the three unit axes
  blob(c,S,k,col,{fill=null,w=1.4,dash=null}={}){const o=this.P(c),J=[[1,0,0],[0,1,0],[0,0,1]].map(e=>{const q=this.P(add(c,e));return[q[0]-o[0],q[1]-o[1]]});
    const M=[[0,0],[0,0]];for(let i=0;i<3;i++)for(let j=0;j<3;j++){M[0][0]+=S[i][j]*J[i][0]*J[j][0];M[0][1]+=S[i][j]*J[i][0]*J[j][1];M[1][1]+=S[i][j]*J[i][1]*J[j][1]}M[1][0]=M[0][1];
    const{vals,vecs}=symEig(M),a=vals.map(v=>k*Math.sqrt(Math.max(v,0))),g=this.g;g.save();g.beginPath();
    for(let t=0;t<=72;t++){const th=t/72*2*Math.PI,x=o[0]+a[0]*Math.cos(th)*vecs[0][0]+a[1]*Math.sin(th)*vecs[1][0],y=o[1]+a[0]*Math.cos(th)*vecs[0][1]+a[1]*Math.sin(th)*vecs[1][1];t?g.lineTo(x,y):g.moveTo(x,y)}
    g.closePath();if(fill){g.fillStyle=fill;g.fill()}g.strokeStyle=col;g.lineWidth=w;if(dash)g.setLineDash(dash);g.stroke();g.restore()}
  // an orientation key pinned to the bottom-left corner: each arrow is a unit axis as this view projects it
  axes3(o,len,labels){const g=this.g,ox=34,oy=this.H-34,L=26,p0=this.P([0,0,0]);
    [[1,0,0],[0,1,0],[0,0,1]].forEach((v,i)=>{const q=this.P(v),dx=(q[0]-p0[0])/this.s,dy=(q[1]-p0[1])/this.s;
      arrowPx(g,ox,oy,ox+L*dx,oy+L*dy,A(COL.muted,.85),1.2,5);g.fillStyle=COL.muted;g.font=`500 11px ${FONT()}`;
      // each label hangs off its arrow's tip on the side the arrow points to, so three labels don't pile up
      g.textAlign=dx>.25?'left':dx<-.25?'right':'center';g.textBaseline=dy>.25?'top':dy<-.25?'bottom':'middle';g.fillText(labels[i],ox+(L+5)*dx,oy+(L+5)*dy)})}
}

/* ================= small line chart (real data and reported numbers) ================= */
// spec: {x:{min,max,log,ticks:[..],label,fmt}, y:{min,max,ticks:[..],label,fmt}, series:[{pts:[[x,y]..],col,w,dash,marker,label}], vline:x, notes:[{x,y,text,col,align}]}
function lineChart(c,spec){const g=c.getContext('2d');const r=fitCanvas(c,g);if(!r)return;const W=r.width,H=r.height;const pad={l:44,r:14,t:spec.title?26:12,b:34};
  const fx=v=>spec.x.log?Math.log10(v):v;const X=v=>pad.l+(fx(v)-fx(spec.x.min))/(fx(spec.x.max)-fx(spec.x.min))*(W-pad.l-pad.r);
  const Y=v=>H-pad.b-(v-spec.y.min)/(spec.y.max-spec.y.min)*(H-pad.t-pad.b);const font=FONT();
  g.font=`500 11px ${font}`;g.lineWidth=1;
  spec.y.ticks.forEach(t=>{g.strokeStyle=COL.grid;g.beginPath();g.moveTo(pad.l,Y(t));g.lineTo(W-pad.r,Y(t));g.stroke();g.fillStyle=COL.muted;g.textAlign='right';g.textBaseline='middle';g.fillText(spec.y.fmt?spec.y.fmt(t):String(t),pad.l-6,Y(t))});
  spec.x.ticks.forEach(t=>{g.strokeStyle=COL.grid;g.beginPath();g.moveTo(X(t),pad.t);g.lineTo(X(t),H-pad.b);g.stroke();g.fillStyle=COL.muted;g.textAlign='center';g.textBaseline='top';g.fillText(spec.x.fmt?spec.x.fmt(t):String(t),X(t),H-pad.b+5)});
  g.strokeStyle=COL.hair;g.beginPath();g.moveTo(pad.l,H-pad.b);g.lineTo(W-pad.r,H-pad.b);g.stroke();
  if(spec.x.label){g.fillStyle=COL.muted;g.textAlign='center';g.textBaseline='bottom';g.fillText(spec.x.label,pad.l+(W-pad.l-pad.r)/2,H-1)}
  if(spec.y.label){g.save();g.translate(11,pad.t+(H-pad.t-pad.b)/2);g.rotate(-Math.PI/2);g.textAlign='center';g.textBaseline='middle';g.fillStyle=COL.muted;g.fillText(spec.y.label,0,0);g.restore()}
  if(spec.title){g.fillStyle=COL.ink;g.font=`600 12px ${font}`;g.textAlign='left';g.textBaseline='top';g.fillText(spec.title,pad.l,4);g.font=`500 11px ${font}`}
  if(spec.band){g.fillStyle=spec.band.fill;g.fillRect(X(spec.band.x0),pad.t,X(spec.band.x1)-X(spec.band.x0),H-pad.t-pad.b)}
  if(spec.vline!=null){g.strokeStyle=A(COL.navy,.7);g.setLineDash([4,3]);g.beginPath();g.moveTo(X(spec.vline),pad.t);g.lineTo(X(spec.vline),H-pad.b);g.stroke();g.setLineDash([])}
  spec.series.forEach(s=>{const pts=s.pts.filter(p=>p[1]!=null);if(!pts.length)return;g.save();g.strokeStyle=s.col;g.lineWidth=s.w||2;if(s.dash)g.setLineDash(s.dash);g.beginPath();
    pts.forEach((p,i)=>{const x=X(p[0]),y=Y(p[1]);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();g.setLineDash([]);
    if(s.marker)pts.forEach(p=>{g.beginPath();g.arc(X(p[0]),Y(p[1]),s.marker,0,2*Math.PI);g.fillStyle=s.hollow?COL.panel:s.col;g.fill();g.strokeStyle=s.col;g.lineWidth=1.5;g.stroke()});
    if(s.label){const p=pts[pts.length-1];g.fillStyle=s.col;g.font=`600 11px ${font}`;g.textAlign='right';g.textBaseline=s.labelBase||'bottom';g.fillText(s.label,X(p[0])-6,Y(p[1])+(s.labelDy||-4))}
    g.restore()});
  (spec.notes||[]).forEach(n=>{g.fillStyle=n.col||COL.muted;g.font=`${n.font||'500 11px'} ${font}`;g.textAlign=n.align||'left';g.textBaseline='middle';g.fillText(n.text,X(n.x),Y(n.y))});
  return{X,Y,pad,W,H}}

/* ================= shared 2D world (toy) ================= */
// Persona and assistant token clouds share a content direction tilted 32 degrees, so content leaks onto the persona axis.
// The persona's own style spread runs at -28 degrees. Drifted tokens are persona tokens pushed part way back to the assistant.
function makeWorld2D(seed,N=150){const r=mulberry32(seed);const muA=[-2,0],muP=[2,0],dTrue=sub(muP,muA);
  const th=32*Math.PI/180,u1=[Math.cos(th),Math.sin(th)],u2=[-Math.sin(th),Math.cos(th)];
  const ph=-28*Math.PI/180,w1=[Math.cos(ph),Math.sin(ph)],w2=[-Math.sin(ph),Math.cos(ph)];
  const content=()=>add(scl(u1,1.25*randn(r)),scl(u2,.22*randn(r)));
  const styleP=()=>add(scl(w1,.42*randn(r)),scl(w2,.17*randn(r)));
  const styleA=()=>[.2*randn(r),.2*randn(r)];
  const P=[],Ap=[],Dr=[];
  for(let i=0;i<N;i++){const cz=content();P.push(add(add(muP,cz),styleP()));Ap.push(add(add(muA,cz),styleA()))}
  for(let i=0;i<N;i++){const t=.3+.55*r();Dr.push(sub(add(add(muP,content()),styleP()),scl(dTrue,t)))}
  const mP=meanOf(P),mA=meanOf(Ap),d=sub(mP,mA),vhat=unit(d);
  const SigP=covOf(P),SigA=covOf(Ap),SigW=mscl(madd(SigP,SigA),.5);
  const D=P.map((p,i)=>sub(p,Ap[i]));
  return{P,A:Ap,Dr,mP,mA,d,vhat,SigP,SigA,SigW,D,u1,w1}}
const W2=makeWorld2D(11);
const ZERO=[0,-3.4];             // where the zero vector sits in this picture, for moves that keep a token's length
const SPi_W2=inv(W2.SigP);
const sCoord=h=>dot(sub(h,W2.mA),W2.d)/dot(W2.d,W2.d);
const inside2s=h=>{const q=sub(h,W2.mP);return dot(q,mv(SPi_W2,q))<=4};
function metrics(edit){const Pe=W2.P.map(edit),De=W2.Dr.map(edit);
  const sb=avg(W2.Dr.map(sCoord)),sa=avg(De.map(e=>sCoord(e.h)));
  const spread=Math.sqrt(avg(W2.P.map(h=>{const q=sub(h,W2.mP);return dot(q,q)})));
  const dist=Math.sqrt(avg(Pe.map((e,i)=>{const q=sub(e.h,W2.P[i]);return dot(q,q)})))/spread;
  const normCh=avg(De.map((e,i)=>nrm(sub(e.h,ZERO))/nrm(sub(W2.Dr[i],ZERO))-1));
  return{restore:(sa-sb)/(1-sb),dist,caught:avg(De.map(e=>e.on?1:0)),ftr:avg(Pe.map(e=>e.on?1:0)),normCh,landed:avg(De.map(e=>inside2s(e.h)?1:0)),Pe,De}}
const BOX2={x0:-5.6,x1:5.8,y0:-3.7,y1:3.7};
function drawAxis(pl){pl.line(add(W2.mA,scl(W2.d,-.45)),add(W2.mP,scl(W2.d,.32)),A(COL.navy,.55),1.2,[5,4]);
  pl.dot(W2.mA,4.5,COL.assist,COL.paper,1.5);pl.dot(W2.mP,4.5,COL.persona,COL.paper,1.5)}
// drawn last, over the dots, so the two labels stay readable
function axisLabels(pl){[[W2.mA,'assistant (the default)',COL.assist],[W2.mP,'persona',COL.persona]].forEach(([p,t,c])=>{
  pl.dot(p,4.5,c,COL.paper,1.5);pl.text(add(p,[0,.5]),t,c,{align:'center',font:'600 13px',halo:true})})}

/* ================= one picture for every toy move ================= */
// The two reference clouds are shaded 2σ regions, not dots. Blue is where the assistant's tokens sit with no persona
// prompt, the model's default; teal is where the persona's sit. The only dots are the tokens being steered, all one
// colour, so a token's position alone says whether it is in character. Two framings:
//   set:  the steered tokens start as the assistant, and the move should carry them into the persona region;
//   hold: they are one prompted persona conversation, 28 in character and 12 slid back toward the assistant,
//         and a good hold moves only those 12.
// Readouts still use the full 150-token clouds, so the numbers don't depend on which 40 are drawn.
const TOK={set:W2.A.slice(0,40),hold:W2.P.slice(0,28).concat(W2.Dr.slice(0,12))};
const shift=(a,b)=>nrm(sub(a,b))/nrm(W2.d);
function drawRegions(pl){pl.covEllipse(W2.mA,W2.SigA,2,A(COL.assist,.8),{w:1.5,fill:A(COL.assist,.1)});
  pl.covEllipse(W2.mP,W2.SigP,2,A(COL.persona,.85),{w:1.5,fill:A(COL.persona,.12)})}
// edit: h -> {h, on}; framing: 'set' or 'hold'. In 'set' a dashed orange outline marks where the whole cloud ends up.
function drawSteer(pl,edit,framing){const from=TOK[framing],to=from.map(h=>edit(h).h);
  if(framing==='set'){const all=W2.A.map(h=>edit(h).h);if(all.some((h,i)=>shift(h,W2.A[i])>.01))pl.covEllipse(meanOf(all),covOf(all),2,COL.edit,{w:1.6,dash:[6,4]})}
  from.forEach((p,i)=>{if(shift(to[i],p)>.01){pl.dot(p,2.6,null,A(COL.muted,.8),1);pl.arrow(p,to[i],A(COL.edit,.8),1.4,6)}});
  to.forEach(h=>pl.dot(h,3.3,A(COL.ink,.88)));axisLabels(pl)}
// readouts for the 'set' framing, over all 150 assistant tokens
function setMetrics(edit){const E=W2.A.map(edit);return{pos:avg(E.map(e=>sCoord(e.h))),landed:avg(E.map(e=>inside2s(e.h)?1:0)),
  normCh:avg(E.map((e,i)=>nrm(sub(e.h,ZERO))/nrm(sub(W2.A[i],ZERO))-1)),edited:avg(E.map(e=>e.on?1:0))}}
const FRAMES=[['set','Get into character'],['hold','Stay in character']];
const FRAME_TXT={set:'Get into character: the dots start as the assistant, the default with no persona prompt.',
  hold:'Stay in character: one prompted conversation, where a few tokens slid back toward the assistant.'};
const pctSigned=x=>(x>=0?'+':'')+(100*x).toFixed(0)+'%';
linkRefs();
