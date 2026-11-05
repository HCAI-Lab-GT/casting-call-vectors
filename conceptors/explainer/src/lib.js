/* ================= lib: 2x2 math, seeded data, canvas drawing, charts, controls ================= */
const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
const deg = Math.PI/180;

/* 2×2 linear algebra; matrices are [a,b,c,d] = [[a,b],[c,d]] */
const I2=[1,0,0,1];
const mm=(A,B)=>[A[0]*B[0]+A[1]*B[2],A[0]*B[1]+A[1]*B[3],A[2]*B[0]+A[3]*B[2],A[2]*B[1]+A[3]*B[3]];
const mv=(A,v)=>[A[0]*v[0]+A[1]*v[1],A[2]*v[0]+A[3]*v[1]];
const madd=(A,B)=>A.map((x,i)=>x+B[i]);
const msub=(A,B)=>A.map((x,i)=>x-B[i]);
const msc=(A,s)=>A.map(x=>x*s);
const T=A=>[A[0],A[2],A[1],A[3]];
const minv=A=>{const d=A[0]*A[3]-A[1]*A[2];return [A[3]/d,-A[1]/d,-A[2]/d,A[0]/d];};
const sym=A=>{const b=(A[1]+A[2])/2;return [A[0],b,b,A[3]];};
const rot=t=>[Math.cos(t),-Math.sin(t),Math.sin(t),Math.cos(t)];
const diag=(a,b)=>[a,0,0,b];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1];
const add=(a,b)=>[a[0]+b[0],a[1]+b[1]];
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1]];
const sc=(a,s)=>[a[0]*s,a[1]*s];
const lerp=(a,b,t)=>[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t];
const norm=a=>Math.hypot(a[0],a[1]);
const unit=a=>{const n=norm(a)||1;return [a[0]/n,a[1]/n];};
const perp=a=>[-a[1],a[0]];
function eigSym(A){
  const a=A[0],b=(A[1]+A[2])/2,d=A[3];
  const tr=a+d,det=a*d-b*b,disc=Math.sqrt(Math.max(0,tr*tr/4-det));
  const l1=tr/2+disc,l2=tr/2-disc;
  let v=Math.abs(b)>1e-12?[l1-d,b]:(a>=d?[1,0]:[0,1]);
  const n=Math.hypot(v[0],v[1]);v=[v[0]/n,v[1]/n];
  if(v[0]<0||(Math.abs(v[0])<1e-12&&v[1]<0))v=[-v[0],-v[1]];
  return {l:[l1,l2],u1:v,u2:[-v[1],v[0]]};
}
function mfun(S,fn){const e=eigSym(S);const U=[e.u1[0],e.u2[0],e.u1[1],e.u2[1]];return sym(mm(mm(U,diag(fn(Math.max(1e-12,e.l[0])),fn(Math.max(1e-12,e.l[1])))),T(U)));}
const msqrt=S=>mfun(S,Math.sqrt);
const misqrt=S=>mfun(S,x=>1/Math.sqrt(x));
const mean=P=>{let x=0,y=0;for(const p of P){x+=p[0];y+=p[1];}return [x/P.length,y/P.length];};
function corr(P,c){c=c||[0,0];let s0=0,s1=0,s3=0;for(const p of P){const x=p[0]-c[0],y=p[1]-c[1];s0+=x*x;s1+=x*y;s3+=y*y;}const n=P.length;return [s0/n,s1/n,s1/n,s3/n];}
function conceptor(R,alpha){const a=1/(alpha*alpha);return sym(mm(R,minv(madd(R,msc(I2,a)))));}
const keepOf=(l,alpha)=>l/(l+1/(alpha*alpha));
const AND=(A,B)=>sym(minv(msub(madd(minv(A),minv(B)),I2)));
const NOT=A=>msub(I2,A);
const sigmoid=x=>1/(1+Math.exp(-x));

/* seeded data: whitened so each cloud's spread is exactly what we set */
function rng(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
function gaussians(n,seed){
  const r=rng(seed),Z=[];
  for(let i=0;i<n;i++){const u=Math.max(1e-9,r()),v=r(),m=Math.sqrt(-2*Math.log(u));Z.push([m*Math.cos(2*Math.PI*v),m*Math.sin(2*Math.PI*v)]);}
  const mu=mean(Z);Z.forEach(z=>{z[0]-=mu[0];z[1]-=mu[1];});
  const e=eigSym(corr(Z));
  const U=[e.u1[0],e.u2[0],e.u1[1],e.u2[1]];
  const W=mm(mm(U,diag(1/Math.sqrt(e.l[0]),1/Math.sqrt(e.l[1]))),T(U));
  return Z.map(z=>mv(W,z));
}
const cloud=(Z,mu,L)=>Z.map(z=>add(mu,mv(L,z)));

/* colors come from the theme tokens, re-read when the theme changes */
const COL={};
function readColors(){
  const cs=getComputedStyle(document.documentElement);
  const g=k=>cs.getPropertyValue('--'+k).trim();
  Object.assign(COL,{paper:g('paper'),panel:g('panel'),ink:g('ink'),muted:g('muted'),hair:g('hair'),grid:g('grid'),
    pirate:g('pirate'),assist:g('assist'),orange:g('orange'),other:g('other'),none:g('none'),probe:g('probe'),
    wash:g('wash'),accent:g('accent-ink'),warn:g('warn-ink')});
}
readColors();
function rgba(hex,a){
  const h=(hex||'#888').replace('#','');
  if(h.length!==6)return hex;
  return 'rgba('+parseInt(h.slice(0,2),16)+','+parseInt(h.slice(2,4),16)+','+parseInt(h.slice(4,6),16)+','+a+')';
}
const MONO='"IBM Plex Mono",ui-monospace,Menlo,monospace';

/* ---------- canvas plumbing ---------- */
const figs=[];
function Canvas(id,aspect,paint,opt){
  opt=opt||{};
  const cv=document.getElementById(id);
  const f={cv,ctx:null,w:0,h:0,lastW:-1,lastH:-1};
  f.resize=()=>{
    const w=cv.clientWidth||600;
    const h=Math.round(Math.max(opt.minH||0,Math.min(w*aspect,opt.maxH||window.FIG_MAX_H||500)));  // the deck raises the cap
    if(w===f.lastW&&h===f.lastH&&f.ctx)return;
    f.lastW=w;f.lastH=h;
    const dpr=Math.min(2,window.devicePixelRatio||1);
    cv.style.height=h+'px';cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);
    f.ctx=cv.getContext('2d');f.ctx.setTransform(dpr,0,0,dpr,0,0);f.w=w;f.h=h;
  };
  f.draw=()=>{
    f.resize();const c=f.ctx;
    c.clearRect(0,0,f.w,f.h);c.fillStyle=COL.panel;c.fillRect(0,0,f.w,f.h);
    paint(c,f.w,f.h,f);
  };
  if(window.ResizeObserver)new ResizeObserver(()=>f.draw()).observe(cv.parentElement);
  figs.push(f);
  return f;
}
function makeView(w,h,b,pad){
  pad=pad==null?14:pad;
  const bw=b[1]-b[0],bh=b[3]-b[2];
  const s=Math.min((w-2*pad)/bw,(h-2*pad)/bh);
  const ox=(w-s*bw)/2-s*b[0], oy=(h+s*bh)/2+s*b[2];
  return {s,X:x=>ox+s*x,Y:y=>oy-s*y,inv:(px,py)=>[(px-ox)/s,(oy-py)/s]};
}
/* a figure in "activation space" coordinates, with a light square grid */
function Fig(id,aspect,bounds,paint,opt){
  return Canvas(id,aspect,(c,w,h,f)=>{
    const b=typeof bounds==='function'?bounds():bounds;
    f.V=makeView(w,h,b);
    if(!(opt&&opt.noGrid))grid(c,f.V,w,h);
    paint(c,f.V,f);
  },opt);
}
function grid(c,V,w,h){
  const [x0,y1]=V.inv(0,0),[x1,y0]=V.inv(w,h),st=0.25;
  c.lineWidth=1;c.strokeStyle=COL.grid;c.beginPath();
  for(let x=Math.ceil(x0/st)*st;x<=x1;x+=st){const px=Math.round(V.X(x))+.5;c.moveTo(px,0);c.lineTo(px,h);}
  for(let y=Math.ceil(y0/st)*st;y<=y1;y+=st){const py=Math.round(V.Y(y))+.5;c.moveTo(0,py);c.lineTo(w,py);}
  c.stroke();
  c.strokeStyle=COL.hair;c.beginPath();
  const ax=Math.round(V.X(0))+.5,ay=Math.round(V.Y(0))+.5;
  c.moveTo(ax,0);c.lineTo(ax,h);c.moveTo(0,ay);c.lineTo(w,ay);c.stroke();
}
function dots(c,V,P,color,r,alpha,hollow){
  c.globalAlpha=alpha==null?1:alpha;c.beginPath();
  for(const p of P){const x=V.X(p[0]),y=V.Y(p[1]);c.moveTo(x+r,y);c.arc(x,y,r,0,2*Math.PI);}
  if(hollow){c.lineWidth=1.1;c.strokeStyle=color;c.stroke();}else{c.fillStyle=color;c.fill();}
  c.globalAlpha=1;
}
function poly(c,V,pts,o){
  o=o||{};c.beginPath();
  pts.forEach((p,i)=>{const x=V.X(p[0]),y=V.Y(p[1]);i?c.lineTo(x,y):c.moveTo(x,y);});
  if(o.closed!==false)c.closePath();
  if(o.fill){c.globalAlpha=o.fillAlpha==null?1:o.fillAlpha;c.fillStyle=o.fill;c.fill();c.globalAlpha=1;}
  if(o.stroke){c.setLineDash(o.dash||[]);c.lineWidth=o.width||1.5;c.strokeStyle=o.stroke;c.globalAlpha=o.alpha==null?1:o.alpha;c.lineJoin='round';c.stroke();c.globalAlpha=1;c.setLineDash([]);}
}
function seg(c,V,a,b,color,width,dash,alpha){
  c.beginPath();c.moveTo(V.X(a[0]),V.Y(a[1]));c.lineTo(V.X(b[0]),V.Y(b[1]));
  c.setLineDash(dash||[]);c.lineWidth=width||1;c.strokeStyle=color;c.globalAlpha=alpha==null?1:alpha;c.stroke();c.globalAlpha=1;c.setLineDash([]);
}
function arrow(c,V,a,b,color,width,head,alpha){
  const ax=V.X(a[0]),ay=V.Y(a[1]),bx=V.X(b[0]),by=V.Y(b[1]);
  const L=Math.hypot(bx-ax,by-ay);if(L<2)return;
  const ux=(bx-ax)/L,uy=(by-ay)/L,hd=Math.min(head||9,L*.45);
  c.globalAlpha=alpha==null?1:alpha;c.strokeStyle=color;c.fillStyle=color;c.lineWidth=width||2;
  c.beginPath();c.moveTo(ax,ay);c.lineTo(bx-ux*hd*.7,by-uy*hd*.7);c.stroke();
  c.beginPath();c.moveTo(bx,by);c.lineTo(bx-ux*hd-uy*hd*.45,by-uy*hd+ux*hd*.45);c.lineTo(bx-ux*hd+uy*hd*.45,by-uy*hd-ux*hd*.45);c.closePath();c.fill();
  c.globalAlpha=1;
}
function text(c,s,x,y,o){
  o=o||{};
  c.font=(o.weight||'500')+' '+(o.size||11)+'px '+MONO;c.textAlign=o.align||'left';c.textBaseline=o.base||'middle';
  if(o.halo!==false){c.lineJoin='round';c.lineWidth=4;c.strokeStyle=COL.panel;c.strokeText(s,x,y);}
  c.fillStyle=o.color||COL.muted;c.fillText(s,x,y);
}
function label(c,V,s,p,o){o=o||{};text(c,s,V.X(p[0])+(o.dx||0),V.Y(p[1])+(o.dy||0),o);}
function circlePts(center,r,n){n=n||120;const o=[];for(let i=0;i<n;i++){const t=i/n*2*Math.PI;o.push([center[0]+r*Math.cos(t),center[1]+r*Math.sin(t)]);}return o;}
function imgCircle(M,center,r,n){center=center||[0,0];r=r||1;n=n||120;const o=[];for(let i=0;i<n;i++){const t=i/n*2*Math.PI;const q=mv(M,[r*Math.cos(t),r*Math.sin(t)]);o.push([center[0]+q[0],center[1]+q[1]]);}return o;}
function shapeEll(S,center,k){const e=eigSym(S),a=k*Math.sqrt(Math.max(0,e.l[0])),b=k*Math.sqrt(Math.max(0,e.l[1]));const o=[];for(let i=0;i<120;i++){const t=i/120*2*Math.PI;o.push([center[0]+a*Math.cos(t)*e.u1[0]+b*Math.sin(t)*e.u2[0],center[1]+a*Math.cos(t)*e.u1[1]+b*Math.sin(t)*e.u2[1]]);}return o;}
/* point marks with a surface ring so overlapping marks stay separable */
function mark(c,x,y,shape,r,color,hollow,ring){
  c.beginPath();
  if(shape==='tri'){const k=r*1.3;c.moveTo(x,y-k);c.lineTo(x+k*.95,y+k*.72);c.lineTo(x-k*.95,y+k*.72);c.closePath();}
  else if(shape==='dia'){const k=r*1.3;c.moveTo(x,y-k);c.lineTo(x+k,y);c.lineTo(x,y+k);c.lineTo(x-k,y);c.closePath();}
  else if(shape==='sq'){c.rect(x-r*.88,y-r*.88,r*1.76,r*1.76);}
  else c.arc(x,y,r,0,2*Math.PI);
  if(hollow){if(ring){c.lineWidth=3.6;c.strokeStyle=COL.panel;c.stroke();}c.lineWidth=1.6;c.strokeStyle=color;c.stroke();}
  else{if(ring){c.lineWidth=2;c.strokeStyle=COL.panel;c.stroke();}c.fillStyle=color;c.fill();}
}
function shapeSVG(shape,colorVar,hollow){
  let d;
  if(shape==='tri')d='<path d="M6 1.4 L10.8 10.2 L1.2 10.2 Z"';
  else if(shape==='dia')d='<path d="M6 .8 L11.2 6 L6 11.2 L.8 6 Z"';
  else if(shape==='sq')d='<rect x="2" y="2" width="8" height="8"';
  else d='<circle cx="6" cy="6" r="4.3"';
  const st=hollow?'style="fill:none;stroke:'+colorVar+';stroke-width:1.7"':'style="fill:'+colorVar+'"';
  return '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">'+d+' '+st+'/></svg>';
}

/* ---------- small DOM helpers ---------- */
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
const f2=x=>(x<0?'−':'')+Math.abs(x).toFixed(2);
const pct=x=>Math.round(x*100)+'%';
const pct1=x=>(x*100).toFixed(1)+'%';
const A_MIN=0.5,A_MAX=50;
const sliderToAlpha=v=>A_MIN*Math.pow(A_MAX/A_MIN,v/1000);
const alphaToSlider=a=>Math.round(1000*Math.log(a/A_MIN)/Math.log(A_MAX/A_MIN));
const fmtA=a=>a<1?a.toFixed(2):a<10?a.toFixed(1):a.toFixed(0);
function out(id,txt){const o=document.querySelector('output[for="'+id+'"]');if(o)o.textContent=txt;}
function onRange(id,fn){const el=$(id);el.addEventListener('input',()=>fn(+el.value));return el;}
function onSeg(id,fn){
  const box=$(id);
  box.addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(!b)return;setSeg(id,b.dataset.v);fn(b.dataset.v);});
}
function setSeg(id,v){$(id).querySelectorAll('button[data-v]').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.v===v)));}
function rows(el,list){el.innerHTML=list.map(r=>'<div><span class="k">'+r[0]+'</span><span>'+r[1]+'</span></div>').join('');}
function bars(el,list,max,one){
  el.innerHTML=list.map(r=>{
    const w=Math.max(0,Math.min(1,r[1]/max))*100;
    return '<div class="barrow"><span>'+r[0]+'</span><span class="bar"><i style="width:'+w.toFixed(1)+'%;background:'+(r[2]||'var(--pirate)')+'"></i>'+(one?'<span class="one" style="left:'+(100/max).toFixed(2)+'%"></span>':'')+'</span><span class="v">'+r[3]+'</span></div>';
  }).join('');
}
function tiles(el,list){
  el.innerHTML=list.map(t=>'<div class="tile"><span class="lab">'+t[0]+'</span><span class="val">'+t[1]+'</span>'+(t[2]?'<span class="sub">'+t[2]+'</span>':'')+'</div>').join('');
}
const ease=t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
function animate(st,fig){
  if(reduce){st.t=1;fig.draw();return;}
  st.t=0;const t0=performance.now();st.token=(st.token||0)+1;const tok=st.token;
  const step=now=>{if(tok!==st.token)return;st.t=Math.min(1,(now-t0)/650);fig.draw();if(st.t<1)requestAnimationFrame(step);};
  requestAnimationFrame(step);
}

/* ---------- tooltip ---------- */
function showTip(html,x,y){
  const t=$('tip');t.innerHTML=html;t.hidden=false;
  const r=t.getBoundingClientRect();
  let L=x+14,Tp=y+14;
  if(L+r.width>innerWidth-8)L=x-r.width-14;
  if(Tp+r.height>innerHeight-8)Tp=y-r.height-14;
  t.style.left=Math.max(8,L)+'px';t.style.top=Math.max(8,Tp)+'px';
}
function hideTip(){const t=$('tip');if(t)t.hidden=true;}
window.addEventListener('scroll',hideTip,{passive:true});
function hoverable(cv,move,leave,click){
  const at=e=>{const r=cv.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];};
  cv.addEventListener('pointermove',e=>{const p=at(e);move(p[0],p[1],e);});
  cv.addEventListener('pointerdown',e=>{if(e.pointerType!=='mouse'){const p=at(e);move(p[0],p[1],e);}});
  cv.addEventListener('pointerleave',e=>{if(e.pointerType==='mouse'&&leave)leave();});
  if(click)cv.addEventListener('click',e=>{const p=at(e);click(p[0],p[1],e);});
}
/* draggable probe point on a Fig; keyboard arrows move it when the canvas has focus */
function draggable(fig,getP,setP,bounds){
  const cv=fig.cv;let on=false;
  const clamp=p=>[Math.max(bounds[0]+.05,Math.min(bounds[1]-.05,p[0])),Math.max(bounds[2]+.05,Math.min(bounds[3]-.05,p[1]))];
  const at=e=>{const r=cv.getBoundingClientRect();return fig.V.inv(e.clientX-r.left,e.clientY-r.top);};
  cv.addEventListener('pointerdown',e=>{on=true;try{cv.setPointerCapture(e.pointerId);}catch(_){}setP(clamp(at(e)));});
  cv.addEventListener('pointermove',e=>{if(on)setP(clamp(at(e)));});
  cv.addEventListener('pointerup',()=>{on=false;});
  cv.addEventListener('pointercancel',()=>{on=false;});
  cv.addEventListener('keydown',e=>{const d=.05,k={ArrowLeft:[-d,0],ArrowRight:[d,0],ArrowUp:[0,d],ArrowDown:[0,-d]}[e.key];if(!k)return;e.preventDefault();setP(clamp(add(getP(),k)));});
}

/* ---------- axis ticks ---------- */
function ticks(a,b,n){
  const span=b-a;if(!(span>0))return [a];
  const raw=span/(n||5),p=Math.pow(10,Math.floor(Math.log10(raw))),m=raw/p;
  const st=(m<1.5?1:m<3?2:m<7?5:10)*p,o=[];
  for(let v=Math.ceil(a/st-1e-9)*st;v<=b+1e-9;v+=st)o.push(+v.toFixed(10));
  return o;
}

/* ---------- line chart: one y-axis, 2px lines, crosshair tooltip, direct labels at line ends ---------- */
function LineChart(id,o){
  const ch={hover:-1};
  const f=Canvas(id,o.aspect||.62,(c,w,h)=>{
    const S=o.series(),xs=o.xs(),yr=o.yRange();
    const M={l:o.ml||42,r:o.mr==null?88:o.mr,t:o.mt||14,b:o.mb||30};
    const x0=o.xMin!=null?o.xMin:xs[0],x1=o.xMax!=null?o.xMax:xs[xs.length-1];
    const X=x=>M.l+(x-x0)/((x1-x0)||1)*(w-M.l-M.r);
    const cl=v=>Math.max(yr[0],Math.min(yr[1],v));
    const Y=y=>M.t+(1-(cl(y)-yr[0])/((yr[1]-yr[0])||1))*(h-M.t-M.b);
    Object.assign(ch,{X,Y,M,w,h,xs});
    c.font='10px '+MONO;c.textBaseline='middle';
    (o.bands?o.bands():[]).forEach(b=>{
      c.globalAlpha=b.alpha==null?1:b.alpha;c.fillStyle=b.color||COL.grid;
      c.fillRect(X(b.x0),M.t,Math.max(1,X(b.x1)-X(b.x0)),h-M.t-M.b);c.globalAlpha=1;
      if(b.label){const right=X(b.x0)>(M.l+w-M.r)/2+40,ly=M.t+8+12*(b.row||0);text(c,b.label,right?X(b.x0)-4:X(b.x1)+4,ly,{size:10,align:right?'right':'left'});}
    });
    const yt=o.yTicks?o.yTicks(yr):ticks(yr[0],yr[1],4);
    c.strokeStyle=COL.grid;c.lineWidth=1;c.beginPath();
    yt.forEach(v=>{const y=Math.round(Y(v))+.5;c.moveTo(M.l,y);c.lineTo(w-M.r,y);});c.stroke();
    c.font='10px '+MONO;c.fillStyle=COL.muted;c.textAlign='right';c.textBaseline='middle';
    yt.forEach(v=>c.fillText(o.yFmt?o.yFmt(v):String(v),M.l-6,Y(v)));
    const xt=o.xTicks?o.xTicks(xs):xs;
    c.textAlign='center';c.textBaseline='top';
    xt.forEach(v=>c.fillText(o.xFmt?o.xFmt(v):String(v),X(v),h-M.b+7));
    c.strokeStyle=COL.hair;c.beginPath();c.moveTo(M.l,Math.round(h-M.b)+.5);c.lineTo(w-M.r,Math.round(h-M.b)+.5);c.stroke();
    if(o.xTitle){c.textAlign='right';c.textBaseline='bottom';c.fillStyle=COL.muted;c.fillText(o.xTitle,w-M.r,h-1);}
    if(o.yTitle){c.textAlign='left';c.textBaseline='top';c.fillStyle=COL.muted;c.fillText(o.yTitle,M.l+2,1);}
    (o.refs?o.refs():[]).forEach(r=>{
      const y=Math.round(Y(r.y))+.5;c.setLineDash([4,4]);c.strokeStyle=COL.muted;c.lineWidth=1;c.globalAlpha=.85;
      c.beginPath();c.moveTo(M.l,y);c.lineTo(w-M.r,y);c.stroke();c.setLineDash([]);c.globalAlpha=1;
      if(r.label)text(c,r.label,r.align==='right'?w-M.r-5:M.l+5,y-8,{size:10,align:r.align||'left'});
    });
    if(o.selected){const si=o.selected();if(si!=null&&si>=0){const x=Math.round(X(xs[si]))+.5;c.strokeStyle=COL.ink;c.lineWidth=1;c.setLineDash([2,3]);c.beginPath();c.moveTo(x,M.t);c.lineTo(x,h-M.b);c.stroke();c.setLineDash([]);}}
    S.forEach(s=>{
      c.strokeStyle=s.color;c.lineWidth=s.width||2;c.lineJoin='round';c.lineCap='round';c.setLineDash(s.dash||[]);
      c.beginPath();let on=false;
      s.values.forEach((v,i)=>{if(v==null||!isFinite(v)){on=false;return;}const x=X(xs[i]),y=Y(v);on?c.lineTo(x,y):c.moveTo(x,y);on=true;});
      c.stroke();c.setLineDash([]);
      if(s.dots)s.values.forEach((v,i)=>{if(v!=null)mark(c,X(xs[i]),Y(v),'o',3.2,s.color,false,true);});
      if(s.rings)s.rings.forEach(i=>{const v=s.values[i];if(v==null)return;c.beginPath();c.arc(X(xs[i]),Y(v),6.5,0,2*Math.PI);c.lineWidth=4;c.strokeStyle=COL.panel;c.stroke();c.lineWidth=1.8;c.strokeStyle=s.color;c.stroke();});
    });
    if(ch.hover>=0&&ch.hover<xs.length){
      const x=Math.round(X(xs[ch.hover]))+.5;c.strokeStyle=COL.muted;c.lineWidth=1;c.beginPath();c.moveTo(x,M.t);c.lineTo(x,h-M.b);c.stroke();
      S.forEach(s=>{const v=s.values[ch.hover];if(v!=null)mark(c,X(xs[ch.hover]),Y(v),'o',4,s.color,false,true);});
    }
    if(o.direct!==false&&M.r>30){
      const L=[];
      S.forEach(s=>{let i=s.values.length-1;while(i>=0&&s.values[i]==null)i--;if(i>=0)L.push({x:X(xs[i]),y:Y(s.values[i]),s});});
      L.sort((a,b)=>a.y-b.y);
      for(let k=1;k<L.length;k++)if(L[k].y-L[k-1].y<13)L[k].y=L[k-1].y+13;
      const maxY=h-M.b-5;
      for(let k=L.length-1;k>=0;k--){if(L[k].y>maxY)L[k].y=maxY;if(k<L.length-1&&L[k+1].y-L[k].y<13)L[k].y=L[k+1].y-13;}
      L.forEach(l=>{
        c.strokeStyle=l.s.color;c.lineWidth=2;c.setLineDash(l.s.dash||[]);c.beginPath();c.moveTo(l.x+7,l.y);c.lineTo(l.x+17,l.y);c.stroke();c.setLineDash([]);
        text(c,l.s.short||l.s.name,l.x+21,l.y,{size:10,color:COL.ink,halo:false});
      });
    }
  },o.canvasOpt);
  ch.f=f;
  hoverable(f.cv,(px,py,e)=>{
    const xs=ch.xs;if(!xs||!ch.X)return;
    let best=-1,bd=1e9;
    xs.forEach((x,i)=>{const d=Math.abs(ch.X(x)-px);if(d<bd){bd=d;best=i;}});
    if(px<ch.M.l-12||px>ch.w-ch.M.r+12)best=-1;
    if(best!==ch.hover){ch.hover=best;f.draw();}
    if(best<0){hideTip();return;}
    const S=o.series();
    let html='<div class="tv">'+esc(o.tipTitle?o.tipTitle(best):(o.xFmt?o.xFmt(xs[best]):xs[best]))+'</div>';
    S.forEach(s=>{const v=s.values[best];if(v==null)return;html+='<div class="row"><i style="background:'+s.color+'"></i><span class="tl">'+esc(s.name)+'</span>&nbsp;<b>'+(o.vFmt?o.vFmt(v,s):v.toFixed(2))+'</b></div>';});
    if(o.tipExtra)html+=o.tipExtra(best);
    showTip(html,e.clientX,e.clientY);
  },()=>{ch.hover=-1;f.draw();hideTip();},o.onPick?()=>{if(ch.hover>=0)o.onPick(ch.hover);}:null);
  return ch;
}

/* ---------- table view under a chart (same numbers, readable without the picture) ---------- */
function dataTable(afterId,title){
  const anchor=typeof afterId==='string'?$(afterId):afterId;
  const d=document.createElement('details');d.className='dtab';
  d.innerHTML='<summary>'+esc(title||'Show the numbers')+'</summary><div class="tbl-wrap"><table class="dt"><thead></thead><tbody></tbody></table></div>';
  anchor.insertAdjacentElement('afterend',d);
  const th=d.querySelector('thead'),tb=d.querySelector('tbody');
  return {set(head,rowsList){
    th.innerHTML='<tr>'+head.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr>';
    tb.innerHTML=rowsList.map(r=>'<tr>'+r.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('');
  }};
}
function quantile(arr,q){const a=arr.slice().sort((x,y)=>x-y);const i=(a.length-1)*q,lo=Math.floor(i),hi=Math.ceil(i);return a[lo]+(a[hi]-a[lo])*(i-lo);}
