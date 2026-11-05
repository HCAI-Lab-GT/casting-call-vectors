/* Deck navigation, plus two presentation adjustments for the field guide's figures:
   canvas text scales with the screen (the plates set fonts in px), and figures may grow to 60% of the height. */
(function(){
"use strict";
const fontDesc=Object.getOwnPropertyDescriptor(CanvasRenderingContext2D.prototype,'font');
let FS=1;
function fit(){
  FS=Math.max(1.1,Math.min(1.55,innerWidth/1100));
  window.FIG_MAX_H=Math.max(240,Math.round(innerHeight*.6));
}
fit();addEventListener('resize',fit);
if(fontDesc&&fontDesc.set){
  Object.defineProperty(CanvasRenderingContext2D.prototype,'font',{configurable:true,
    get(){return fontDesc.get.call(this);},
    set(v){fontDesc.set.call(this,String(v).replace(/(\d+(?:\.\d+)?)px/,(m,n)=>(+n*FS).toFixed(1)+'px'));}});
}

const deck=document.getElementById('deck');
const slides=Array.from(deck.children).filter(s=>s.classList.contains('slide'));
const main=slides.filter(s=>!s.classList.contains('appendix'));
const nav=document.createElement('nav');nav.className='navbar';nav.setAttribute('aria-label','Slide navigation');
nav.innerHTML='<span class="where" id="navWhere"></span><button type="button" id="navOver" title="All slides (G)">All slides</button>'+
  '<button type="button" id="navPrev" aria-label="Previous slide">&larr;</button><button type="button" id="navNext" aria-label="Next slide">&rarr;</button>'+
  '<span class="count" id="navCount"></span>';
document.body.appendChild(nav);
const bar=document.createElement('div');bar.className='progress';document.body.appendChild(bar);
const over=document.createElement('div');over.className='overview';over.hidden=true;over.setAttribute('role','dialog');over.setAttribute('aria-label','All slides');
over.innerHTML='<h2>All slides</h2><ol></ol>';document.body.appendChild(over);
const label=(s,i)=>s.classList.contains('appendix')?'A'+(i-main.length+1):String(i+1);
const esc=t=>String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
over.querySelector('ol').innerHTML=slides.map((s,i)=>'<li><button type="button" data-i="'+i+'"><small>'+label(s,i)+' · '+esc(s.dataset.part)+'</small>'+esc(s.dataset.title)+'</button></li>').join('');

let cur=-1;
function show(i,push){
  i=Math.max(0,Math.min(slides.length-1,i));
  if(i===cur)return;
  if(cur>=0){slides[cur].classList.remove('on');if(slides[cur].contains(document.activeElement))document.activeElement.blur();}
  cur=i;const s=slides[i];s.classList.add('on');s.scrollTop=0;
  const ap=s.classList.contains('appendix');
  document.getElementById('navWhere').textContent=(s.dataset.part?s.dataset.part+' · ':'')+(s.dataset.title||'');
  document.getElementById('navCount').textContent=ap?'appendix '+(i-main.length+1)+' / '+(slides.length-main.length):(i+1)+' / '+main.length;
  nav.classList.toggle('dark',s.classList.contains('navy'));
  bar.style.width=(100*Math.min(1,(i+1)/main.length))+'%';
  over.querySelectorAll('button').forEach(b=>b.classList.toggle('cur',+b.dataset.i===i));
  const h='slide-'+(i+1);
  if(push!==false&&location.hash!=='#'+h){try{history.replaceState(null,'','#'+h);}catch(e){location.hash=h;}}
  window.dispatchEvent(new CustomEvent('deckslide',{detail:{index:i,slide:s}}));
}
function fromHash(){const m=/^#slide-(\d+)$/.exec(location.hash);return m?(+m[1]-1):0;}
addEventListener('hashchange',()=>show(fromHash(),false));
document.getElementById('navPrev').addEventListener('click',()=>show(cur-1));
document.getElementById('navNext').addEventListener('click',()=>show(cur+1));
document.getElementById('navOver').addEventListener('click',()=>{over.hidden=false;const b=over.querySelector('button.cur');if(b)b.focus();});
over.addEventListener('click',e=>{const b=e.target.closest('button[data-i]');if(b){over.hidden=true;show(+b.dataset.i);}else if(e.target===over)over.hidden=true;});
addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  const a=document.activeElement,tag=a&&a.tagName;
  const ownsArrows=a&&(tag==='INPUT'||tag==='SELECT'||tag==='TEXTAREA'||(tag==='CANVAS'&&a.classList.contains('drag')));
  const k=e.key;
  if(!over.hidden){if(k==='Escape'||k==='g'||k==='G'){over.hidden=true;e.preventDefault();}return;}
  if(k==='PageDown'||(k==='ArrowRight'&&!ownsArrows)||(k===' '&&tag!=='BUTTON'&&!ownsArrows)||k==='n'){show(cur+1);e.preventDefault();}
  else if(k==='PageUp'||(k==='ArrowLeft'&&!ownsArrows)||k==='p'){show(cur-1);e.preventDefault();}
  else if(k==='Home'&&!ownsArrows){show(0);e.preventDefault();}
  else if(k==='End'&&!ownsArrows){show(main.length-1);e.preventDefault();}
  else if(k==='g'||k==='G'){over.hidden=false;e.preventDefault();}
});
show(fromHash(),false);
})();
