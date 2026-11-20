const $=id=>document.getElementById(id);
let result,step=0,requestId=0;
const element=(tag,value)=>{const node=document.createElement(tag);node.textContent=value;return node;};
function inspect(index){
  step=index;
  $('token').value=String(index);
  [...$('matrix').querySelectorAll('tbody tr')].forEach((row,i)=>row.classList.toggle('sel',i===index));
  $('bars').replaceChildren(...result.source.map((word,i)=>{
    const row=element('div',''),track=element('div',''),fill=element('div','');
    row.className='barrow'; track.className='track';fill.className='fill';
    const percentage=100*result.attention[index][i];
    fill.style.width=percentage+'%';track.append(fill);
    row.append(element('span',word),track,element('span',percentage.toFixed(1)+'%'));
    return row;
  }));
}
function render(){
  $('output').textContent=result.translation ? result.translation[0].toUpperCase()+result.translation.slice(1)+(result.completed?'.':'…') : 'No target tokens generated.';
  $('count').textContent=result.source.length+' source tokens  /  '+result.target.length+' target tokens'+(result.completed?'':'  /  decoding limit reached');
  const table=document.createElement('table'),head=table.createTHead().insertRow();
  table.setAttribute('aria-label','Spanish target rows by English source columns, attention weights');
  table.style.minWidth=(75+result.source.length*48)+'px';
  for(const word of ['',...result.source])head.append(element('th',word));
  const body=table.createTBody();
  $('token').replaceChildren();
  result.target.forEach((word,i)=>{
    $('token').add(new Option((i+1)+' · '+word,i));
    const row=body.insertRow(),label=element('th',word);
    label.scope='row';row.append(label);
    row.onclick=()=>inspect(i);
    result.attention[i].forEach((weight,j)=>{
      const cell=row.insertCell();const scale=Math.sqrt(weight);
      const color=[238,241,230].map((v,k)=>Math.round(v+([52,78,30][k]-v)*scale));
      cell.style.backgroundColor='rgb('+color.join(',')+')';
      cell.style.color=scale>.66?'#fff':'#45503b';
      cell.textContent=weight.toFixed(2);
      cell.title=word+' / '+result.source[j]+': '+weight.toFixed(6);
    });
  });
  $('matrix').replaceChildren(table);
  $('bars').replaceChildren();
  if(result.target.length)inspect(0);
}
async function run(){
  const id=++requestId;
  $('error').textContent='';$('run').disabled=true;$('run').textContent='Translating…';
  try{
    const response=await fetch('/api/translate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sentence:$('source').value})});
    const data=await response.json();if(id!==requestId)return;
    if(!response.ok)throw Error(data.error||'Translation failed.');
    result=data;render();
  }catch(error){
    if(id!==requestId)return;
    $('error').textContent=error.message;
    $('output').textContent='Translation unavailable';$('count').textContent='';
    for(const name of ['matrix','token','bars'])$(name).replaceChildren();
  }finally{if(id===requestId){$('run').disabled=false;$('run').textContent='Translate';}}
}
$('form').onsubmit=e=>{e.preventDefault();run();};
$('examples').onchange=e=>{if(e.target.value){$('source').value=e.target.value;e.target.selectedIndex=0;run();}};
$('token').onchange=e=>inspect(Number(e.target.value));
$('guide').onclick=()=>$('modal').showModal();
$('explore').onclick=()=>$('source').focus();
$('close').onclick=()=>$('modal').close();
fetch('/api/model').then(r=>r.json()).then(m=>$('vocab').textContent=m.vocabulary.join(', ')).catch(()=>$('vocab').textContent='Vocabulary unavailable.');
run();
