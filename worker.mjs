// build.mjs embeds model, inference code, assets and assignment archive.
export default {async fetch(req){
  const path=new URL(req.url).pathname;
  if(path==='/api/translate'){
    if(req.method!=='POST')return Response.json({error:'Use POST.'},{status:405,headers:{Allow:'POST'}});
    let body;
    try{body=await req.json();}catch{return Response.json({error:'Expected a JSON object.'},{status:400});}
    try{return Response.json(translate(model,body?.sentence));}
    catch(error){return Response.json({error:error.message},{status:422});}
  }
  if(req.method!=='GET')return new Response('Use GET',{status:405});
  if(path==='/api/model')return Response.json({...report,vocabulary:model.source,heads:1});
  if(path==='/project.zip')return new Response(Uint8Array.from(atob(archive),c=>c.charCodeAt(0)),{headers:{'Content-Type':'application/zip','Content-Disposition':'attachment; filename="alignment-studio.zip"'}});
  const key=path==='/'?'/index.html':path;
  if(!Object.hasOwn(assets,key))return new Response('Not found',{status:404});
  const type=key.endsWith('html')?'text/html':key.endsWith('css')?'text/css':'text/javascript';
  return new Response(assets[key],{headers:{'Content-Type':type+'; charset=utf-8'}});
}};
