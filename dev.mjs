import http from 'node:http';
import app from './dist/server/index.js';
const port=Number(process.env.PORT || process.argv[process.argv.indexOf('--port')+1]) || 3000;
http.createServer(async(req,res)=>{
  try {
    const chunks=[];
    for await(const part of req) chunks.push(part);
    const init={method:req.method,headers:req.headers};
    if(chunks.length) init.body=Buffer.concat(chunks);
    const response=await app.fetch(new Request('http://localhost'+req.url,init));
    res.writeHead(response.status,Object.fromEntries(response.headers));
    res.end(Buffer.from(await response.arrayBuffer()));
  } catch { res.writeHead(500);res.end(); }
}).listen(port,'0.0.0.0');
