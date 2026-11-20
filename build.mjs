import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
const read=p=>fs.readFileSync(p,'utf8');
fs.mkdirSync('dist/server',{recursive:true});
execFileSync('python3',['-c',`import glob,zipfile
with zipfile.ZipFile('dist/project.zip','w',zipfile.ZIP_DEFLATED) as z:
 for pat in ['*.mjs','package.json','README.md','training/*.py','model/*.json','web/*']:
  for path in glob.glob(pat): z.write(path)`]);
const assets=Object.fromEntries(['index.html','style.css','app.js'].map(p=>['/'+p,read('web/'+p)]));
const out=read('inference.mjs')+'\nconst model='+read('model/weights.json')+';\nconst report='+read('model/report.json')+';\nconst assets='+JSON.stringify(assets)+';\nconst archive='+JSON.stringify(fs.readFileSync('dist/project.zip').toString('base64'))+';\n'+read('worker.mjs');
fs.writeFileSync('dist/server/index.js',out);
