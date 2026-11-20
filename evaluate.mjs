import fs from 'node:fs';
import assert from 'node:assert/strict';
import {translate} from './inference.mjs';
const read = path => JSON.parse(fs.readFileSync(path,'utf8'));
const model = read('model/weights.json'), test = read('model/test.json');
let exact = 0;
for (const [source, reference] of test) {
  const result = translate(model,source);
  assert.equal(result.attention.length,result.target.length);
  for (const row of result.attention) {
    assert.equal(row.length,result.source.length);
    assert.ok(row.every(x => Number.isFinite(x) && x>=0));
    assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-9);
  }
  exact += Number(result.completed && result.translation===reference);
}
const report={training:13500,heldOut:test.length,exact};
fs.writeFileSync('model/report.json',JSON.stringify(report));
console.log(report);
