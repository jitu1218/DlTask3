// Vectors are row vectors throughout, matching the NumPy training implementation.
function project(vector, matrix) {
  const output = new Float64Array(matrix[0].length);
  for (let i = 0; i < vector.length; i++) {
    for (let j = 0; j < output.length; j++) output[j] += vector[i] * matrix[i][j];
  }
  return output;
}
function recurrent(previous, matrix, embedding) {
  return project(previous, matrix).map((x, i) => Math.tanh(x + embedding[i]));
}
export function translate(model, sentence) {
  if (typeof sentence !== 'string' || sentence.length > 240) throw Error('Enter a sentence of up to 240 characters.');
  const tokens = sentence.trim().toLowerCase().replace(/[.!?]+$/, '').trim().split(/\s+/).filter(Boolean);
  if (!tokens.length || tokens.length > 24) throw Error('Enter between 1 and 24 English words.');
  const unknown = [...new Set(tokens.filter(t => !model.source.includes(t)))];
  if (unknown.length) throw Error('Outside the training vocabulary: ' + unknown.join(', ') + '. Try an example or check the model guide.');
  const w = model.weights;
  let state = new Float64Array(w.R.length);
  const states = tokens.map(t => state = recurrent(state, w.R, w.E[model.source.indexOf(t)]));
  const target = [], attention = [];
  let previous = 0, completed = false;
  for (let step = 0; step < 24; step++) {
    state = recurrent(state, w.S, w.D[previous]);
    const query = project(state, w.Q);
    const scores = states.map(s => s.reduce((a,b,i) => a + b*query[i], 0)/Math.sqrt(state.length));
    const max = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s-max));
    const sum = exps.reduce((a,b) => a+b);
    const row = exps.map(e => e/sum);
    const context = states[0].map((_,i) => row.reduce((a,b,j) => a+b*states[j][i],0));
    const direct = project(state,w.O);
    const logits = project(context,w.C).map((v,i) => v+direct[i]);
    logits[0] = -Infinity;
    previous = logits.indexOf(Math.max(...logits));
    if (previous === 1) {completed = true; break;}
    target.push(model.target[previous]);
    attention.push(row);
  }
  return {source:tokens,target,attention,translation:target.join(' '),completed};
}
