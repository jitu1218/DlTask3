# Alignment Studio
An English-to-Spanish recurrent translation lab. Requires Node 18+ for inference, Python 3 with NumPy for retraining, and Python 3 for packaging.

## Run
From the project root:
```sh
npm run build
npm run dev
```
Visit http://localhost:3000. The deployable output is a self-contained Worker ES module in dist/server/index.js. No external model service is needed.

## Reproduce
```sh
OPENBLAS_NUM_THREADS=1 python3 training/train.py
npm run evaluate
npm run build
```
The seeded generator produces 15,360 distinct English–Spanish sentence pairs. A shuffled split uses 13,500 for training and 1,500 for testing, leaving 360 unused. Test sentences have unseen word combinations drawn from the same templates. The corpus is synthetic, limited to singular subjects, object descriptions and optional time expressions. Test accuracy measures in-domain generalization only.

## Model
A 32-unit tanh recurrent encoder embeds each English word and updates its hidden state. The decoder starts from the final encoder state and recurrently consumes the previous Spanish token. A learned linear query scores each encoder state by a scaled dot product. The context is the attention-weighted sum of those states. Context and decoder state jointly determine next-token logits. Training uses teacher forcing, cross-entropy, backpropagation through time and Adam for nine epochs. Inference uses greedy decoding and stops at the end token or after 24 steps. The matrix omits the final end-token step. This network has one attention layer and one head.

## API
POST /api/translate with JSON such as {"sentence":"The girl sees a red book."}. Returns source and target token arrays, translation, completed (whether the end token was reached), and attention[target_index][source_index]. Unsupported words return HTTP 422. GET /api/model returns vocabulary and measured test results. Training and evaluation run offline.
