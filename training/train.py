"""Train a word-level recurrent encoder-decoder with scaled dot-product attention."""
import json
import itertools
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(17)
SUBJECTS = [('the girl','la niña'),('the boy','el niño'),('my sister','mi hermana'),
            ('my brother','mi hermano'),('the teacher','el profesor'),('my friend','mi amigo')]
VERBS = [('sees','ve'),('finds','encuentra'),('buys','compra'),('wants','quiere'),('has','tiene')]
NOUNS = [('book','libro'),('car','coche'),('notebook','cuaderno'),('bag','bolso'),
         ('hat','sombrero'),('coat','abrigo'),('pencil','lápiz'),('watch','reloj')]
ADJECTIVES = [('', ''),('red','rojo'),('blue','azul'),('green','verde'),('small','pequeño'),
              ('new','nuevo'),('black','negro'),('white','blanco')]
TIMES = [('', ''),('today','hoy'),('now','ahora'),('again','otra vez')]

def corpus():
    pairs = []
    for sub, verb, det, adj, noun, time in itertools.product(
            SUBJECTS, VERBS, [('a','un'),('the','el')], ADJECTIVES, NOUNS, TIMES):
        en = [sub[0],verb[0],det[0],adj[0],noun[0],time[0]]
        es = [sub[1],verb[1],det[1],noun[1],adj[1],time[1]]
        pairs.append((' '.join(filter(None,en)), ' '.join(filter(None,es))))
    rng.shuffle(pairs)
    return pairs

def softmax(x):
    e = np.exp(x-x.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)

def fit():
    pairs = corpus()
    train, test = pairs[:13500], pairs[13500:15000]
    source, target = [sorted(set(' '.join(x[k] for x in train).split())) for k in (0,1)]
    target = ['<start>','<end>']+target
    h = 32
    shapes = dict(E=(len(source),h), R=(h,h), D=(len(target),h), S=(h,h),
                  Q=(h,h), C=(h,len(target)), O=(h,len(target)))
    p = {k:rng.normal(0,0.19,shape).astype('f') for k,shape in shapes.items()}
    first = {k:np.zeros_like(w) for k,w in p.items()}
    second = {k:np.zeros_like(w) for k,w in p.items()}
    groups = {}
    for a,b in train:
        x=[source.index(w) for w in a.split()]
        y=[target.index(w) for w in b.split()]+[1]
        groups.setdefault((len(x),len(y)),[]).append((x,y))
    step=0
    for epoch in range(9):
        batches=[]
        for group in groups.values():
            rng.shuffle(group)
            batches.extend(group[i:i+96] for i in range(0,len(group),96))
        rng.shuffle(batches)
        for batch in batches:
            x,y=(np.array(t) for t in zip(*batch))
            n,m=x.shape
            zeros=np.zeros((n,h),'f')
            enc=[zeros]
            for ids in x.T:
                enc.append(np.tanh(p['E'][ids]+enc[-1]@p['R']))
            states=np.stack(enc[1:],axis=1)
            dec=[enc[-1]]
            cache=[]
            grad={k:np.zeros_like(w) for k,w in p.items()}
            denc=np.zeros_like(states)
            prev=np.zeros(n,int)
            for ids in y.T:
                z=np.tanh(p['D'][prev]+dec[-1]@p['S'])
                q=z@p['Q']
                a=softmax((states*q[:,None]).sum(2)/np.sqrt(h)).astype('f')
                c=(a[:,:,None]*states).sum(1)
                dl=softmax(c@p['C']+z@p['O'])
                dl[np.arange(n),ids]-=1
                dl/=y.size
                grad['C']+=c.T@dl
                grad['O']+=z.T@dl
                dc=dl@p['C'].T
                da=(states*dc[:,None]).sum(2)
                ds=(a*(da-(a*da).sum(1,keepdims=True))/np.sqrt(h)).astype('f')
                denc+=a[:,:,None]*dc[:,None]+ds[:,:,None]*q[:,None]
                dq=(states*ds[:,:,None]).sum(1)
                grad['Q']+=z.T@dq
                cache.append((prev,dl@p['O'].T+dq@p['Q'].T))
                dec.append(z)
                prev=ids
            carry=zeros
            for t in reversed(range(len(cache))):
                ids,dz=cache[t]
                delta=(carry+dz)*(1-dec[t+1]**2)
                np.add.at(grad['D'],ids,delta)
                grad['S']+=dec[t].T@delta
                carry=delta@p['S'].T
            for t in reversed(range(m)):
                delta=(carry+denc[:,t])*(1-enc[t+1]**2)
                np.add.at(grad['E'],x[:,t],delta)
                grad['R']+=enc[t].T@delta
                carry=delta@p['R'].T
            step+=1
            for k,g in grad.items():
                first[k]=.9*first[k]+.1*g
                second[k]=.999*second[k]+.001*g*g
                p[k]-=.008*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
        print('Epoch',epoch+1,flush=True)
    model=dict(source=source,target=target,weights={k:w.tolist() for k,w in p.items()})
    (ROOT/'model/weights.json').write_text(json.dumps(model,separators=(',',':')))
    (ROOT/'model/test.json').write_text(json.dumps(test,ensure_ascii=False))

if __name__=='__main__':
    fit()
