"""Intermediate-state neural-execution upper control, inspired by NAR literature.
Not a reproduction of any published implementation and not information-matched.
Full per-node reachability traces are computed by an exact oracle for training.
"""
from __future__ import annotations
import sys,random,json,time
from pathlib import Path
import torch,numpy as np
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from witness_core.gated_baseline import GatedNet
from witness_core.robust_study import split
from witness_core.legacy_directed import pack
from extended_study import make_test_draws,score_prepacked


def learn_trace(net,ex):
    adj,source,target,steps=ex
    visited=source.bool()
    h=torch.tanh(net.initial(source[:,None]))
    losses=[]
    for t in range(steps):
        nbr=torch.where(adj[:,:,None],h[None,:,:],torch.full_like(h[None,:,:],-1e5)).amax(1)
        nbr=torch.where(adj.any(1,keepdim=True),nbr,0.)
        h=net.gate(torch.cat((nbr,source[:,None]),-1),h)
        visited=visited | ((adj.float() @ visited.float())>0)
        p=torch.sigmoid(net.readout(h).squeeze(-1)).clamp(1e-5,1-1e-5)
        losses.append(F.binary_cross_entropy(p,visited.float()))
    return p[target],torch.stack(losses).mean()


def run():
    torch.set_num_threads(1)
    tr=split(314159,[10,12],['chain_scc','layered'],3)
    draws=make_test_draws();rows=[]
    for seed in [71,79,89]:
        torch.manual_seed(seed);net=GatedNet();opt=torch.optim.AdamW(net.parameters(),lr=.003)
        prepared=[[pack(g,q,13) for g,q,y in p.examples()] for p in tr]
        labels=[[y for g,q,y in p.examples()] for p in tr]
        start=time.perf_counter()
        for ep in range(6):
            order=list(range(len(tr)));random.Random(seed*61+ep).shuffle(order)
            for j in order:
                preds=[];trace=[]
                for ex in prepared[j]:
                    p,l=learn_trace(net,ex);preds.append(p);trace.append(l)
                target=torch.tensor(labels[j],dtype=torch.float32)
                loss=.7*F.binary_cross_entropy(torch.stack(preds).clamp(1e-5,1-1e-5),target)+.3*torch.stack(trace).mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        for i,draw in enumerate(draws):
            cqa,_=score_prepacked(net,draw)
            rows.append({'seed':seed,'draw_index':i,'cqa':cqa,'training_seconds':time.perf_counter()-start})
        print('TRACE',seed,'mean cqa',np.mean([x['cqa'] for x in rows if x['seed']==seed]),flush=True)
    out=Path(__file__).resolve().parent/'results'/'trace_control.json';out.write_text(json.dumps({'note':'Veličković-style intermediate-state supervision adaptation; additional oracle information, not an information-matched baseline','rows':rows},indent=2))
if __name__=='__main__':run()
