"""Alternative gated, 16-dimensional recurrent message-passing architecture.
This is an in-study architecture diagnostic, not a published baseline reproduction.
"""
import random,time,json,pathlib
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from .robust_study import split,observe,infer
from .legacy_directed import pack,evaluate

class GatedNet(nn.Module):
    def __init__(self,dim=16):
        super().__init__();self.initial=nn.Linear(1,dim);self.gate=nn.GRUCell(dim+1,dim);self.readout=nn.Linear(dim,1)
    def forward(self,ex):
        adj,source,target,steps=ex
        h=torch.tanh(self.initial(source[:,None]))
        for _ in range(steps):
            nbr=torch.where(adj[:,:,None],h[None,:,:],torch.full_like(h[None,:,:],-1e5)).amax(1)
            nbr=torch.where(adj.any(1,keepdim=True),nbr,0.)
            h=self.gate(torch.cat((nbr,source[:,None]),-1),h)
        return torch.sigmoid(self.readout(h[target]).squeeze())

def run():
    torch.set_num_threads(1)
    tr=split(314159,[10,12],['chain_scc','branch','layered'],3)
    te=split(271828,[18,28,40],['dense_scc','layered'],4)
    rows=[]
    for seed in (13,23,37):
        units=[observe(p,seed*100003+i*31,'moderate') for i,p in enumerate(tr)]
        for arm in ('weighted_soft','oracle'):
            torch.manual_seed(seed);net=GatedNet();opt=torch.optim.AdamW(net.parameters(),lr=.003)
            prep=[[pack(g,q,13) for g,q,_ in u.examples()] for u in units]
            targets=[]
            for u in units:targets.append(torch.tensor(infer(u,arm)[0],dtype=torch.float32))
            start=time.perf_counter()
            for ep in range(6):
                order=list(range(len(units)));random.Random(seed*61+ep).shuffle(order)
                for i in order:
                    p=torch.stack([net(x) for x in prep[i]]).clamp(1e-6,1-1e-6)
                    loss=F.binary_cross_entropy(p,targets[i]);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            s=evaluate(net,te)
            row={'seed':seed,'arm':arm,'updates':6*len(tr),'seconds':round(time.perf_counter()-start,2),'query_accuracy':s['accuracy'],'cqa':s['cqa'],'positive_recall':s['positive_recall'],'negative_recall':s['negative_recall']}
            rows.append(row);print('GATED',row,flush=True)
            pathlib.Path('results/gated_baseline.json').write_text(json.dumps({'note':'architecture ablation, not a published baseline reproduction','rows':rows},indent=2))
if __name__=='__main__':run()
