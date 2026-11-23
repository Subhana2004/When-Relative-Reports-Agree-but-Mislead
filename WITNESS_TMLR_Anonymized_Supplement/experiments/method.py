"""Constraint-marginalized neural training and gauge-symmetry audits.
No ground-truth labels except one observed anchor are used in the proposed loss.
This is a research candidate, not an established novel algorithm.
"""
from __future__ import annotations
import itertools
import torch
from torch.nn import functional as F
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from witness_core.robust_study import RELATIONS

ASSIGN = torch.tensor([[0,*v] for v in itertools.product([0,1], repeat=3)],dtype=torch.float32)
EDGES = torch.tensor(RELATIONS,dtype=torch.int64)
XORS = torch.remainder(ASSIGN[:,EDGES[:,0]]+ASSIGN[:,EDGES[:,1]],2)

def constraint_nll(pred:torch.Tensor,unit, mode:str='independent'):
    """Marginal observation likelihood over all assignments consistent with one anchor.
    mode=independent uses reported error probabilities, or the conservative mixture
    of independent and star-flipped reporting. Designed for 4 labelled instances.
    """
    assert pred.shape==(4,)
    eps=torch.tensor(unit.error_probs,dtype=pred.dtype,device=pred.device).clamp(.001,.499)
    assignments=ASSIGN.to(pred.device)
    vals=assignments.clone()
    if unit.anchor_label: vals=1-vals
    # Observation order's true anchor is always 0 in the current evaluator.
    assert unit.anchor_index==0
    bits=torch.tensor(unit.observed_bits,dtype=pred.dtype,device=pred.device)
    true_bits=XORS.to(pred.device)
    likelihood=torch.where(true_bits==bits[None,:],1-eps[None,:],eps[None,:])
    if mode=='independent':
        rel_log=torch.log(likelihood).sum(1)
    elif mode=='correlation_mixture':
        # 20% mass on an unobserved common-mode fan flip. This is a fixed,
        # predeclared sensitivity model; no access to the synthetic hidden flag.
        fan=torch.tensor([int(a==0 or b==0) for a,b in RELATIONS], device=pred.device,dtype=pred.dtype)
        alt_bits=bits[None,:]*(1-fan[None,:])+(1-bits[None,:])*fan[None,:]
        alt_like=torch.where(true_bits==alt_bits,1-eps[None,:],eps[None,:])
        rel_log=torch.logaddexp(torch.log(torch.tensor(.8,device=pred.device))+torch.log(likelihood).sum(1),torch.log(torch.tensor(.2,device=pred.device))+torch.log(alt_like).sum(1))
    else:raise ValueError(mode)
    # Condition on anchor exactly. Only non-anchor model probabilities are needed
    # in the marginal sum; independently anchor supervised with BCE.
    p=pred.clamp(1e-5,1-1e-5)
    y=vals[:,1:]
    log_model=(y*torch.log(p[None,1:])+(1-y)*torch.log1p(-p[None,1:])).sum(1)
    # Normalize by observation evidence to avoid penalizing irreducible bad reports.
    out=-(torch.logsumexp(log_model+rel_log,dim=0)-torch.logsumexp(rel_log,dim=0))
    anchorloss=F.binary_cross_entropy(p[0],p.new_tensor(float(unit.anchor_label)))
    return out+anchorloss

def syndrome(bits):
    d={e:int(bits[i]) for i,e in enumerate(RELATIONS)}
    return tuple(d[(a,b)]^d[(a,c)]^d[(b,c)] for a,b,c in itertools.combinations(range(4),3))

def starflip(bits):
    return tuple(int(bit)^(int(a==0 or b==0)) for bit,(a,b) in zip(bits,RELATIONS))

def robust_soft_targets(unit,mixture=.2):
    """Exact posterior targets under *same* fixed latent-star mixture as joint_robust.
    Strong, information- and model-matched soft-label baseline; no graph predictions.
    """
    import numpy as np
    from witness_core.robust_study import ASSIGNMENTS, XOR_TABLE
    p=np.clip(np.asarray(unit.error_probs),.001,.499)
    bits=np.asarray(unit.observed_bits)
    star=np.asarray([int(a==0 or b==0) for a,b in RELATIONS])
    l0=np.where(XOR_TABLE==bits[None,:],1-p[None,:],p[None,:]).prod(1)
    alt=bits^star
    l1=np.where(XOR_TABLE==alt[None,:],1-p[None,:],p[None,:]).prod(1)
    post=(1-mixture)*l0+mixture*l1
    post=post/post.sum()
    vals=(ASSIGNMENTS^unit.anchor_label).astype(float)
    return post @ vals
