import numpy as np, torch
from method import constraint_nll, syndrome, starflip
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from witness_core.robust_study import make_diverse_pair,observe

def test_syndrome_gauge():
    from itertools import product
    for b in product((0,1),repeat=6):
        assert syndrome(b)==syndrome(starflip(b))

def test_gradient_and_anchor():
    u=observe(make_diverse_pair(10,42,'chain_scc',True),33,'moderate')
    z=torch.tensor([.51,.49,.53,.45],requires_grad=True)
    for m in ('independent','correlation_mixture'):
        loss=constraint_nll(z,u,m)
        (g,)=torch.autograd.grad(loss,z,retain_graph=True)
        assert torch.isfinite(g).all()
        assert g.abs().sum()>0

def test_likelihood_permutation_bounds():
    u=observe(make_diverse_pair(12,43,'layered',True),34,'correlated')
    for m in ('independent','correlation_mixture'):
        for p in ([.1,.9,.2,.6],[.5,.5,.5,.5],[.99,.01,.99,.01]):
            l=constraint_nll(torch.tensor(p),u,m)
            assert torch.isfinite(l) and l.item()>0

def test_robust_posterior_and_objective():
    from method import robust_soft_targets
    u=observe(make_diverse_pair(10,44,'layered',False),35,'correlated')
    p=robust_soft_targets(u)
    assert abs(p[0]-u.anchor_label)<1e-12
    assert (p>=0).all() and (p<=1).all()
    # With uniform predictions on the nonanchor queries, the joint objective is finite.
    z=torch.full((4,),.5,requires_grad=True)
    loss=constraint_nll(z,u,'correlation_mixture')
    assert torch.isfinite(loss)
