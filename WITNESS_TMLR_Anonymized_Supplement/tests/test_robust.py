import numpy as np,pytest,networkx as nx
from witness_core.robust_study import *

@pytest.mark.parametrize('family',['chain_scc','branch','dense_scc','layered'])
@pytest.mark.parametrize('change',[False,True])
@pytest.mark.parametrize('seed',[2,18,514])
def test_graph_pair(family,change,seed):
    p=make_diverse_pair(18,seed,family,change)
    check_pair(p,deep=True)
    assert sum(1 for a,b,_ in p.examples())==4

@pytest.mark.parametrize('effect',[0,1])
def test_clean_relations_exact(effect):
    p=make_diverse_pair(18,25,'branch',bool(effect))
    u=observe(p,13,'clean',0)
    for arm in ['tree','unweighted_map','weighted_map','weighted_soft']:
        pred=infer(u,arm)[0]
        assert np.allclose(pred,np.array([y for _,_,y in u.examples()]),atol=1e-7)

@pytest.mark.parametrize('regime',['moderate','heavy','correlated'])
def test_posterior_respects_anchor_and_normalizes(regime):
    p=make_diverse_pair(18,3,'layered',True);u=observe(p,81,regime,0)
    pred,post=posterior(u,True,True)
    assert post.sum()==pytest.approx(1.,abs=1e-9)
    assert pred[0]==pytest.approx(u.anchor_label)
    assert np.all((pred>=0)&(pred<=1))


def test_known_inconsistent_reports():
    p=make_diverse_pair(18,3,'branch',True);u=observe(p,4,'clean')
    u.observed_bits[0]^=1
    assert relation_violation_rate(u)>0


def test_posterior_no_hidden_labels():
    p=make_diverse_pair(18,3,'branch',True);u=observe(p,4,'moderate')
    a,_=posterior(u,True,True)
    u.pair=None
    b,_=posterior(u,True,True)
    assert np.array_equal(a,b)


def test_query_randomization_breaks_first_positive_shortcut():
    p=make_diverse_pair(18,299,'branch',True)
    anchors=[observe(p,s,'moderate').anchor_label for s in range(40)]
    assert 0 in anchors and 1 in anchors


def test_noisy_constraints_can_conflict():
    p=make_diverse_pair(18,288,'branch',True)
    vals=[relation_violation_rate(observe(p,i,'heavy')) for i in range(60)]
    assert any(v>0 for v in vals)
    assert all(0<=v<=1 for v in vals)


def test_loss_gradient_for_gated_alternative():
    import torch
    from witness_core.gated_baseline import GatedNet
    from witness_core.legacy_directed import pack
    p=make_diverse_pair(12,711,'chain_scc',True)
    g,q,y=p.examples()[0]
    net=GatedNet()
    z=net(pack(g,q,13))
    z.backward()
    assert all(t.grad is not None for t in net.parameters())
