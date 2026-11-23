"""Tests for published-algorithm adaptations and external stochastic graph controls."""
import itertools,numpy as np,pytest
from published_baselines import (spectral_synchronization,path_votes,DawidSkeneEM,
  WrenchStyleDawidSkene,seed_parity_truth_tests)
from independent_graphs import one_pair,test_draw as generate_test_draw
from witness_core.robust_study import RELATIONS,check_pair

def test_basic_published_algorithm_fixtures():
    assert seed_parity_truth_tests()

@pytest.mark.parametrize('family',['random_recursive','preferential','balanced_branch','layered_random'])
@pytest.mark.parametrize('change',[False,True])
def test_new_graph_family_degree_switch(family,change):
    p=one_pair(34,401091+7*change+sum(map(ord,family)),family,change)
    check_pair(p,deep=True)
    ys=[y for _,_,y in p.examples()]
    assert ys==([1,0,0,1] if change else [1,0,1,0])

def test_draw_balanced_and_disjoint():
    p=generate_test_draw(20261)
    assert len(p)==48
    assert len({x.seed for x in p})==48
    assert set(x.n for x in p)=={22,34,48}
    assert sum(x.effect for x in p)==24

def test_spectral_identifies_each_perfect_parity_assignment():
    class U:pass
    for values in itertools.product((0,1),repeat=4):
        u=U();u.anchor_label=values[0];u.error_probs=np.array([.07]*6)
        u.observed_bits=np.array([values[a]^values[b] for a,b in RELATIONS],dtype=int)
        assert np.array_equal(spectral_synchronization(u),values)
        assert np.array_equal(spectral_synchronization(u,False),values)
        assert np.array_equal(path_votes(u),np.repeat(np.asarray(values[1:])[:,None],3,axis=1))

def test_wrench_style_em_normalization():
    rng=np.random.default_rng(4)
    V=rng.integers(0,2,size=(500,3))
    model=WrenchStyleDawidSkene(maxiter=300).fit(V)
    probs=model.predict_proba(V)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(1),1)
    assert np.allclose(model.confusion_.sum(2),1)
    assert np.isfinite(model.priors_).all()

def test_ds_standard_em_monotonic():
    V=np.random.default_rng(19).integers(0,2,size=(100,3))
    m=DawidSkeneEM(maxiter=100).fit(V)
    assert min(np.diff(m.history_))>=-1e-6
