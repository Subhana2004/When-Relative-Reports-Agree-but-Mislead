"""Regression tests for new reviewer-response experiments; synthetic-only fixture tests."""
from __future__ import annotations
import json, dataclasses
import numpy as np, networkx as nx
from nonmirrored_graphs import draw, make_pair
from witness_core.robust_study import observe, posterior, RELATIONS
from WITNESS_TMLR_Anonymized_Supplement.experiments.method import syndrome,starflip
from run_factorial_targeted import SIZES,POOLS,REPORT_SEEDS,MODEL_SEEDS
from run_size_sweep import nested_training_sets,build_units

def test_three_zone_nonmirrored_degree_truth():
    for seed in (17,19,23,29):
        for n in (22,34,48):
            for effect in (False,True):
                p=make_pair(n,seed+n*917+int(effect)*11,effect)
                assert p.g0.number_of_nodes()==2*n
                assert p.g0.number_of_edges()==p.g1.number_of_edges()
                assert all(p.g0.in_degree(v)==p.g1.in_degree(v) for v in p.g0)
                assert all(p.g0.out_degree(v)==p.g1.out_degree(v) for v in p.g0)
                assert [int(nx.has_path(g,*q)) for g,q,_ in p.examples()]==[y for _,_,y in p.examples()]

def test_randomized_draws_balanced_and_new():
    for seed in (6101,6103):
        p=draw(seed)
        assert len(p)==24 and sum(x.effect for x in p)==12
        assert len(set(x.seed for x in p))==24
        assert all(x.family=='nonmirrored_three_zone' for x in p)

def test_gauge_invariance_and_anchor_observation():
    unit=observe(make_pair(22,7131,True),800017,'moderate')
    bits=tuple(map(int,unit.observed_bits))
    assert syndrome(bits)==syndrome(starflip(bits))
    fake=dataclasses.replace(unit,observed_bits=np.asarray(starflip(bits),dtype=np.int8))
    p,_=posterior(fake);assert np.isfinite(p).all() and len(p)==4
    assert unit.anchor_label==fake.anchor_label

def test_strictly_independent_factors():
    pool=nested_training_sets(314159)
    u1=build_units(pool[24],9001,'heavy')
    u2=build_units(pool[24],9007,'heavy')
    # Identical graph seeds and truth, different reported bits.
    assert [x.pair.seed for x in u1]==[x.pair.seed for x in u2]
    assert any(not np.array_equal(x.observed_bits,y.observed_bits) for x,y in zip(u1,u2))
    assert len(POOLS)==4 and len(REPORT_SEEDS)==len(MODEL_SEEDS)==2
    assert SIZES==(24,48,96,192)

def test_all_possible_parity_gauges():
    import itertools
    for truth in itertools.product([0,1],repeat=4):
        base=[truth[i]^truth[j] for i,j in RELATIONS]
        for s in itertools.product([0,1],repeat=4):
            changed=[base[k]^s[i]^s[j] for k,(i,j) in enumerate(RELATIONS)]
            assert syndrome(tuple(base))==syndrome(tuple(changed))

if __name__=='__main__':
    for name,fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):fn();print('PASS',name)
