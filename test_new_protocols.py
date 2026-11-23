"""Deterministic protocol tests, not claims about an external natural dataset."""
import random
import networkx as nx
from run_size_sweep import nested_training_sets,build_units,train_fixed_budget
from run_external_snap_protocol import directed_swap,draw_pair,generate

def test_nested_size_sets_and_strata():
    groups=nested_training_sets(314159)
    assert {k:len(v) for k,v in groups.items()}=={24:24,48:48,96:96,192:192}
    assert all(set(p.seed for p in groups[a]).issubset(set(p.seed for p in groups[b])) for a,b in zip([24,48,96],[48,96,192]))

def test_repeated_noise_consistent_across_nested_training_sizes():
    g=nested_training_sets(314159)
    unit_24=build_units(g[24],71,'heavy')
    unit_192=build_units(g[192],71,'heavy')
    mapping={u.pair.seed:u for u in unit_192}
    for u in unit_24:
        t=mapping[u.pair.seed]
        assert tuple(u.observed_bits)==tuple(t.observed_bits)
        assert tuple(u.error_probs)==tuple(t.error_probs)

def test_training_budget_metadata():
    p=nested_training_sets(314159)[24]
    u=build_units(p,71,'heavy')
    for arm in ['weighted_soft','joint_independent']:
        model,meta=train_fixed_budget(u,71,arm,3)
        assert meta['updates']==3 and meta['graph_forwards']==12

def test_directed_swap_retains_named_degrees():
    g=nx.DiGraph();g.add_nodes_from(range(20))
    g.add_edges_from((i,i+1) for i in range(19))
    g.add_edges_from((i,i-2) for i in range(2,20))
    swap,t=directed_swap(g,random.Random(91))
    assert swap is not None and t>0
    h=swap['graph']
    assert all(g.in_degree(v)==h.in_degree(v) and g.out_degree(v)==h.out_degree(v) for v in g)
    assert g.number_of_edges()==h.number_of_edges()

def test_offline_synthetic_fixture_is_not_claimed_as_natural():
    g=nx.DiGraph();g.add_edges_from((i,i+1) for i in range(99))
    g.add_edges_from((i,i-3) for i in range(3,100))
    out=generate(g,'fixture-only',17,40,5)
    assert out['requested_draws']==5
    assert out['original_graph_nodes']==100
    assert out['valid_draws']<=5
    for row in out['rows']:
        if row['status']=='valid':assert len(row['exact_oracle_labels'])==4

def test_non_mirror_intervention_can_flip_reachability():
    g=nx.DiGraph([(0,1),(1,2),(3,4),(4,5)])
    h=g.copy();h.remove_edges_from([(1,2),(3,4)]);h.add_edges_from([(1,4),(3,2)])
    assert [g.in_degree(v) for v in range(6)]==[h.in_degree(v) for v in range(6)]
    assert [g.out_degree(v) for v in range(6)]==[h.out_degree(v) for v in range(6)]
    assert nx.has_path(g,0,2) and not nx.has_path(h,0,2)
    assert not nx.has_path(g,3,2) and nx.has_path(h,3,2)
