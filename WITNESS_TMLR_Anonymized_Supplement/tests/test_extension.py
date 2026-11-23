import sys,unittest,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from experiments.extended_study import em_parity,SEEDS,GRAPH_SEEDS,run_audit
from witness_core.robust_study import split,observe
from experiments.method import syndrome,starflip

class ExtensionTests(unittest.TestCase):
    def test_new_seeds(self):
        self.assertGreater(len(SEEDS),10);self.assertEqual(len(set(SEEDS)),len(SEEDS));self.assertEqual(len(set(GRAPH_SEEDS)),len(GRAPH_SEEDS))
    def test_star_invariance_all_observations(self):
        for n in range(4):
            for a in range(64):
                bits=tuple((a>>i)&1 for i in range(6))
                self.assertEqual(syndrome(bits),syndrome(starflip(bits)))
    def test_em_bounded_and_no_truth_access(self):
        pairs=split(314159,[10,12],['chain_scc','layered'],3)
        units=[observe(p,seed*31+1,'heavy') for seed,p in enumerate(pairs)]
        soft,eps=em_parity(units)
        self.assertEqual(soft.shape,(24,4));self.assertTrue(np.all((eps>=.015)&(eps<=.48)))
        self.assertTrue(np.allclose(soft[:,0],[u.anchor_label for u in units]))
        self.assertTrue(np.all((soft>=0)&(soft<=1)))
    def test_no_ghost_graph_split(self):
        tr=split(314159,[10,12],['chain_scc','layered'],3)
        te=split(1618033,[22,34,48],['branch','dense_scc'],2)
        self.assertTrue(set(x.n for x in tr).isdisjoint(set(x.n for x in te)))
        self.assertTrue(set(x.family for x in tr).isdisjoint(set(x.family for x in te)))
if __name__=='__main__':unittest.main()
