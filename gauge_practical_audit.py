"""Paired anchor-fan corruption stress test on two genuinely distinct graph families.

The parity proposition is not a new synchronization theorem. This experiment
quantifies which downstream decision diagnostics fail, using randomized reported
errors and a matched no-fan control. Additional verified anchor is MORE
information and cannot be compared as if it had identical supervision cost.
"""
from __future__ import annotations
import json, dataclasses
from pathlib import Path
import numpy as np
from witness_core.robust_study import split,observe,ASSIGNMENTS,posterior
from WITNESS_TMLR_Anonymized_Supplement.experiments.method import syndrome,starflip
from nonmirrored_graphs import draw

def prediction(unit, extra_anchor=None):
    marginals,pr=posterior(unit,weighted=True,soft=True)
    if extra_anchor is not None:
        labels=ASSIGNMENTS^unit.anchor_label
        retained=(labels[:,1]==int(extra_anchor))
        weighted=pr*retained
        assert weighted.sum()>0
        post=weighted/weighted.sum()
        marginals=np.matmul(post,labels.astype(float))
    return (marginals>=.5).astype(int),np.asarray(marginals)

def main():
    families={
      'mirrored_original':split(1618033,[22,34,48],['branch','dense_scc'],2),
      'nonmirrored_asymmetric':draw(6101)}
    out={'protocol':{'per_quadruple_report_draws':16,
      'original_graphs':len(families['mirrored_original']),
      'nonmirrored_graphs':len(families['nonmirrored_asymmetric']),
      'forced_anchor_fan':'all three anchor-incident relation bits flipped as a paired perturbation',
      'cycle_syndrome':'triangle XOR values before and after fan; no model training',
      'extra_anchor':'independent exact label for observed position 1; additional-information control',
      'meaning':'diagnostic consequence of known gauge invariance; not a novel theorem'},
      'results':{}}
    for family,pairs in families.items():
        stats={k:0 for k in ('cases','same_syndrome','base_correct','fan_correct','fan_plus_extra_anchor_correct',
                            'wrong_fan_with_low_syndrome','fan_total_brier','base_total_brier','extra_total_brier')}
        for i,p in enumerate(pairs):
            for j in range(16):
                u=observe(p,800071+i*104729+j*1009,'moderate')
                labels=np.array([y for _,_,y in u.examples()],dtype=int)
                f=dataclasses.replace(u,observed_bits=np.asarray(starflip(tuple(map(int,u.observed_bits))),dtype=np.int8))
                assert f.anchor_label==u.anchor_label
                same=syndrome(tuple(map(int,u.observed_bits)))==syndrome(tuple(map(int,f.observed_bits)))
                p0,m0=prediction(u);p1,m1=prediction(f);p2,m2=prediction(f,int(labels[1]))
                stats['cases']+=1;stats['same_syndrome']+=int(same)
                stats['base_correct']+=int(np.array_equal(p0,labels))
                stats['fan_correct']+=int(np.array_equal(p1,labels))
                stats['fan_plus_extra_anchor_correct']+=int(np.array_equal(p2,labels))
                stats['wrong_fan_with_low_syndrome']+=int(not np.array_equal(p1,labels) and sum(syndrome(tuple(map(int,f.observed_bits))))==0)
                stats['fan_total_brier']+=float(np.mean((m1-labels)**2))
                stats['base_total_brier']+=float(np.mean((m0-labels)**2))
                stats['extra_total_brier']+=float(np.mean((m2-labels)**2))
        n=stats['cases']
        out['results'][family]={**stats,'base_cqa':stats['base_correct']/n,'fan_cqa':stats['fan_correct']/n,
                                'fan_extra_anchor_cqa':stats['fan_plus_extra_anchor_correct']/n,
                                'base_brier':stats['base_total_brier']/n,
                                'fan_brier':stats['fan_total_brier']/n,
                                'extra_brier':stats['extra_total_brier']/n}
    dest=Path('results/gauge_practical_audit.json');dest.write_text(json.dumps(out,indent=2))
    for fam,stats in out['results'].items():
        print(fam,'paired',stats['cases'],'same syndromes',stats['same_syndrome'],
              'base CQA',round(stats['base_cqa'],4),'corrupt CQA',round(stats['fan_cqa'],4),
              'corrupt with 2 anchors CQA',round(stats['fan_extra_anchor_cqa'],4),
              'base/fan Brier',round(stats['base_brier'],4),round(stats['fan_brier'],4))
if __name__=='__main__':main()
