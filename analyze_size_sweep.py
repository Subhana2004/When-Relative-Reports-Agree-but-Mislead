"""Independent deterministic regeneration of exploratory WITNESS scaling summary."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np

SIZES=(24,48,96,192)
ARMS=('weighted_soft','joint_independent')
KINDS=('original','arborescence')

def summarize(path:Path):
    data=path.read_bytes(); raw=json.loads(data); rows=raw['rows']
    assert len(rows)==384, f'incomplete expected fixed pilot: {len(rows)} draw rows'
    assert len({(r['data_seed'],r['size'],r['model_report_seed'],r['arm']) for r in rows})==48
    result=[]
    for kind in KINDS:
        for size in SIZES:
            by_arm={}
            for arm in ARMS:
                clusters=defaultdict(list)
                for row in rows:
                    if row['draw_family']==kind and row['size']==size and row['arm']==arm:
                        key=(row['data_seed'],row['model_report_seed'])
                        clusters[key].append(float(row['cqa']))
                        assert len(row['correct'])==row['n_quadruples']
                        assert abs(sum(row['correct'])/len(row['correct'])-row['cqa'])<1e-10
                        assert row['training']['updates']==192
                        assert row['training']['graph_forwards']==768
                assert len(clusters)==6
                assert all(len(v)==(5 if kind=='original' else 3) for v in clusters.values())
                by_arm[arm]=np.asarray([np.mean(v) for _,v in sorted(clusters.items())])
            a=by_arm['weighted_soft'];b=by_arm['joint_independent'];d=b-a
            result.append({'family':kind,'size':size,'soft_mean':round(float(a.mean()),6),
             'joint_mean':round(float(b.mean()),6),'diff_pp':round(float(d.mean()*100),4),
             'joint_min':round(float(b.min()),6),'joint_zero_seeds':int((b==0).sum()),
             'pair_wins':int((d>1e-9).sum()),'pair_ties':int((abs(d)<1e-9).sum()),'pair_losses':int((d<-1e-9).sum())})
    return {'status':'posthoc-exploratory','protocol':raw['protocol'],'summary':result,
        'sha256_raw':hashlib.sha256(data).hexdigest()}

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--input',type=Path,default=Path('results/size_sweep_exploratory.json'))
    pa.add_argument('--reference',type=Path,default=Path('results/size_sweep_summary.json'))
    args=pa.parse_args()
    result=summarize(args.input)
    expected=json.loads(args.reference.read_text())
    if result!=expected:raise AssertionError('Recorded exploratory scaling summary differs from raw measurements')
    print('PASS: regenerated',len(result['summary']),'size × test-family cells from 384 raw draw rows; SHA-256',result['sha256_raw'])
if __name__=='__main__':main()
