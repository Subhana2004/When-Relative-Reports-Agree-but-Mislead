# WITNESS — anonymized, expanded experimental supplement

The enclosed code and outputs support the accompanying anonymous TMLR-format research draft.

## Exact scope

- 12 neural model/report seeds, 5 old test draws, 3 new stochastic arborescence test draws.
- 96 complete training runs; 768 seed/arm/draw evaluation records in `results/neural_validation.json`.
- 128 held-out quadruples and 8 report draws each, for **each** of two label-recovery noise conditions.
- Published algorithm families adapted mathematically: Singer (2011) top-eigenvector binary synchronization; Dawid–Skene (1979) EM confusion model. A WRENCH-style no-abstention EM specialization uses update equations inspected at upstream Git blob `7c59cf863f2630c6d9cc89939acb0c5be049e2f9`.
- **Not official package executions.** The algorithms receive task-adapted parity-derived inputs. We did not run original WEASEL or WRENCH benchmark software on their published benchmark tasks.
- The new graph distributions remain **synthetic** and share a mirrored intervention construction. Do not claim natural or real-world generalization.

## Run checks

In this directory after installing `requirements.txt`:

```
PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement python -m pytest -q
PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement python analyze_validation.py
```

The first command runs the previous study tests and the published-algorithm / random-tree tests. The second reconstructs the main validation summary from raw results. To regenerate the individual experiments:

```
PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement python run_label_validation.py
PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement python run_wrench_style.py
PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement python run_neural_validation.py --seeds 12
```

Full neural reproduction requires CPU time and may vary across software/hardware versions. Reproducing the numeric summary from the shipped results is much faster.

## Data and ethics

All input graphs are synthetic. Only the observed anchor is provided as exact truth for weak-supervision training. All graph oracles used for evaluation are computed independently via NetworkX traversal. No private data or human-subject observations are included.

## Provenance

- Singer (2011): https://doi.org/10.1016/j.acha.2010.02.001
- Dawid and Skene (1979): https://doi.org/10.2307/2346806
- WRENCH DawidSkene source: https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/dawid_skene.py

No assertion of journal submission or acceptance is made by this package.
