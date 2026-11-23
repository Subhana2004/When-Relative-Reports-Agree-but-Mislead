# WITNESS — Published-Estimator and Topology Validation

## Execution and provenance

- **Study completed:** 12 model/report seeds × 4 training arms × 2 noise regimes = 96 complete neural training runs.
- **Evaluation:** 5 original held-out graph draws (24 quadruples/draw) and 3 independently generated random-tree graph draws (48 quadruples/draw). This creates 768 arm–seed–draw evaluation rows; each row retains individual correct/incorrect outcomes.
- **Original-result cross-check:** Re-trained original Bayesian/joint arms matched the pre-existing per-draw correctness scores exactly (maximum CQA discrepancy = 0).
- **Label reconstruction:** 128 new held-out graph quadruples × 8 report draws = 1,024 examples in each noise regime; the unsupervised Dawid–Skene model was fitted on separate graph seeds and without evaluation labels.
- **Additional topology validation:** 144 independent random-arborescence graph quadruples spanning four topology families and three component sizes, checked for reachability, exact degree-preserving intervention, and graph isomorphism. All passed the parity-cycle gauge-invariance audit.
- **Algorithm provenance:** Singer (2011) top-eigenvector synchronization specialized to binary signs; Dawid–Skene (1979) confusion-matrix EM; an additional E/M update-equation specialization checked against WRENCH `dawid_skene.py`, upstream Git blob SHA `7c59cf863f2630c6d9cc89939acb0c5be049e2f9`. None of these results are represented as original authors’ full-system executions or original published benchmark replications.

## Held-out noisy-label reconstruction

All figures below are measured proportions of **complete four-label recovery**, not neural correctness.

| Label-recovery method | Heavy noise | Correlated noise |
|---|---:|---:|
| Tree completion | 31.9% | 39.3% |
| Unweighted MAP | 39.3% | 50.3% |
| Confidence-weighted MAP | 48.0% | 54.9% |
| Singer-style weighted spectral | 48.0% | 55.1% |
| Regularized Dawid–Skene | 37.0% | 44.9% |
| WRENCH-style unsmoothed EM | 0.0% | 44.9% |
| Bayesian soft (thresholded) | 46.9% | 54.4% |

The weighted spectral method is close to exact weighted MAP in this small four-query problem. The comparison does not establish equivalence in general.

The path-derived votes used by Dawid–Skene share underlying parity reports and violate its conditional-independence assumption. The unsmoothed WRENCH-style EM variant collapsed under heavy noise, with an estimated positive-class prior near 0.978. This is a result about the WITNESS adaptation, not a general critique of Dawid–Skene.

## Matched neural performance

Complete-quadruple accuracy means across all 12 seeds and graph draws.

| Noise | Training method | Original generator | New random-tree topologies |
|---|---|---:|---:|
| Heavy | Bayesian posterior soft | 84.7% | 85.9% |
| Heavy | Joint marginal | 68.9% | 71.6% |
| Heavy | Singer spectral labels | 80.6% | 87.3% |
| Heavy | Dawid–Skene soft labels | 32.7% | 35.9% |
| Correlated | Mixture-aware soft | 88.5% | 89.1% |
| Correlated | Mixture-aware joint | 85.4% | 85.9% |
| Correlated | Singer spectral labels | 70.9% | 78.8% |
| Correlated | Dawid–Skene soft labels | 53.3% | 61.9% |

## Paired exploratory uncertainty — new topologies

| Paired comparison | Mean difference | Two-way 95% percentile interval |
|---|---:|---:|
| Heavy: Joint vs Bayesian soft | -14.4 pp | [-39.4, +4.3] pp |
| Heavy: Spectral vs Bayesian soft | +1.4 pp | [-1.3, +5.5] pp |
| Heavy: Dawid–Skene vs Bayesian soft | -50.1 pp | [-75.3, -24.9] pp |
| Correlated: Mixture joint vs mixture soft | -3.2 pp | [-22.6, +10.9] pp |
| Correlated: Spectral vs mixture soft | -10.4 pp | [-29.7, +2.6] pp |
| Correlated: Dawid–Skene vs mixture soft | -27.2 pp | [-50.8, -6.9] pp |

## Scientific verdict

**Validated:** exact reachability oracle checks, paired controlled evaluation, additional algorithm-level baselines, seed-level failure reporting, and one new synthetic graph-topology shift. The original matched controls reproduce their recorded values.

**Not established:** a new superior WITNESS algorithm; a performance advantage across independent populations; a completely independent natural-graph benchmark; faithful executions of published WEASEL or other end-to-end weak-supervision systems on their authors’ original benchmark; exact matched total computation. A source-equation adaptation is not a full external software reproduction.

**TMLR submission status: scientific validation still conditional.** The current paper is an anonymized TMLR-formatted research draft. Before submission, independently verify source equivalence and add natural-network data / faithfully reproduced published end-to-end systems, then obtain author review of all claims and requirements.

### Published primary sources

- Singer, *Angular Synchronization by Eigenvectors and Semidefinite Programming*, Applied and Computational Harmonic Analysis 30(1):20–36 (2011), DOI: https://doi.org/10.1016/j.acha.2010.02.001
- Dawid and Skene, *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm* (1979), DOI: https://doi.org/10.2307/2346806
- WRENCH official source file: https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/dawid_skene.py (Git blob SHA above).
- TMLR official author guide: https://www.jmlr.org/tmlr/author-guide.html

### Reproducibility

- Run `pytest -q` from the root of the supplement with `PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement`.
- Run `python analyze_validation.py` to regenerate the statistical summary from measured JSON.
- Use `run_label_validation.py` and `run_wrench_style.py` to repeat the label-model experiments.
- Use `run_neural_validation.py --seeds 12` to retrain all neural arms; CPU time will vary by environment.
- Compile the source manuscript twice with `pdflatex -interaction=nonstopmode main.tex`.

Generated using the included measured raw result files. No manuscript acceptance or external upstream benchmark execution is claimed.
