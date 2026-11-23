# Exploratory WITNESS training-set-size sensitivity (measured)

**Status:** completed, post-hoc **feasibility/pilot analysis**. It was prompted by reviewer-style criticism after inspection of the original TMLR draft. It must not be represented as preregistered confirmation or as a natural-network study.

## Design

- **Training sizes:** 24, 48, 96, 192 certified quadruples, nested and balanced within eight strata (two component sizes × two directed generator families × two intervention-effect conditions).
- **Independent training-graph construction seeds:** two (`314159`, `271828`).
- **Initialization/noise seeds:** three (`71`, `73`, `79`), coupled as a common seed for model initialization and weak-report simulation; this study **does not separate** those two sources of randomness.
- **Noise:** heavy independent parity reporting noise only. The correlated-noise scaling experiment has **not** been run.
- **Methods:** Bayesian posterior soft-label training (`weighted_soft`) and the candidate independent joint constraint marginalization (`joint_independent`). The same noisy reports/anchor inputs, GNN architecture, learning rate, training schedule, and initialization are used within matched cells.
- **Matched compute:** exactly 192 optimizer steps and 768 graph forwards (four graphs per quadruple update) per trained model. Larger training sets do **not** receive more optimizer steps. A 192-quadruple set is presented once; 24-quadruple examples recur eight times. Posterior/likelihood preprocessing arithmetic is **not** cost-matched.
- **Evaluations:** five original-generator draws of 24 quadruples each and three alternate arborescence-generator draws of 48 quadruples each, fixed across methods. These synthetic test families were known from preceding studies; this is exploratory and not an independently blinded confirmation.
- **Measured totals:** 2 training-data seeds × 4 sizes × 3 model/report seeds × 2 methods = **48 complete training runs**, each evaluated on 8 test draws = **384 draw-level records** with per-quadruple correctness arrays.

## Complete-quadruple accuracy (mean % across six trained model replicates per size)

| Test family | Train quadruples | Bayesian soft | Joint marginal | Joint minus soft (pp) | Joint model replicates at exactly 0% CQA |
|---|---:|---:|---:|---:|---:|
| original | 24 | 95.8% | 99.7% | +3.9 | 0/6 |
| original | 48 | 98.6% | 83.3% | -15.3 | 1/6 |
| original | 96 | 96.8% | 79.7% | -17.1 | 0/6 |
| original | 192 | 98.6% | 98.3% | -0.3 | 0/6 |
| arborescence | 24 | 99.9% | 99.9% | +0.0 | 0/6 |
| arborescence | 48 | 99.9% | 83.3% | -16.6 | 1/6 |
| arborescence | 96 | 100.0% | 83.6% | -16.4 | 0/6 |
| arborescence | 192 | 100.0% | 100.0% | +0.0 | 0/6 |

## Interpretation

The striking pattern is **non-monotonic failure of the joint objective under a fixed optimizer budget**. In one matched data/seed cell at 48 training quadruples, the joint method recorded **0% CQA** on both test families; other cells were near-perfect. At 96 quadruples, multiple joint training cells remained well below their Bayesian counterpart. At 192, the severe failures seen in these six pilot cells disappeared, but the study is too small to establish a reliable scaling law or a method superiority claim.

The alternate arborescence evaluations were frequently at or near **100% even with 24 quadruples**. Therefore, their near-ceiling scores are poor evidence of a meaningful gain with more training data. The original-generator draws retained more variation and should be reported alongside, not replaced by, the easy new draws.

Do not describe every 0%-CQA run as 'one-class collapse' without saving per-query class predictions; CQA alone identifies complete-group failure, not the specific prediction distribution. The saved pilot includes per-quadruple correctness but not class-level logits.

## Critical remaining limits

1. Only **two** independently generated training pools and **three** model/report seeds. No robust statistical superiority or generalizable data-scaling conclusion follows.
2. Training report randomness and neural initialization are coupled rather than independently crossed; vary them separately in a confirmatory factorial replication.
3. This checks **one fixed 192-update budget**. A fixed-epoch comparison is also needed to characterize performance versus total training compute. The original 144-update experiments are not exactly replicated.
4. All test draws are synthetic and development-inspected; neither a natural graph nor non-mirrored network intervention was used in these neural scaling results.
5. Only the two main neural objectives were compared. Published end-to-end baseline software and real-world results remain outstanding.

## Reproduction

```bash
PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement python run_size_sweep.py \
  --sizes 24,48,96,192 --data-seeds 314159,271828 \
  --model-seeds 71,73,79 --updates 192 --regime heavy \
  --output results/size_sweep_exploratory.json
```

The complete measurements and protocol are in `results/size_sweep_exploratory.json`. This file's SHA-256 is `2760d75573daacec273670db0f8e9919ff4043026122280dff14541ccc8f876b`. The run log from the working environment is included in the separate reviewer extension ZIP. Timings are local machine observations, not end-to-end method costs.

**Publication rule:** This pilot may be described as exploratory supplemental evidence. A stronger confirmatory size study with separately varied random factors and broader graph families is still required before changing TMLR effectiveness claims.
