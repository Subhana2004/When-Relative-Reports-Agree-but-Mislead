# Reviewer-facing gap register (open items; not completed results)

This file separates independently verified evidence from experiments that are still needed. Nothing below is represented as a completed result unless linked to a measured output.

| Issue | Current measured evidence | Status / required work |
|---|---|---|
| Train-set size | Only 24 certified quadruples in the validated neural runs | **Open:** repeat 24, 48, 96, 192 (or a preregistered budget-constrained subset) with fixed evaluation; separate examples from optimizer steps. |
| Uncertainty | 12 model/report seeds, 5 original and 3 alternate test draws | **Partly addressed:** add independent training-data draws, report-noise draws, model-initialization draws, and test-graph draws; cluster/paired analysis, uncertainty over both training and test populations. |
| External generalization | New random-arborescence families are still synthetic and share mirrored intervention | **Open:** natural directed graph dataset or genuinely independent intervention construction, with reachability oracles and matched query balance. |
| Published baselines | Task-adapted Singer-style spectral, Dawid–Skene EM, WRENCH-style update specialization; original WRENCH source inspected | **Open:** faithful executions of original published software plus documented input adaptation, config parity, compute, and licensing. |
| Reproducibility | Complete code/results ZIP, raw per-draw neural rows, 55 passing tests, pinned SHA-256 | **Published in this branch:** verify CI on the actual GitHub commit; no claim of third-party independent replication until one exists. |

**Scientific framing:** failure modes of unreliable relational supervision. The current experiments do not demonstrate a novel superior end-model algorithm, statistically consistent gains, or realistic labeling-cost savings.

**Important:** Publishing to a public personal GitHub account may undermine double-blind review. Do not cite this repository within the anonymized submission.
