# When Relative Reports Agree but Mislead — WITNESS

Research software and measured outputs for a study of **noisy relational supervision in directed neural graph reachability**. This is the executable companion to the anonymous TMLR-style research draft, not an announcement of journal acceptance.

## What the experiments establish

The training procedure uses one verified anchor label per four-query counterfactual group and six noisy pairwise parity reports. It compares Bayesian posterior soft labels, a joint constraint-marginal likelihood, and task-adapted versions of spectral synchronization and Dawid–Skene EM. The original training set contains **24 graph quadruples**. In the expanded validation, **12 model/report seeds**, five original-generator evaluation draws, and three additional random-arborescence draws were used. The study records 96 neural training runs.

Across the new arborescence evaluation draws, complete-quadruple accuracy averaged:

| Noise | Posterior soft labels | Joint marginalization | Spectral labels | Dawid–Skene-derived labels |
|---|---:|---:|---:|---:|
| Heavy independent | 85.9% | 71.6% | 87.3% | 35.9% |
| Correlated | 89.1% | 85.9% | 78.8% | 61.9% |

The two posterior/joint arms in the correlated condition are **both** mixture-aware. These are synthetic-benchmark averages, not evidence of superiority on natural networks; paired uncertainty intervals do not establish a stable ranking. The random-tree tests still share the mirrored degree-preserving intervention construction. The methods derived from Singer and Dawid–Skene are **mathematical adaptations to parity reports, not runs of original authors' end-to-end benchmark software**.

## Quick start — complete archived experiment

1. Download or clone this repository. Python 3.11–3.13 is expected to work, although cross-platform numeric results can vary.
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate                 # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

3. Extract `reproducibility/WITNESS_TMLR_Validated_Supplement.zip`, preserving paths, then run:

   ```bash
   cd WITNESS_TMLR_Validated_Supplement
   PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement python -m pytest -q
   PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement python analyze_validation.py
   ```

   On Windows PowerShell, set `$env:PYTHONPATH='.;WITNESS_TMLR_Anonymized_Supplement'` before the Python commands. The complete archive includes the raw predictions and numerical results. **55 automated tests passed locally before release.** A GitHub Actions workflow independently checks the committed ZIP and runs the tests and summary reconstruction.

4. Optional full neural retraining (CPU-intensive):

   ```bash
   PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement python run_neural_validation.py --seeds 12
   ```

This repository also exposes the Python source as ordinary browseable files, preserving the layout from the archived experiment. Results over 100 KB are not duplicated in the browseable tree; the **checksum-pinned archive is the complete measured dataset**.

## Audit and limitations

- [`docs/SCIENTIFIC_VALIDATION.md`](docs/SCIENTIFIC_VALIDATION.md): measured numbers, provenance, and caveats.
- [`docs/REVIEWER_GAPS.md`](docs/REVIEWER_GAPS.md): what remains open before a TMLR submission.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): grouping of train/report/model/test randomness, data-size experiment proposal, and cost-control requirements.
- [`docs/ARCHIVE_README.md`](docs/ARCHIVE_README.md): original packaged-run instructions.
- [`paper/main.tex`](paper/main.tex): anonymous TMLR-layout LaTeX source, with complete figures/style bundled separately in `paper/WITNESS_TMLR_Validated_Source.zip`.

The original MIT license in this repository is retained. All experiments are synthetic. No confidential data or human subjects are used.

### Double-blind submission warning

**This GitHub repository is public and linked to a named GitHub account. Its URL and commit history could reveal author identity. Do not put this URL into a double-blind TMLR submission or anonymized supplementary material.** Submit the separate anonymized supplement through the review platform, subject to the venue's current policies; the public repository is intended for later open reproducibility and collaboration.
