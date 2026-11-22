# mmb: multi-modal boundary testing of vision-language models

`mmb` searches for the inputs at which a vision-language model (VLM) changes its answer. Given a seed (an image and a question with a fixed set of candidate answers), it edits the image (VQGAN codebook swaps) and the question (synonym, fragmentation, homoglyph and saliency operators) through one discrete genotype, scores every edited input by the teacher-forced log-likelihood of each candidate answer, and searches for **crossings**: inputs whose answer differs from the seed's. A second, label-only stage can expand a crossing into a minimal crossing edge.

The repository also contains the equal-budget comparison harness (eight search arms, three SUT families, two tasks), the analysis pipeline that turns run directories into the paper's tables and figure, and an automatic validity-proxy scorer. Design decisions and their reasons are in [METHODOLOGY.md](METHODOLOGY.md); the operational recipe for every number is in [EXP_SETUP.md](EXP_SETUP.md).

## Install

Python >= 3.11. A GPU is needed only to run real VLMs; tests, the analysis pipeline and the stub run below need none.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                          # package, pytest, ruff
python -m spacy download en_core_web_sm          # text operators
pip install matplotlib                           # analysis figure
pip install lpips scikit-image sentence-transformers   # only for mmb.validity
```

Extras: `.[stylegan]`, `.[diffusion]` (alternative image backends), `.[cache]` (Redis score cache). 8-bit SUTs additionally need `bitsandbytes`.

## Quick start without a GPU

1. **Tests.** The whole evolutionary loop runs against a fake SUT (no model, no images downloaded):
   ```bash
   pytest -q tests/test_evolutionary.py tests/test_analysis_collect.py
   ```
2. **Analysis on synthetic runs.** The fixtures write run directories with the same files the tester writes, so the pipeline can be tried end to end (from the repository root):
   ```python
   import sys; sys.path.insert(0, "tests")
   import numpy as np
   from pathlib import Path
   from test_analysis_fixtures import write_run
   from mmb.analysis import collect, stats, plots

   rng = np.random.default_rng(0)
   root = Path("/tmp/mmb_quickstart")
   for arm, algo, p in [("a0", "agemoea2", 0.012), ("a1", "random", 0.004)]:
       for cell in range(10):
           g = np.abs(rng.normal(2, 1, 1500)) + 0.05        # pair contrast of 50 generations x 30 individuals
           if rng.random() < (0.8 if arm == "a0" else 0.5):
               k = int(rng.geometric(p))
               if k < 1500:
                   g[k] = -0.1                              # a crossing
           write_run(root / arm, f"{arm}_{cell}", g, pop=30, seed_idx=cell,
                     config={"optimizer": {"algorithm": algo}, "seed": 0})
   df = collect.collect(root, defaults={"exp_id": "demo"})
   stats.run_all(df, "a0_directed", root / "tables")                 # CSV and booktabs tables
   plots.plot_crossing_curve(df, root / "curve.pdf", ref_arm="a0_directed")
   ```
3. **A real run** needs weights and data (see below): `mmb-evolve <config.yaml> --device cuda`. `configs/paper/smoke_boundary_pair.yaml` is a minimal budget configuration for checking the data flow of the combined pipeline (`mmb-pair`) on a real model; it makes no scientific claim.

## Reproducing the paper's experiments

The comparison design is balanced: every task x SUT x arm uses the same cells, the same random seeds and the same budget (population 30 x 50 generations = 1,500 proposals, early stopping off). [EXP_SETUP.md](EXP_SETUP.md) lists hardware, data, models and the matrix; in short:

```bash
# 1. cells shared by every SUT (ImageNet roster cells; POPE yes/no items)
python scripts/build_cells.py pool imagenet --out cells/imagenet_pool.yaml --per-class 4
python scripts/build_cells.py pool pope --out cells/pope_pool.yaml --n 200
python scripts/build_cells.py score cells/<task>_pool.yaml --sut <model_id>     # once per SUT
python scripts/build_cells.py select cells/<task>_pool.yaml --k K --tau 1.0 --out cells/<task>_cells.yaml

# 2. the search arms and the experiment matrix (E1: arms; E3: SUTs; E4: tasks)
python scripts/make_arm_configs.py configs/templates/evolutionary_template.yaml configs/paper/arms
python scripts/make_matrix.py --exp E1 --tasks imagenet --suts llava --arms all --rng-seeds 0 1 2
scripts/run_queue.sh configs/generated/E1          # strictly serial, resumable

# 3. tables and figure
scripts/analysis/run_analysis.sh runs/E1 E1 a0_directed
scripts/validity/run_validity.sh runs/E1 E1          # E7: descriptive validity proxies
```

K and the number of seeds are fixed after a speed benchmark (EXP_SETUP.md section 3). E3 and E4 use the same commands with `--exp E3` / `--exp E4`.

## Reproducing the results

The results are computed from the run directories under `runs/MAIN` (one directory per run, written by the search). Everything after the runs is CPU only.

1. Collect the runs into tables: `python -m mmb.analysis.collect runs/MAIN --out results/MAIN.csv --exp-id MAIN`, and the validity proxies of the crossing inputs with `python -m mmb.validity.score runs/MAIN --out results/validity_MAIN.csv --default exp_id=MAIN --offline` (the proxy models are loaded from the local cache).
2. Work through the notebooks in `notebooks/`, one per research question, in this order:
   - `00_collect.ipynb`: builds or reloads `results/MAIN.csv` and `results/validity_MAIN.csv` and checks completeness.
   - `rq1_effectiveness.ipynb`: crossing shares, paired tests against the directed arm, the primary comparison, the secondary comparisons.
   - `rq2_efficiency.ipynb`: crossing curve over the budget, cost of the crossing input, wall-clock time.
   - `rq3_validity.ipynb`: validity proxies, the a-priori thresholds, the visible text view.
   - `rq4_generalization.ipynb`: per-SUT and per-task strata, transfer, agreement with greedy generation, predictors of hardness.
   Each notebook only calls functions of `mmb.analysis` and `mmb.validity`; it contains no analysis code of its own. They are committed without outputs. Run one with `jupyter nbconvert --to notebook --execute notebooks/rq1_effectiveness.ipynb --output-dir /tmp` (needs `pip install nbconvert ipykernel`, and the repository on the Python path). `N_BOOT` at the top of each notebook defaults to 2000 for speed; the paper uses 10000.
3. The tables, figures and numbers of the paper come from one command: `PYTHON=python scripts/analysis/export_to_paper.sh <paper_repo>`. It writes `<paper_repo>/generated/` from the same functions.

## What a run produces

`runs/<exp>/<task>/<sut>/<arm>/s<seed>/<name>_seed_<cell>_<ts>/`:

| file | content |
|---|---|
| `trace.parquet` | one row per evaluated individual: genotype, per-candidate log-likelihoods, probabilities, predicted answer, cache flag, fitness columns |
| `convergence.parquet` | one row per generation |
| `stats.json` | seed metadata, runtime, `run_meta` (arm-defining settings and tags), `anchor_g` (the unmodified seed's pair contrast), `config_snapshot` |
| `origin.png`, `first_cross.png`, `first_cross.json`, `pareto_*` | the unmodified seed, the first crossing input and the Pareto front |

The analysis layer: `python -m mmb.analysis.collect <runs> --out results/<exp>.csv` (one row per run: crossing, first crossing evaluation, floor, label flip, magnitudes), `python -m mmb.analysis.stats <csv> --ref a0_directed` (Wilson intervals, paired tests with Holm correction, bootstrap bands, LaTeX tables), `python -m mmb.analysis.plots <csv>` (crossing share against budget), `python -m mmb.analysis.predictors <csv>` (E2b: how well the gen-0 probe, the content-free prior and the seed confidence gap predict crossing and floor), and `python -m mmb.validity.score <runs> --out <csv>` (image and text similarity of each crossing input to its seed).

## Layout

`mmb/evolutionary` search loop and trace writer; `mmb/optimizer` AGE-MOEA-II, NSGA-II, GA, random and hill-climber arms; `mmb/manipulator` image and text operators; `mmb/objectives`; `mmb/sut` VLM scorers; `mmb/pdq` label-only mapping stage; `mmb/analysis`, `mmb/validity`; `configs/`; `cells/` fixed cell lists (identifiers only); `scripts/`; `tests/`.

## Data and weights are not included

ImageNet and COCO images and all model weights are fetched at run time from their publishers (EXP_SETUP.md section 2) and are never redistributed here; cell lists carry identifiers only. Weights keep their publishers' licences, which can be more restrictive than this repository's.

## Licences

- Source code: [LICENSE](LICENSE) (PolyForm Noncommercial 1.0.0).
- Documentation, derived data and figures: [LICENSE-DATA](LICENSE-DATA) (CC BY-NC 4.0).
- Third-party code, datasets and model weights, including terms that override the two licences above: [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

[ARTIFACT.md](ARTIFACT.md) describes the anonymous review artifact and how it is built.
