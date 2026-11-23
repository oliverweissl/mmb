# MMBA: Multi-Modal Boundary Analyzer for vision-language models

MMBA (package `mmb`) finds the inputs at which a vision-language model (VLM) changes its answer. It edits image regions and question words through one vector of edit strengths, scores every edited input by the teacher-forced log-likelihood of each candidate answer, and searches until the preferred answer flips.
Each crossing is then reduced, delta-debugging style, to a boundary pair: two inputs one edit apart that the model answers differently, confirmed by scoring each input on its own.
This repository holds the tester, the equal-budget comparison harness (MMBA, random sampling and five search baselines), and the analysis that turns stored runs into every table and figure of the paper.

## Install

Python 3.12 and, for real models, a CUDA 12.8 GPU (the study used one 16 GB RTX 5080). Tests, the analysis and the stub runs need no GPU.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
# exact study environment (torch 2.11.0+cu128, transformers 5.23.0, pymoo 0.6.1.6, ...)
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e . --no-deps
# or, unpinned: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && pip install -e ".[dev]"
python -m spacy download en_core_web_sm       # text operators (already in the lock file)
pytest -q                                      # model-backed tests skip themselves without cached weights
```

`requirements-lock.txt` is `pip freeze` of the study's environment; its header records the Python, CUDA and driver versions.

## Data and models

Nothing is redistributed here; images and weights are fetched from their publishers (see [EXP_SETUP.md](EXP_SETUP.md), section 2, and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)).

Both tasks expect their files under `~/data` (`MMB_DATA`); the seed-input lists reference images as `~/data/cells/<task>/...`.

- **ImageNet** (validation split): a local parquet snapshot of the ungated Hugging Face mirror `Tsomaros/Imagenet-1k_validation` (same label indices as ILSVRC-2012 val):
  ```bash
  huggingface-cli download Tsomaros/Imagenet-1k_validation --repo-type dataset --local-dir ~/data/imagenet-val-hf
  export MMB_IMAGENET_DIR=~/data/imagenet-val-hf
  ```
- **POPE / COCO**: the parquet of `lmms-lab/POPE` (adversarial split; it embeds the COCO val2014 images):
  `huggingface-cli download lmms-lab/POPE --repo-type dataset --local-dir ~/data/pope`.
- **Seed inputs**: `cells/imagenet_cells.yaml` and `cells/pope_cells.yaml` fix the seed inputs shared by every model (identifiers, class pairs and anchors only). Their images are extracted from the two snapshots, deterministically, by the pool step of `scripts/build_cells.py` (EXP_SETUP.md, section 4):
  ```bash
  python scripts/build_cells.py pool imagenet --out /tmp/imagenet_pool.yaml --per-class 4   # writes ~/data/cells/imagenet/
  python scripts/build_cells.py pool pope --out /tmp/pope_pool.yaml --n 200                 # writes ~/data/cells/pope/
  ```
  The committed seed-input lists are not rebuilt; the `score` and `select` steps are needed only for a new design.
- **Models** (Hugging Face ids): `Qwen/Qwen3.5-4B`, `OpenGVLab/InternVL3_5-4B-HF`, `mistralai/Ministral-3-3B-Instruct-2512`, `llava-hf/llava-v1.6-mistral-7b-hf` (NF4), and the VQGAN f8-16384 image codebook. They download on first use.

## Regenerating the results from stored runs (CPU only)

Stored runs live in `runs/MAIN/<task>/<sut>/<method>/s<seed>/<run>/`; the replay CSVs (greedy agreement, transfer, minimal crossings) in `results/`. Every table, figure and number of the paper comes from one command:

```bash
PYTHON=python scripts/analysis/export_to_paper.sh <paper_repo>     # writes <paper_repo>/generated/
```

It collects the runs (`python -m mmb.analysis.collect runs/MAIN --out results/MAIN.csv --exp-id MAIN --require-stats`), scores the validity proxies (`python -m mmb.validity.score runs/MAIN --out results/validity_MAIN.csv --default exp_id=MAIN --offline`) and runs `python -m mmb.analysis.export`. Per research question:

| RQ | what | command (after the collect step above) |
|---|---|---|
| RQ1 effectiveness | crossing rate, stealth, LPIPS (Table "RQ1"), transfer, greedy agreement | `jupyter nbconvert --to notebook --execute notebooks/rq1_effectiveness.ipynb --output-dir /tmp` |
| RQ2 efficiency | calls to first and minimal crossing, run time, crossing curve | `... notebooks/rq2_efficiency.ipynb`; figure: `python -m mmb.analysis.plots results/MAIN.csv --out results/figures/crossing_curve.pdf` |
| RQ3 validity | automatic validity proxies; rating set for the human study | `... notebooks/rq3_validity.ipynb`; `python -m mmb.validity.rating_set results/MAIN.csv --out study/main_validity` |
| RQ4 ablation | variants pooled over models and tasks | `... notebooks/rq4_generalization.ipynb` (the ablation table is written by the export) |
| Further analysis | failure types, boundary hardness | `python -m mmb.analysis.min_types runs/MAIN --out results/minimal_types_MAIN.csv`; `python -m mmb.analysis.hardness results/MAIN.csv runs/MAIN` |
| Fig. 1 | minimal crossings of MMBA beside random search | written by the export; the row rule is `select_teaser` in `mmb/analysis/gallery.py` |

The notebooks only call functions of `mmb.analysis` and `mmb.validity` and are committed without outputs (`pip install nbconvert ipykernel`). `N_BOOT` defaults to 2000 in the notebooks; the paper uses 10000.

The replay CSVs are produced on the GPU from the stored runs: `scripts/agreement_replay.py runs/MAIN --sut <sut> --out results/agree_MAIN_<sut>.csv`, `scripts/transfer_replay.py runs/MAIN --arms a0_directed a1_random_tiered --out results/transfer_MAIN.csv`.

## Re-running one configuration (GPU)

`configs/generated/MAIN/` has one YAML per experiment configuration, named `<block>__s<seed>__<task>__<sut>__<method>.yaml` (`a0_directed` is MMBA, `a1_random_tiered` random sampling, the others the baselines and ablation variants).

```bash
export MMB_IMAGENET_DIR=...                       # see Data and models
mmb-evolve configs/generated/MAIN/A0__s0__imagenet__qwen__a0_directed.yaml --device cuda
#   = python -m mmb.runners.run_boundary_test <config> --device cuda   (writes runs/MAIN/imagenet/qwen/a0_directed/s0/...)
python scripts/confirm_crossings.py runs/MAIN --sut qwen
#   re-scores each run's minimal crossing and its boundary partner as single inputs; writes min_cross.json
```

The whole matrix runs strictly serially and resumably with `scripts/run_queue.sh configs/generated/MAIN`. The design (seed inputs, seeds, budget of 30 x 50 = 1,500 proposals per run) is identical for every task, model and method; see [EXP_SETUP.md](EXP_SETUP.md) and [METHODOLOGY.md](METHODOLOGY.md).

## What a run produces

`runs/<exp>/<task>/<sut>/<arm>/s<seed>/<name>_seed_<cell>_<ts>/`:

| file | content |
|---|---|
| `trace.parquet` | one row per evaluated individual: genotype, per-candidate log-likelihoods, probabilities, predicted answer, cache flag, fitness columns |
| `convergence.parquet` | one row per generation |
| `stats.json` | seed metadata, runtime, `run_meta` (arm-defining settings and tags), `anchor_g` (the unmodified seed's pair contrast), `config_snapshot` |
| `origin.png`, `first_cross.png`, `first_cross.json`, `pareto_*` | the unmodified seed, the first crossing input and the Pareto front |


## Layout

`mmb/evolutionary` search loop and trace writer; `mmb/optimizer` AGE-MOEA-II, NSGA-II, GA, random and hill-climber arms; `mmb/manipulator` image and text operators; `mmb/objectives`; `mmb/sut` VLM scorers; `mmb/pdq` label-only mapping stage; `mmb/analysis`, `mmb/validity`; `configs/`; `cells/` fixed cell lists (identifiers only); `scripts/`; `tests/`.

## Data and weights are not included

ImageNet and COCO images and all model weights are fetched at run time from their publishers (EXP_SETUP.md section 2) and are never redistributed here; cell lists carry identifiers only. Weights keep their publishers' licences, which can be more restrictive than this repository's.

## Licences

- Source code: [LICENSE](LICENSE) (PolyForm Noncommercial 1.0.0).
- Documentation, derived data and figures: [LICENSE-DATA](LICENSE-DATA) (CC BY-NC 4.0).
- Third-party code, datasets and model weights, including terms that override the two licences above: [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

[ARTIFACT.md](ARTIFACT.md) describes the anonymous review artifact and how it is built.
