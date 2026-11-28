# MMBA: Multi-Modal Boundary Analyzer for vision-language models

MMBA (package `mmb`) finds the inputs at which a vision-language model (VLM) changes its answer. One real-valued genotype edits the image (a low-rank displacement of the VQGAN latent) and the question (a choice from a validated pool of whole rephrased questions); every edited input is scored by the teacher-forced log-likelihood of each candidate answer. MMBA searches for a crossing (the preferred answer flips), shrinks it to the smallest magnitude, and returns an ON/OFF boundary pair confirmed by scoring each input on its own.
This repository holds the tester, the balanced comparison harness (MMBA, MIMICRY-style and DeepJanus-style baselines, ablations), the weight calibration and the analysis that turns stored runs into every table and figure of the paper.

Design: [docs/METHODOLOGY_REVIEW_2026-11-28.md](docs/METHODOLOGY_REVIEW_2026-11-28.md) (continuous genotype), [docs/SEARCH_REAL_CODED.md](docs/SEARCH_REAL_CODED.md) (search and MMBA phases), [docs/DETERMINISM_2026-11-28.md](docs/DETERMINISM_2026-11-28.md) (canonical scoring). Decisions: [METHODOLOGY.md](METHODOLOGY.md). Recipe: [EXP_SETUP.md](EXP_SETUP.md), [docs/EXPERIMENTS-RUNBOOK.md](docs/EXPERIMENTS-RUNBOOK.md).

## Install

Python 3.12 and, for real models, a CUDA 12.8 GPU (the study used one 16 GB RTX 5080). Tests, the analysis and the stub runs need no GPU.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
# exact study environment (torch 2.11.0+cu128, transformers 5.23.0, pymoo 0.6.1.6, ...)
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e . --no-deps
# or, unpinned: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && pip install -e ".[dev]"
python -m spacy download en_core_web_sm       # tokenizer of the question pool (already in the lock file)
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
- **Models** (Hugging Face ids): `Qwen/Qwen3.5-4B`, `OpenGVLab/InternVL3_5-4B-HF`, `mistralai/Ministral-3-3B-Instruct-2512`, the VQGAN f8-16384 image codec, and for the offline question pool `answerdotai/ModernBERT-large`, `cross-encoder/nli-deberta-v3-base`, `textattack/roberta-base-CoLA`, `sentence-transformers/all-MiniLM-L6-v2`, `humarin/chatgpt_paraphraser_on_T5_base`. They download on first use.

## Regenerating the results from stored runs (CPU only)

Stored runs live in `runs/MAIN/<task>/<sut>/<method>/s<seed>/<run>/`; replay CSVs (greedy agreement, transfer, modality swap) in `results/`. Every table, figure and number of the paper comes from one command:

```bash
PYTHON=python scripts/analysis/export_to_paper.sh <paper_repo>     # writes <paper_repo>/generated/
```

It runs `python -m mmb.analysis.collect runs/MAIN --out results/MAIN.csv --exp-id MAIN --require-stats` (one row per run; every budget-bound metric from in-budget evaluations only), the validity proxies (`python -m mmb.validity.score`) and `python -m mmb.analysis.export`. A quick look without the paper repo: `scripts/analysis/run_analysis.sh runs/MAIN MAIN` (summary, paired tests vs MMBA, crossing curve). The notebooks `notebooks/rq*.ipynb` call the same functions (committed without outputs).

Replays (GPU, canonical scoring of the stored crossings, `mmb/analysis/replay.py`):

```bash
python scripts/agreement_replay.py runs/MAIN --sut qwen --out results/agree_MAIN_qwen.csv
python scripts/transfer_replay.py runs/MAIN --arms a0_directed a5_mimicry_objective a9_deepjanus --out results/transfer_MAIN.csv
python scripts/modality_swap.py runs/MAIN --sut qwen --out results/modswap/MAIN_qwen.csv
```

## Running (GPU)

`configs/generated/<EXP>/` holds one YAML per configuration (`<block>__s<seed>__<task>__<sut>__<method>.yaml`; `a0_directed` is MMBA).

```bash
export MMB_IMAGENET_DIR=...                       # see Data and models
mmb-evolve configs/generated/MAIN/<config>.yaml --device cuda
#   = python -m mmb.runners.run_boundary_test <config> --device cuda
```

The matrix runs strictly serially and resumably with `scripts/run_queue.sh configs/generated/<EXP>`. Every task, model and method uses the same seed inputs, rng seeds and budget (30 x 50 = 1,500 model calls). Weight calibration: `python -m mmb.analysis.calibration configs|analyze` (EXP_SETUP.md, section 5).

## What a run produces

`runs/<exp>/<task>/<sut>/<arm>/s<seed>/<name>_seed_<cell>_<ts>/`:

| file | content |
|---|---|
| `trace.parquet` | one row per evaluation: real genotype, per-candidate log-likelihoods, predicted answer, cache flag, magnitude diagnostics |
| `stats.json` | seed metadata, `run_meta` (arm, seed, budget, algorithm), `anchor_g`/`prior_g`, w_img/w_txt, genotype slices, config snapshot |
| `anchor_recon.png`, `origin.png` | the unedited input as the model sees it (all-zero genotype), the raw photograph |
| `first_cross.*` | the first crossing |
| `min_cross.*` | the smallest in-budget crossing (magnitude, mag_image, mag_text, question, confirmed on the canonical path) |
| `boundary_pair.json`, `min_cross_neighbour.*`, `boundary_on.png` | the ON/OFF pair (own_phase2 or posthoc_bisection), gap, kind (continuous / text_step), extra calls outside the budget |

## Layout

`mmb/evolutionary` tester and trace writer; `mmb/optimizer` real-coded MMBA, baselines and shared variation; `mmb/manipulator` latent image block and question-pool text block (`configs/text/question_pool.json`, built by `scripts/build_question_pool.py`); `mmb/sut` VLM scorers; `mmb/analysis`, `mmb/validity`; `configs/`; `cells/` fixed seed-input lists (identifiers only); `scripts/`; `tests/`.

## Data and weights are not included

ImageNet and COCO images and all model weights are fetched at run time from their publishers (EXP_SETUP.md section 2) and are never redistributed here; cell lists carry identifiers only. Weights keep their publishers' licences, which can be more restrictive than this repository's.

## Licences

- Source code: [LICENSE](LICENSE) (PolyForm Noncommercial 1.0.0).
- Documentation, derived data and figures: [LICENSE-DATA](LICENSE-DATA) (CC BY-NC 4.0).
- Third-party code, datasets and model weights, including terms that override the two licences above: [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

[ARTIFACT.md](ARTIFACT.md) describes the anonymous review artifact and how it is built.
