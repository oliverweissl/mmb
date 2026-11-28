# EXP_SETUP: how every number in the paper is produced

Operational recipe for the continuous design (2026-11-28). Decisions and reasons: [METHODOLOGY.md](METHODOLOGY.md);
design notes: [docs/METHODOLOGY_REVIEW_2026-11-28.md](docs/METHODOLOGY_REVIEW_2026-11-28.md),
[docs/SEARCH_REAL_CODED.md](docs/SEARCH_REAL_CODED.md), [docs/DETERMINISM_2026-11-28.md](docs/DETERMINISM_2026-11-28.md).
The integer-genotype setup (MAIN campaign of 2026-11-20) is in git history before a50514a.

## 1. Hardware and software
- 1x NVIDIA RTX 5080 16 GB, 32 CPU threads, 62 GB RAM. GPU jobs run strictly serially (`scripts/run_queue.sh`).
- Python 3.12; exact versions in `requirements-lock.txt`.

## 2. Data and models
| item | source | notes |
|---|---|---|
| ImageNet-1k validation | HF `Tsomaros/Imagenet-1k_validation` (ungated mirror, ILSVRC-2012 label indices) | `MMB_IMAGENET_DIR` |
| POPE | HF `lmms-lab/POPE` (adversarial split, COCO val2014 images) | |
| VQGAN f8-16384 | latent-diffusion `vq-f8` | image codec (fp32) |
| SUT qwen | `Qwen/Qwen3.5-4B`, bf16 | |
| SUT internvl | `OpenGVLab/InternVL3_5-4B-HF`, bf16 | |
| SUT ministral | `mistralai/Ministral-3-3B-Instruct-2512`, FP8 checkpoint dequantised to bf16 | |
| question pool models | ModernBERT-large, nli-deberta-v3-base, roberta-base-CoLA, all-MiniLM-L6-v2, T5 paraphraser | offline only (`scripts/build_question_pool.py`) |

Scoring: full-sequence teacher forcing (length-normalised log-likelihood of every candidate); no INT8 / FP8 activations.

## 3. Seed inputs (identical for every SUT and method)
`cells/<task>_cells.yaml`: 18 study items per task; `cells/<task>_calib.yaml` and `_calib2.yaml`: 4 + 4 held-out
calibration items per task. Built by `scripts/build_cells.py` (pool -> score -> select; an item is eligible when every
SUT answers the unedited input with a margin of at least 1 nat). The lists hold identifiers only.

## 4. Manipulation (one genotype, two blocks)
- Genotype `[c in [-1,1]^16 | t in [0,1], u in [0,1)]`; all-zero = the unedited input (`anchor_recon.png`).
- Image: `z = z0 + w_img * m_max * ||z0|| * (B c) / sqrt(d)` on the VQGAN latent, DCT basis B, d = 16, m_max = 0.2,
  decoded without re-quantisation (`mmb/manipulator/image/latent.py`).
- Text: `question_pool` operator: target distance `w_txt * t * d_max` picks the nearest non-empty distance band of the
  validated pool of whole questions (`configs/text/question_pool.json`, d_max = 0.278, 20 bands); u picks a member;
  t = 0 is the original question. Pool review: `docs/figs/text_pool/review.html`.
- Magnitudes: mag_image = w_img * RMS(c), mag_text = w_txt * t, magnitude = L2 of the two. w_img, w_txt are applied
  by the manipulator; the optimizer never sees them.

## 5. Calibration of (w_img, w_txt)
Each step runs the text-only (a6), image-only (a7) and MMBA (a0) arms on the calibration items, all SUTs, rng seed 0,
full budget. Target: text-only and image-only cross about equally often (exact McNemar on matched pairs, p >= 0.05),
both mostly crossing; raise w_txt before lowering w_img.
```
python -m mmb.analysis.calibration configs --base '<a6/a7/a0 YAMLs>' --step <name> --w-img X --w-txt Y
scripts/run_queue.sh configs/generated/CALIB/<name>
python -m mmb.analysis.calibration analyze runs/CALIB/<step1> runs/CALIB/<step2> ... --out results/calib --final CALIB_<adopted>
```
Outputs: `results/calib/trajectory.csv`, `calib-trajectory.pdf`; the export turns them into ECalibTraj* macros.

## 6. Matrix (MAIN)
| tasks | SUTs | methods | rng seeds | budget |
|---|---|---|---|---|
| imagenet, pope | qwen, internvl, ministral | a0 MMBA, a5 MIMICRY-style, a9 DeepJanus-style, a6 text-only, a7 image-only, x1 MMBA w/o guidance | 0, 1 | 30 generations x 50 = 1,500 calls, early stop off |

Configs: `configs/generated/MAIN/<block>__s<seed>__<task>__<sut>__<method>.yaml` (`scripts/make_arm_configs.py`),
run by `scripts/run_queue.sh configs/generated/MAIN`. After the search the tester confirms the smallest in-budget
crossing and the boundary pair on the canonical path (extra calls, outside the budget).

## 7. Replays (GPU, after the matrix)
`scripts/agreement_replay.py`, `scripts/transfer_replay.py`, `scripts/modality_swap.py` (canonical scoring of the
stored crossings; `scripts/replay_maintenance.sh` runs all three).

## 8. Analysis and export (CPU)
`scripts/analysis/export_to_paper.sh <paper_repo>`: collect -> validity proxies -> `mmb.analysis.export` (tables,
figures, numbers.tex). Quick check: `scripts/analysis/run_analysis.sh runs/MAIN MAIN`.
