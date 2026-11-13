# Plan — Code workstream (owner: main agent, /loop)

Source: `../_sources/thesis-code/` (read-only; 478 .py, 22k LOC in src/, 299 configs, 215MB samples).
Target: `mmb/` — a lean, installable library + small CLI that reproduces the paper's core
pipeline and hosts new backends/tasks.

## Phase 1 — Minimise & restructure
- Keep: `src/` packages needed by the paper pipeline (common, sut, manipulator, objectives,
  optimizer, evolutionary, pdq, boundary_pair, data, config), smoo as pip dep (submodule is empty).
- Drop: archive_alpamayo, infrastructure, slides, notebooks, samples, Archive/ dirs, HS-01 app
  (keep anonymised analysis inputs only if small), 299 configs → templates + one config per
  paper experiment, scratch/, pipeline-explorer.
- Dead-code sweep (vulture + grep), merge `image_stylegan`/`image` behind one `ImageBackend`
  protocol, split 700-1000-line modules where cohesive, `pyproject.toml`, ruff, tests kept green.
- Gate: `pytest` green on the kept subset, import graph has no evolutionary<->pdq cycle.

## Phase 2 — Backends (the generative extension)
Image: `ImageBackend` protocol {prepare, n_genes, apply}. Existing: VQGAN codebook.
 New: (a) StyleGAN-XL latent/W+ edit (exists, port), (b) diffusion backend: latent-noise /
 prompt-embedding-direction / DDIM-inversion-step genes (SD/SDXL via diffusers; low-dim
 genotype = per-block guidance/offset indices), (c) VQ-diffusion style token swap as
 bridge. Text: generative backend `LLMParaphrase` (instruction-tuned LM proposes k ranked
 candidates per position/sentence → same "gene k = k-th candidate" contract), keeping
 the existing MLM-synonym stack as baseline.
Experiments (config + runner, CPU smoke only here — no GPU on this box): same seeds, same SUTs,
compare backends on flip rate, gens-to-flip, L0/L2 distance, validity proxy (CLIP/DINOv3 sim,
LPIPS), diversity. Tiny dry-run harness with stub SUT so CI exercises the whole path.

## Phase 3 — Tasks & SUTs
Beyond ImageNet-pair classification: VQA (VQAv2/GQA two-answer boundary), image captioning
(CLIP-score boundary between two captions), OCR/chart QA, medical/pathology-style closed-set,
hallucination (POPE yes/no), safety-refusal boundary. SUTs beyond LLaVA/Qwen: PaliGemma, InternVL,
SmolVLM, and a CLIP/DINOv3-based zero-shot SUT (weights at ~/Downloads/dino3 = DINOv3 ViT-L, useful
as embedding/validity model). Define a `Task` protocol (seeds, options, score) so a task is ~50 LOC.

## Phase 4 — Hand-off
`docs/` short; `EXPERIMENTS-RUNBOOK.md` for what must be run on a GPU box; cross-read
`../6af72.../GAPS.md` and `EXPERIMENTS-WANTED.md` each loop tick and align.

## Loop cadence
Each tick: pick next unchecked item in `PROGRESS.md`, do it, run tests, record. Stop when Phases 1–3 are checked.
