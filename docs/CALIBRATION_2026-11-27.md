> **Superseded (2026-11-28):** integer-genotype design; see docs/METHODOLOGY_REVIEW_2026-11-28.md and METHODOLOGY.md for the current continuous design. Kept for the record.

# Operator calibration (T-calib), 2026-11-27

Goal (user): text-only and image-only manipulation in equilibrium, at a moderate level with headroom, so that full MMBA
(both channels) is clearly better. Use the **weakest** balanced setting (as little input change as possible). Text edits
must keep the question valid; image edits must keep the image valid.
**Equilibrium criterion (user):** text-only and image-only crossing shares are not distinguishable by an exact paired
McNemar test at p = 0.05 on matched (seed input, SUT) runs. *My own addition, not the user's criterion:* a guard of
"within about 10 percentage points", because the test has little power at 24-48 pairs.

All runs: held-out seed inputs only (never the 18 test cells or their images), rng seed 0, full budget (30 x 50 = 1,500
calls), MMBA search, 3 SUTs (Qwen3.5-4B, InternVL3.5-4B, Ministral-3-3B). Search-time (batched) crossings, the same
treatment for every arm. Scripts: scripts/build_calib_cells.py, scripts/make_calib_configs.py,
scripts/analysis/{calib_summary,calib_final,ladder_demo,text_ops_demo,calib_montage}.py. Settings: configs/calib/*.yaml.
Results (untracked): results/CALIB_*.csv, results/calib_*_summary.csv, results/calib_*_tests.csv, results/calib_final/.

## 1. Operators and knobs

**Image (VQGAN f8-16384, region edits).** New: `image.knn_metric: l2_live`, i.e. neighbours by L2 over the 880 live
codewords only. The old default `cosine` over all 16,384 codewords mostly lands on the 15,504 collapsed codes, which
decode to olive patches (docs/EDIT_DIAGNOSIS_2026-11-26.md). The cosine default is kept so the stored MAIN runs replay
unchanged. Knobs:
- `region_levels` (ladder; rank r = r-th nearest live codeword; its length sets the levels per gene);
- `region_block` (region size in codes; 4 = 32 x 32 px, 64 regions).

Strength in interpretable units, LPIPS of ONE edited region against the codec reconstruction (mean over 4 held-out
images per task; scripts/analysis/ladder_demo.py --regions 1):

| rank | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| ImageNet | 0.003 | 0.002 | 0.003 | 0.003 | 0.005 | 0.007 | 0.010 | 0.011 | 0.016 | 0.021 |
| POPE | 0.002 | 0.002 | 0.002 | 0.002 | 0.003 | 0.006 | 0.006 | 0.010 | 0.012 | 0.018 |

The old cosine operator: 6 regions at its weakest level already reach LPIPS 0.088, with 84% dead replacement codes.
Visual ladders: docs/figs/calib/ladder_l2live_wide_imagenet.png and ladder_l2live_pope.png (6 regions per image, every
level). They are realistic up to rank ~128-256; rank 512+ produces garish blobs.

**Text (composite operators).** Knobs per operator:
- `severity` / `k_max`: levels per site. Synonym: depth of the masked-LM pool. Character noise: characters changed
  per word (homoglyph / zero-width, invisible). Typo (saliency): visible letter swaps/replacements per word.
  Fragmentation: inserted spaces.
- Operator mix.
- Sites: `all_words` lets typos hit function words; `min_word_len`.

Protected words are never edited (ImageNet: subject, image; POPE: object noun, image). **Fix:** a multi-word protected
object ("dining table") now protects each of its words; before, it was matched as one token. MAIN check: 0 of 33
minimal and 0 of 33 first crossings of POPE cell 8 edited "dining"/"table".

## 2. Rounds (4 held-out seed inputs per task = set 1; 24 runs per arm)

| round | image setting | text setting | image-only CR | text-only CR | McNemar p | text validity |
|---|---|---|---|---|---|---|
| r1 | rank <= 128 (8 levels) | T1: synonym 5, fragmentation, 3 chars + 3 typos/word, typos on function words | 79.2 (IN 66.7, POPE 91.7) | 54.2 (IN 41.7, POPE 66.7) | 0.146 | INVALID on ImageNet ("What is the foi aq subject", "tpe m a in"); question_preserved 0% on ImageNet |
| r2a | rank <= 32 (6 levels) | T2: synonym 3, 2 invisible chars/word, <= 1 typo on words >= 4 letters, no fragmentation | 45.8 (IN 25.0, POPE 66.7) | 12.5 (IN 8.3, POPE 16.7) | 0.022 | valid |
| r2b | rank <= 8 (4 levels) | (T2, from r2a) | 33.3 (IN 25.0, POPE 41.7) | 12.5 | 0.125 | valid |
| r3t + r2b | rank <= 8 | T3: as T2 but up to 4 invisible chars/word | 33.3 | 29.2 (IN 16.7, POPE 41.7) | 1.000 | mostly valid; question_preserved 86% |

Image validity: no olive dead-code patches at any setting. At rank <= 128 with 20-30 regions, faces visibly distort
(docs/figs/calib/r1_image_crossings.png); at rank <= 8, edits are subtle. Reading: lowering the rank from 32 to 8 hardly
changes the ImageNet share (25%), because the search compensates with more regions (median 22 instead of 8), so the
per-edit strength is not the only lever.

## 3. Final check of the candidate (image rank <= 8 + text T3): set 1 + a disjoint set 2 (8 seed inputs per task, 48 pairs)

| arm | CR pooled | ImageNet | POPE | calls to first (median) | minimal edits (median) | LPIPS (median) | visible text change (median) | question_preserved | output margin (nats, median) |
|---|---|---|---|---|---|---|---|---|---|
| text-only | 18.8 | 12.5 | 25.0 | 1500 | 4 | 0.000 | 3.2% | 77.8% | 0.12 |
| image-only | 33.3 | 29.2 | 37.5 | 1500 | 12.5 | 0.037 | 0% | 100% | 0.02 |
| full MMBA | 50.0 | 37.5 | 62.5 | 1170 | 11.5 | 0.021 | 3.0% | 75.0% | 0.12 |

Paired McNemar tests (exact, discordant pairs in brackets):
- text vs image: pooled p = 0.119 (4 vs 11), ImageNet p = 0.219, POPE p = 0.508. **Not distinguishable, so the user's
  criterion holds.** The pooled gap of 14.5 pp misses my own ±10 pp guard. Text varies between the two sets (29% vs 8%).
- MMBA vs text: pooled p < 0.001 (15 vs 0); MMBA vs image: pooled p = 0.008 (8 vs 0). MMBA never fails where a single
  modality crosses. Per task: ImageNet MMBA vs image p = 0.5 (37.5 vs 29.2, n.s.); POPE both significant.

Text examples at T3 (minimal crossings, ImageNet):
- 'What is the focal subject in this image?' (synonym, invisible characters);
- 'Whae is the importanw subject in this image?' (typos);
- 'What is the dominant subject in thin image?' ("this"->"thin": a typo that makes a real word);
- 'What is the fxcal subject ...' (typo stacked on a synonym).
The last two are the invalid cases. Refinement for the re-run: forbid typos that produce dictionary words, and typos on
synonym-replaced words.

## 4. Recommendation

Setting C* = image `knn_metric: l2_live, region_block: 4, region_levels: [1, 2, 4, 8]` + text T3
(`synonym k_max 3; character_noise severity 0.75 [homoglyph, zero_width]; saliency k_max 1, all_words, min_word_len 4`;
no fragmentation) + the two typo validity rules above. This is the weakest setting tried at which text-only and
image-only are statistically indistinguishable, both are moderate (19-33%), and full MMBA (50%) is significantly better
than each. Per edit: image LPIPS <= 0.003 per region (rank <= 8); text <= 1 visible character per word, otherwise
invisible characters or a top-3 synonym.
Open choice: if the ±10 pp guard should also hold, run one more step with text slightly stronger (e.g. typos also on
3-letter words) or image slightly weaker (fewer regions), ~2.5 h + re-check.

## 5. GPU estimate for the full balanced re-run with C* (measured ~0.84 GPU-h per 18-run config)

| block | configs | GPU-h |
|---|---|---|
| RQ1/RQ2 MMBA (a0), seeds 0+1 | 12 | ~10 |
| RQ1/RQ2 MIMICRY-style (a5), seeds 0+1 | 12 | ~10 |
| RQ1/RQ2 DeepJanus-style (a9), seeds 0+1 | 12 | ~10.5 |
| RQ4 ablations seed 0: text-only a6, image-only a7, w/o guidance x1, patch edits x2 | 24 | ~21 |
| single-input confirmation per block | | ~1.5 |
| agreement + transfer replays (if kept) | | ~3 |
| **total** | **60** | **~56 (~2.4 days on the one GPU)** |
Random search is not re-run (omitted from the paper).
