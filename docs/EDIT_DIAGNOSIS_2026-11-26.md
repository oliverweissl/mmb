> **Superseded (2026-11-28):** integer-genotype design; see docs/METHODOLOGY_REVIEW_2026-11-28.md and METHODOLOGY.md for the current continuous design. Kept for the record.

# Image-edit realism, channel balance and headline metric: diagnosis on stored runs (2026-11-26)

CPU only, stored runs only (runs/MAIN, random seed 0, MMBA a0 / random a1 / MIMICRY-style a5 / DeepJanus-style a9, 108 runs each).
Scripts: scripts/analysis/diagnose_edits.py, scripts/analysis/codebook_knn_check.py, scripts/analysis/dead_code_demo.py.
Outputs (untracked): results/diag_edits/*.csv|json. Figure: docs/figs/dead_code_demo.png.

## A. Unrealistic image edits (green/olive blurry patches)

**Root cause: dead codewords in the VQGAN codebook plus cosine KNN.** This is a property of the shared operator, so it affects every method.
- vq-f8-16384: only **880 of 16,384 codewords are live** (norm > 1e-3; median live norm 1.95). The other 15,504 have collapsed to norm ~7e-5.
- The region operator (region_block 4 = 32x32 px, 64 regions; level k -> the 2^(k-1)-th nearest codeword, k = 1..14, rank 1..8192) ranks neighbours by **cosine** distance over **all** codewords (mmb/manipulator/image/selection.py build_codebook_knn). Cosine ignores the norm, so the "nearest" neighbours of a live code are mostly dead ones. For the live codes, the share of dead codewords at rank 1 / 8 / 128 / 512 / 8192 is **71% / 76% / 85% / 90% / 96%**.
- A region edit rewrites all 16 codes of the region, so practically every image edit in the study (any method, any level) is mostly near-zero latents. The decoder renders these as the smooth olive/grey-green patch seen in Fig. 1.
- Demo (docs/figs/dead_code_demo.png): the Fig. 1 iguana run (InternVL, MMBA, 7 regions, levels 2-14). The re-render with the operator as run matches the stored min_cross.png (mean abs diff 0.4/255); **91% of its 112 replacement codes are dead**. The same genotype with L2-KNN over live codes only has no olive blur, but its high levels (rank > 879) become garish, so a fix also needs a ladder sized to the live codes.
- Measured per edited region against the codec reconstruction (all methods): the median colour shift of the region mean is dE76 = 15 even at level 1 and 21-22 at levels >= 9. Sharpness (variance of the Laplacian of L*, edited / reconstruction) has a median of 0.22 at level 1 and falls to 0.05 at level 14. Blur is present at every level and worsens with level.
- By method (edited regions of the confirmed minimal crossings):

| method | regions | mean level | share level >= 10 | median dE | share dE > 15 | median sharpness ratio |
|---|---|---|---|---|---|---|
| MMBA | 1202 | 8.0 | 42% | 18.2 | 65% | 0.11 |
| Random | 1744 | 7.8 | 39% | 20.0 | 75% | 0.12 |
| MIMICRY-style | 688 | 3.9 | 11% | 11.9 | 35% | 0.31 |
| DeepJanus-style | 2197 | 6.2 | 28% | 18.4 | 65% | 0.15 |

  My hue-band "olive" flag is not informative (about 0.43 in every arm and at every level), so it is not used.
- **Why MMBA's levels are high:** Phase-1 mutation and the Phase-2 "move" step draw the level uniformly in 1..14, and Phase 2 only removes or moves edits and never lowers a level. Minimising the edit count therefore keeps a few strong (far-rank) edits. The median of the maximum level in MMBA's minimal crossings is 14.
- **Selection-time fix on existing runs? No.** Every recorded crossing uses the same operator, so none is free of dead codes. Restricting to crossings with all levels <= 8 (rank <= 128, still ~80% dead codes) leaves MMBA 19%, random 11%, MIMICRY-style 66% and DeepJanus-style 12% of runs with any such crossing (batched, unconfirmed trace scores). It would also need GPU re-confirmation.
- **Recommendation A:** the realistic fix is an operator change: KNN over live codewords only, by L2, with a ladder over the 879 live neighbours (for example ranks 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128). It is shared by all methods, so it requires re-running every method (see D). If that is not possible, the paper must describe the image edit as it is: it replaces a region's codes by codewords that decode to a smooth, low-detail patch (closer to local masking than to a "barely visible to replaced content" ladder). The current Section 3 wording does not match the produced inputs.

## B. Text vs image balance

- Genotype: 64 image genes (14 levels) and 8 text genes (synonym 1, character noise 6, saliency/typo 1; 3-4 levels).
- Generation 0 (shared sampler, all methods): on average 9.1 active image genes and 4.05 active text genes (text genes are drawn dense-uniform, image genes sparse by tier).
- MMBA mutation: "add an edit" picks a uniformly random inactive gene, so ~89% of added edits are image edits; the level is uniform in 1..14.
- Phase 2 minimises the **count** of edits and treats a near-invisible character-noise edit and a level-14 region edit as equal. Result: MMBA's minimal crossings are **94% image edits** (median 10 image, 0 text; 51% image-only, 5% text-only). Random: 83% image (median 18 / 4); MIMICRY-style: 88% (6 / 1); DeepJanus-style: 85% (22 / 4).
- Text alone rarely suffices (text-only ablation crosses 7%), so image edits are needed. But the balance is plausibly off in the sense that matters: visibility. Count-minimality keeps strong image edits and drops invisible text edits.
- **Recommendation B:** make MMBA's Phase 2 minimise a visibility cost instead of the count. Per-edit cost: image by level (or measured patch LPIPS), text character noise ~0, synonym/typo moderate. Add a "weaken" move (lower a level by one) beside reset/move. Tie-break by cost, not by margin. This is MMBA-internal (fair under D).

## C. Which metric fits the objective (boundary pair on as many seed inputs as possible, with the least visible change)

Proposal: make **SCR per seed input** the headline effectiveness metric. SCR = stealth S of the confirmed minimal crossing if the run crossed, else 0; mean over seed inputs; paired Wilcoxon vs MMBA on matched seed inputs. The paper already defines it (CR x S).
It replaces reading CR and LPIPS/edit medians separately. Those medians are computed only over crossed runs (survivor-conditioned) and say nothing about failures.
The **edit count** is not a visibility measure (one level-14 region is more visible than three character-noise edits), so it should stay a secondary "size" column, not a quality claim.
Values (seed 0, 108 matched seed inputs):
- MMBA 81.1, random 67.9, MIMICRY-style 76.6, DeepJanus-style 77.5.
- Paired: vs random -13.2 (MMBA higher on 77, lower on 28, p < 0.001); vs DeepJanus-style -3.6 (70 vs 34, p < 0.001); vs MIMICRY-style -4.5 (39 vs 60, p = 0.07, n.s.).

So SCR shows a clear, honest advantage over random and DeepJanus-style and none over MIMICRY-style. For comparison, per seed input on edit count MMBA is smaller than MIMICRY-style on 37, equal on 13 and larger on 55; on LPIPS it is better on 28 and worse on 71.
**Recommendation C:** headline = CR and SCR (both per seed input, paired), keep LPIPS and d_edit as the visibility columns, demote the edit count to the cost table. Do not add further views.

## D. Re-running only MMBA (a0)

- **Fair (MMBA-internal, baselines unchanged):** Phase-2 objective and moves (visibility cost, weaken move; B); MMBA's own mutation level prior (for example geometric toward low levels instead of uniform); MMBA's guidance weights. Generation 0 must stay the shared sampler, otherwise "random uses the same initial sampler" no longer holds.
- **Not fair with only a0 re-run:** any operator change (live-code KNN, L2 distance, ladder, region size), codec, sampler or text operators. These change the search space of all methods, so the baselines and ablations would compare against a different input space.
- **Cost:** a0 = 6 configs per seed (3 models x 2 tasks, 18 runs each); 2 seeds = 12 configs = 216 runs, ~0.84 GPU-h per config => ~10 GPU-h, plus single-input confirmation (~1.5 h) and agreement/transfer replays for a0 if they are kept (~2 h).
- An operator fix with a full re-run (a0, a1, a5, a9 + ablations a6, a7, x1, x2 = 48 configs per seed, 96 for 2 seeds) is ~80 GPU-h plus replays (~3.5 days on the one GPU).
- **Important:** an MMBA-only fix reduces the number and strength of MMBA's image edits, but **cannot remove the olive patches**. 71% of even level-1 swaps hit dead codes, because the operator is unchanged.

**Recommendation D:** the artifact the user sees is an operator defect. Only an operator fix plus a re-run of every method removes it. Under the "re-run only MMBA" constraint, implement B (visibility-cost Phase 2 + weaken move + low-level mutation prior), re-run a0 (12 configs, ~10 GPU-h + replays), and state the operator limitation honestly in the paper. Decision for the user: operator fix + full re-run (~80 GPU-h) versus MMBA-only re-run with the limitation stated.
