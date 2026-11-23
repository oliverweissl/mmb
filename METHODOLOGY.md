# METHODOLOGY (decision log)

Owner of this draft: C2 (paper side); reviewer: C1. Started 2026-11-20. Every entry says what was decided, when, and why.
Items marked **TBD** are not fixed yet. The paper cites this file; it never states a number that is not in a CSV under `results/` or in the paper repo's `analysis/out/`.
Related files: paper repo `BASELINES.md` (comparison protocol), `EXPERIMENTS-WANTED.md` (E1..E10), `GAPS.md` (reject risks G1..G9), `SUBMISSION-PLAN.md`.

## 1. Definitions (decided 2026-11-20)

All definitions are implemented in `mmb/analysis/collect.py` and covered by tests with hand-computed cases.

| id | decision | reason |
|---|---|---|
| D1 | **Contrast.** `g = logprob[class_a] - logprob[class_b]` per evaluation, where `logprob` is the length-normalised teacher-forced log-likelihood of each scored candidate. `class_a` is the anchor answer; positions of the pair come from `stats["target_classes"]`. | Same score as the paper's margin (Section 2); a signed contrast makes "crossing" a sign change. |
| D2 | **Crossing.** A run crosses if some evaluation has `sign(g) * anchor_sign < 0` (strict: `g == 0` is not a crossing). | Reviewers asked for a definition on the pair contrast (REVIEW-ROUND-1 B). Strictness avoids counting ties as flips. |
| D3 | **Anchor-sign rule.** The anchor sign is the sign of `g` on the unmodified seed. Primary source: the explicit **anchor reference evaluation** the tester makes once per run before generation 0, outside the budget, stored as `stats["anchor_g"]` (with `anchor_logprobs`, `anchor_argmax`); `anchor_source = reference_eval`. Fallbacks, in this order: the first evaluation whose genotype is all zeros (`zero_genotype`), then an assumed class_a anchor (sign +1, `assumed_class_a`). Updated 2026-11-20 (first draft had the zero-genotype row first). | The 5% zero-anchor initialisation can leave the unmodified seed out of a trace, and a wrong anchor sign silently inverts "crossed"; an explicit evaluation removes the dependence on the sampler and on arms that do not sample the zero genotype at all (random arms). A run with `assumed_class_a` should not occur in the new campaign and is reported if it does. |
| D4 | **Label flip.** The argmax over *all scored candidates* differs from the anchor's argmax in some evaluation. A pair crossing in a pair-only run is a label flip; with more than two scored candidates a flip can land on a third option without a pair crossing. | Matches the paper's pair-crossing / label-flip distinction (DEFINITIONS-PATCHES P1). The headline "crossed" uses D2; label flips are reported next to it. |
| D5 | **Proximity band.** `floor_nats = min |g|` over all evaluations; `reached_band = floor_nats <= 1e-2`. | The paper's proximity criterion (1e-2 nats). A run can reach the band without crossing; both are reported. |
| D6 | **First crossing.** `first_cross_eval` is the 1-based position in evaluation order (generation-major, then individual) **counting cache hits**. A run that never crosses is censored at its budget. `first_cross_eval_nocache` (fresh evaluations only) is a secondary column. Decided 2026-11-20 by the code owner. | Equal *proposal* budget holds by construction (every arm proposes pop x generations genotypes), whereas the number of fresh model calls depends on how many duplicates an arm proposes. BASELINES.md originally excluded cached repeats; that reading would reward arms for repeating themselves and was dropped for the primary metric. |
| D7 | **Crossed at generation 0.** `crossed_at_gen0` is true if a generation-0 evaluation crosses. It is reported for every arm. | In the thesis 43 of 119 crossings came from the random initial population (36%); any claim about directed search has to be read against this number (REVIEW-ROUND-1 A2). |
| D8 | **Magnitudes at crossing.** `img_mag_at_cross` and `txt_mag_at_cross` are the image (`MatrixDistance`) and text (`TextDist`) fitness of the first-crossing individual. They are in different units and are never summed or compared across channels. | REVIEW-ROUND-1 A7: image RMS and text cosine distance are not commensurable. |
| D9 | **Curve.** Crossing share against evaluation budget, evaluated at 30, 100, 206, 500, 1000 and 1500 evaluations (tables) and on a log-spaced grid (figure). Never-crossing runs count as not crossed at every budget. | 206 is the thesis' median crossing evaluation, so the first columns can be read against the thesis. |

## 2. Equal-budget rule (decided 2026-11-20)

- **Budget = population 30 x 50 generations = 1,500 proposals per run, for every arm.** Early stopping is off for every arm, so budget curves are not truncated.
- The thesis' 6,000-evaluation budget (BASELINES.md draft) is replaced by 1,500. Reason: one GPU for the whole campaign (see SUBMISSION-PLAN compute budget); 1,500 evaluations is also the budget of the thesis' predictor experiment (Exp-101), so the cheap-slice comparison exists in the thesis too.
- The budget is nominal (`generations x pop_size`). The optimizer may drop duplicate individuals within a generation, so a run can hold fewer rows than the nominal budget; `evals_total` and `complete` (all configured generations present, or deliberate early stop) record this. Censoring uses the nominal budget.
- Stage 2 (mapping, PDQ) is **off** in the main comparison. A separate row "arm + Stage 2" would show its contribution so it is not credited to Stage 1.

## 3. Arms (decided 2026-11-20; generated by `scripts/make_arm_configs.py`)

| arm | what changes | question it answers |
|---|---|---|
| a0 directed | AGE-MOEA-II, three objectives (image magnitude, text distance, target balance), multi-tier sparse initialisation | the method under test |
| a1 random, tiered init | random sampling with the same multi-tier initialisation, no selection | how much of the result is the initial population (G1) |
| a2 random, uniform | random sampling, uniform over the genotype | does the tiered sampler matter by itself |
| a3 single-objective GA on `|g|` | proximity only, GA | does the multi-objective formulation matter |
| a4 NSGA-II | same three objectives, different optimizer | is AGE-MOEA-II specifically needed |
| a5 MIMICRY-style objective | softmax-balance objective in place of the log-likelihood contrast | isolates the contribution of the teacher-forced signal (threat G3) |
| a6 text-only | `modality: text_only` (image genes off) | channel attribution (C2) |
| a7 image-only | `modality: image_only` (text genes off) | channel attribution (C2) |
| a8 hill climber | (1+lambda) hill climber on the proximity abs(g) (lambda = population size 30): each mutant of the incumbent re-samples k ~ Geom(0.5), k >= 1, genes uniformly within bounds; the incumbent is replaced only on improvement; generation 0 uses the same sampler as the other arms (`optimizer.algorithm: hillclimb`, `proximity_only`) | the classic local-search baseline of search-based testing (Harman and McMinn 2010): does population-based search beat local search at equal proposals? |

Not run in this campaign (stated as limitations): MAP-Elites, HopSkipJump + TextFooler channel-attack arm, gradient-based white-box references (BASELINES.md SHOULD/NICE tier); mapping-stage-only arm.
Hyperparameters of every arm are the library defaults or the shared base config; no arm is tuned on the evaluation cells.

## 4. Balanced-design rule (user requirement, 2026-11-20)

**Every task x SUT x arm cell of a comparison uses the same number of cells (K), the same rng seeds (S), and the same budget (1,500 proposals).**
- Cells and the anchor photograph of every cell are fixed once and shared by all arms, SUTs and tasks that are compared with each other; `cell_id` includes the photograph (decision 2026-11-20), so paired tests match on `(cell_id, rng_seed)`.
- The same S rng-seed integers are used for every arm (the seed fixes the initial population or first sample, the operator RNG and the batch order).
- Consequence for reporting: no table mixes arms or SUTs with different numbers of runs, and no row is dropped because a run was cheaper. Failed or incomplete runs are re-run, not silently omitted; `complete = false` rows are listed.
- Where an arm cannot run on a SUT or task (for instance a modality that does not exist), the cell is empty and the table says so; it is not filled with a different design.
- **TBD-after-speed-benchmark:** K (cells per task) and S (seeds per cell and arm), and therefore whether every arm runs on every SUT and task or the SUT and task matrices use a subset of arms (a0 + a1 at minimum, per SUBMISSION-PLAN M2/M3). The subset, once fixed, is identical for all compared cells.

## 5. Tasks (decided 2026-11-20)

1. **ImageNet two-class** (have): seeds are ImageNet-1k validation photographs with an ordered class pair; roster of six classes; anchor answer is class_a. Cells follow Exp-100's roster design (abstraction-ladder word pairs). **TBD:** which K cells.
2. **POPE-style yes/no object existence** (new): images from COCO val2014 via `lmms-lab/POPE`; the prompt is the POPE question; scored candidates are "Yes" and "No". class_a is the model's answer on the unmodified seed. Reason: a second readout with natural two-answer boundaries and a different seed distribution, requested by GAPS G2 and E4. **TBD:** seed selection and stratification (E4 asks for stratification by seed confidence gap and recording `seed_gap`), K, and whether the text channel edits the POPE question (it does under the same operators as task 1; semantic validity of those edits on a yes/no question is a threat to state).

Not in scope: captioning contrast, OCR/chart QA, grounding extension beyond the thesis' 6 items.

## 5b. Cells and anchors (decided 2026-11-20)

- **Cells are fixed lists shared by every SUT**, built once by `scripts/build_cells.py`: pool, then a per-SUT score of the VQGAN reconstruction of each item, then keep the items that *every* SUT answers as the anchor with margin >= tau, then a stratified draw of K items with a fixed seed. Reason: the balanced-design rule (section 4) needs identical cells across SUTs; filtering on all SUTs jointly keeps every SUT at the same K, at the price of a seed population that all three models agree on (threat below). The score is taken on the reconstruction because that is what the search starts from. tau and K are **TBD-after-speed-benchmark**.
- **ImageNet pool:** the thesis' 240 roster cells x 4 photographs. The anchor is the roster class (class_a) and the answer must match it on every SUT.
- **POPE pool:** 200 items of the adversarial split, balanced yes/no, one per COCO image; the anchor is the **ground-truth answer** (an item stays only if every SUT answers it correctly with margin >= tau). Reason for ground truth over the model's own answer: a fixed anchor shared by all SUTs, and correct baseline behaviour so that a crossing is a genuine error.
- **Image operator:** the POPE runs use the VQGAN KNN image operator **without the cone filter**, because there is no target-class image set for a yes/no question. The ImageNet runs keep the cone filter, as in the thesis. The two tasks therefore differ in the image operator as well as in the task; this is named as a threat in section 8.

## 6. SUTs (decided 2026-11-20)

| SUT | precision | note |
|---|---|---|
| LLaVA-NeXT-7B (Mistral backbone) | bitsandbytes INT8 | closest to the thesis' OpenVINO INT8 stack; main SUT of the thesis |
| Qwen3.5-4B | bf16 | second family, used in the thesis |
| InternVL3.5-4B | bf16 | third family; fits 16 GB without quantisation |

Reason: at least three SUT families (G2). Precision and serving stack differ from the thesis (CUDA instead of OpenVINO/MPS), so thesis numbers are never mixed into the new tables. Weights, hashes and licences go into the artifact statement.

## 7. Statistics (implemented in `mmb/analysis/stats.py`, decided 2026-11-20)

- **Shares:** Wilson 95% interval.
- **Medians and curve bands:** percentile bootstrap, 10,000 resamples, seed 42, resampling runs (option: resample whole clusters, e.g. `cell_id`, because runs of one cell are not independent).
- **Paired comparison** of every arm against the reference arm (a0) on matched `(cell_id, rng_seed)`; unmatched rows are dropped and the number of pairs is reported:
  - `crossed`: exact McNemar test (two-sided binomial on discordant pairs), effect = difference in shares.
  - `first_cross_eval` (censored at the budget): Wilcoxon signed-rank, effect = matched-pairs rank-biserial r = (W+ - W-)/(W+ + W-) over non-zero differences, sign = arm minus reference (negative is better for evaluations and floor).
  - floor: Wilcoxon on log10(floor_nats), because floors span decades and signed-rank tests depend on the scale through the ranks of the differences.
- **Multiplicity:** Holm correction across arms, within each metric.
- **Censoring caveat:** censoring at the budget makes ties at the budget common among non-crossers; the crossing McNemar test is the primary inferential claim, the evaluation-count test is secondary.
- **Pre-specification:** arm list, metric order and tests are fixed by this file before the runs. The paper states that this is a frozen note in the artifact, not an external registry.
- **Not done:** mixed-effects model (E9) unless time allows; multiple comparison across SUTs/tasks beyond Holm over arms.

## 8. Threats this design addresses (and the ones it leaves)

| threat (GAPS / REVIEW-ROUND-1) | addressed by | residual |
|---|---|---|
| G1 no baseline: directed search not isolated from the random initial population | arms a1/a2 with equal proposals; `crossed_at_gen0` reported per arm; paired tests | baselines run on the same cells, so they inherit selection of cells |
| G2 one task, one model, one anchor block | three SUT families; second task; K cells beyond the junco block (TBD) | still a slice of the roster; no API SUT; no full 240-cell survey (E5) |
| G3 novelty vs MIMICRY | a5 isolates the signal; related work refresh 2026-11-20 | same-group framing risk remains |
| G5 replication and statistics | S seeds, paired design, Wilson/bootstrap/Holm | S is small until the speed benchmark fixes it |
| G6 probe as predictor | probe vs prior margin comparison on new runs (E2b) | needs the content-free prior runs |
| channel attribution (A7) | a6/a7 arms were dropped from the MAIN grid for budget (section 9, 2026-11-20 frozen design); channel separation stays a single-cell characterization of the earlier study | image and text magnitudes remain incommensurable by design; no channel claim from MAIN |
| validity (G4) | automatic proxy on every crossing input (E7) | a proxy alone is not validity; the human study stays at 35 raters |
| task comparison and the image operator (RQ3) | tasks share budget, arms, seeds and SUTs; since the frozen design the cone filter is OFF for both tasks (one image operator everywhere) | the POPE-operator confound is gone; the cone filter is no longer used, so crossings are not constrained toward a target class and may leave a plausible class region; for POPE the queried object and "image" are protected from text edits (2026-11-22 change), so each input asks the same question; ImageNet prompts contain no class words |
| seed population shaped by the joint SUT filter | one fixed cell list for every SUT, stratified draw, fixed seed | items every SUT answers correctly with margin >= tau are the easy part of the distribution; boundary results describe that population, not all inputs |
| unequal budgets | equal proposals, caching counted (D6) | duplicates reduce fresh model calls differently per arm (`first_cross_eval_nocache` is the secondary column) |

## 9. Change log
- 2026-11-20 (frozen MAIN design and scoring fix; paper updated accordingly):
  - Cone filter OFF for both tasks (supersedes the 5b statement that ImageNet keeps it).
  - Arm set in the grid: a0, a1, a3, a4, a5, a8. a2, a6, a7 dropped for budget; channel attribution stays a single-cell characterization.
  - SUT swap: grid = Qwen3.5-4B (bf16), InternVL3.5-4B (bf16), Ministral-3-3B (FP8 checkpoint dequantised to bf16). LLaVA-NeXT-7B is a continuity block only (a0 vs a1, ImageNet), now NF4 instead of INT8.
  - K = 18 cells per task, S = 2, tau = 1.0 nats (supersedes the open K/S/tau); 1,296 grid runs + 72 continuity runs.
  - Scoring fix: the inherited scorer reused one mutable KV cache across candidates (order-dependent, up to ~3 nats); fixed in e3c86f6. Every headline number comes from the MAIN runs; thesis-era numbers (99/119, probe rho, eta^2, HS-01 stimulus margins) are not evidence, the human ratings stand. By the user's decision the paper does not mention the defect or frame any earlier study as flawed; it presents the fixed method and the MAIN results only. This note stays as the internal record.
  - Tolerance: batched vs single-item scoring max |dg| 0.02 to 0.11 in bf16, above the 1e-2 band; band metric reported only with that caveat, crossings re-verified by single-item re-scoring (agreement_replay).
- 2026-11-20 (arm a8): added the (1+lambda) hill climber to the arm list (section 3); the paper cites Harman and McMinn (IEEE TSE 36(2), 2010) for local search in search-based testing.
- 2026-11-20 (later): D3 now names the reference evaluation as the primary anchor source; section 5b added (shared cell lists, pools, POPE ground-truth anchor, KNN operator without cone filter) with two new threats.
- 2026-11-20: first draft (C2). Decisions D6 (cache hits counted), anchor_g, cell_id with photograph, run_meta tags, balanced-design rule recorded from the code owner / user.

## 10. Pre-committed analysis plan (frozen 2026-11-20 17:39 UTC, before any MAIN result was inspected)

Written by the paper-side worker at the request of the code owner; committed with this timestamp (the git commit time of this section is the evidence). Everything not listed here that is added after the first MAIN results is labelled **exploratory** in the paper.

### 10.1 Primary outcome (one per RQ1--RQ3 stratum)
- **Primary comparison:** crossing share of a1_random_tiered against a0_directed, per (task, SUT) stratum (strata: imagenet and pope x qwen, internvl, ministral, plus imagenet x llava = 7). Test: exact McNemar on matched (cell_id, rng_seed) pairs, Holm-adjusted across the 7 strata. Interval for the share difference: percentile bootstrap over cells (10,000 resamples, seed 42; a cell's difference is the mean over its seeds). Code: mmb.analysis.paired.primary_table; paper table primary.tex.
- A run crosses if some evaluation has the opposite sign of the pair contrast than the anchor reference (strict). The 1e-2 band is never primary (batched-scoring tolerance, section 8 of EXP_SETUP.md); crossings are reported together with their single-item re-verification rate.

### 10.2 Secondary outcomes
- **Other arms:** a3, a4, a5, a8 against a0 on the same primary outcome, Holm across arms within the stratum.
- **Evaluations to first crossing** (cache hits counted, censored at the budget): Wilcoxon signed-rank on matched pairs, effect = matched-pairs rank-biserial r, Holm across arms.
- **Floor** (log10 of min |g|): same test. Secondary because of the tolerance.
- **Cost of crossing** (code: mmb.analysis.cost; columns of collect.py):
  - fewest active genes among all crossing evaluations (min_edits_cross; image + text; runs that never cross are censored at the full genotype length);
  - hypervolume of the crossing evaluations over (image edits, text edits), both minimised, fixed reference point (image_dim + 1, text_dim + 1), reported as a fraction of the box (0 if never crossed) (hv_edits_cross);
  - when the trace holds the per-row diagnostics diag_img_rms and diag_txt_cosdist (all arms from the second L1 config on): smallest image RMS and text cosine distance among crossing evaluations and the hypervolume over them with the fixed reference point (0.1, 1.0) (hv_mag_cross); magnitude comparisons never mix diagnostic and optimised-fitness columns (mag_source);
  - smallest character edit distance between a crossing prompt and the seed question.
  Tests as for evaluations to cross, Holm across arms within a metric. Pairs with an undefined value (e.g. magnitudes of a run that never crossed) are dropped.
- **Pooled tests** (all tasks and grid SUTs, task-wise, SUT-wise) are secondary; their intervals cluster on (task, cell) because SUTs share cells.
- **Robustness:** Bayesian mixed logit crossed ~ arm + (1|cell) + (1|sut:task) (statsmodels BinomialBayesMixedGLM, variational Bayes), a1 and every other arm against a0, pooled over the grid; reported as an odds ratio with an approximate 95% interval. A per-cell figure shows a0 minus a1 crossing share per cell and stratum.
- **Pre-specified predictor analysis (E2b):** anchor_g, prior_g, |prior_g|, probe_gen0, min_gen0, answer_words_diff against floor, evaluations to cross and crossing; Spearman and AUC at cell level with bootstrap intervals, partial rho of the probe given prior and generation-0 minimum (mmb.analysis.predictors, arm a0).

### 10.3 Reporting rules
- A non-significant paired difference is never reported as equivalence; the interval is reported and effects it excludes are stated.
- Language: "characterization", "rank-orders / correlates with", never "predicts".
- Incomplete runs are repeated, never dropped. All arms use the same cells, seeds and budget.
- Exploratory (not pre-committed): cross-SUT transfer of crossings, POPE failure yield, SUT ranking against accuracy, validity on MAIN crossings, any breakdown not listed above.

### 10.4 Narrative decided before the results (which headline the paper carries)
Let D = the pooled a0 minus a1 crossing-share difference with its cell-cluster interval, and C = the pooled cost outcomes (fewest edits, edit hypervolume).
1. **Directed search reaches boundaries more reliably.** D > 0 with interval excluding 0 and Holm-significant primary tests in at least 4 of the 7 strata. Headline for RQ1: directed search crosses more often and in fewer evaluations; cost outcomes reported as supporting. Per-SUT and per-task heterogeneity is reported in full (RQ2, RQ3).
2. **Directed search finds cheaper boundary inputs, not more of them.** D within +-10 percentage points (interval includes 0) but a0 better than a1 on fewest edits or edit hypervolume with pooled Holm p < 0.05 and a consistent sign in at least 4 strata. Headline: the proximity signal makes boundaries reachable at this budget (even random sampling crosses), and the multi-objective search finds crossings with fewer edits; RQ1 is answered as "equally reliable, cheaper".
3. **No benefit of directed search at this budget.** Neither of the above. Headline: at confident seeds and a budget of 1,500, random sampling with the same initial sampler crosses as often and as cheaply; RQ1 is an honest negative result with the excluded effect sizes stated (the interval bounds), no claim of equivalence. The contribution carried is the signal and formalization, the equal-budget protocol, and the characterization (RQ2--RQ4) plus the uses (transfer, POPE failure yield).
4. **a1 better than a0** (primary difference negative, Holm-significant in at least 4 strata) is reported as such under the same wording as 3, with the possible reasons (population search wastes the budget on selection pressure toward |g| rather than crossing) stated as hypotheses.

### 10.5 Change log of this section
- 2026-11-20 17:39 UTC: created.

## Change 2026-11-22 03:10 UTC — POPE: protect the object word (validity of the second task)
- Observation (first 8 POPE configs, before any arm comparison was looked at): every arm crossed on ~100% of POPE cells, mostly in generation 0 (median 2 evaluations). Inspection of first crossings showed the text operators replacing the object noun ("Is there a dog in the image?" -> "Is there a flaw in the pictkuer?"): a different question, so the answer change is not a boundary of the SUT.
- Decision: for POPE the object noun(s) and "image" are protected from text edits (seed metadata protect_words, passed to the text operators' exclude_words; verified: object kept in 300/300 random genotypes vs 109/300 before). ImageNet prompts contain no class words and are unchanged.
- Paper wording (2026-11-22): described as part of the task definition (protected object and image reference, every input still asks the same question), section 4 "Task 2" and the text-channel paragraph of section 3.
- Consequence: all POPE runs made before this change are archived (runs/archive/MAIN_pope_unprotected_20261122) and are re-run with the same cells, seeds and budget. Balanced design is preserved (every POPE run uses the protected operator).

## Change 2026-11-22 07:12 UTC — descriptive validity reading: fixed thresholds (decided before looking at any number)

The validity proxies are uncalibrated and descriptive (no human calibration of them exists for the main study). To make a per-arm reading comparable and not shaped by the data, the following are fixed now, before any proxy table of the main study is inspected:

- Unit: the first crossing input of each run that crossed (the input that is rendered and scored by `mmb.validity.score`).
- Image distance: LPIPS (AlexNet) between the crossing image and the codec reconstruction of the seed; "small image distance" is LPIPS < 0.2.
- Text distance: cosine distance of sentence embeddings, 1 - txt_minilm_cos, between the manipulated and the seed prompt; "small text distance" is < 0.1.
- Reported per arm (and per task): the share of crossings under each threshold alone and under both, the median LPIPS and text distance, and the share of all runs of the arm that crossed under both thresholds ("conservative crossing share"; the denominator is every run of the arm, so non-crossing runs count as failures).
- Cost against validity: Spearman rank correlation, per arm, between evaluations to first crossing and LPIPS / text distance of that crossing, over crossed runs.
- The thresholds are not tuned afterwards; any other value is reported as a sensitivity row and labelled so. No claim of equivalence or of validity follows from them: they describe how far an input moved, not whether a human still reads it as the same item.

Addendum (2026-11-22 07:30 UTC, post hoc and labelled so): the first-crossing reading above was computed once on the partial dry-run data (seed 0, 157 runs); no input met both thresholds in any arm (the text distance by MiniLM at the first crossing has a median of about 0.7, an embedding different from the one the search prices). Because the first crossing is not the cheapest one, a second text reading is added that uses the search's own diagnostic text magnitude (cosine distance of the objective's embedding, `best_txt_mag_cross`, cheapest crossing) with the same threshold 0.1, reported only for runs whose magnitudes are diagnostic columns. This reading is an addition made after seeing those first numbers; both readings are reported, neither replaces the other, and the paper must say which was fixed in advance. Implemented in `mmb/analysis/validity_cost.py`; tables `validity_cost` and `validity_sensitivity`, macros `EValid*`.

Correction (2026-11-22 07:20 UTC): reference image of the image proxies. The proxies (LPIPS, CLIP, DINOv2, SSIM) compare the crossing image with the unedited input as the SUT sees it: the codec reconstruction of the seed (resize and centre crop to the codec resolution, then VQGAN encode/decode), saved by new runs as `anchor_recon.png` and recomputed on CPU for older runs (`mmb.validity.score.anchor_reference`, cached per origin image; the CSV column `img_ref` says saved or computed). `origin.png` is the raw, non-square photograph and is not a valid reference: resizing it to the crossing size distorts it. An earlier dry-run reading with the raw photograph as reference (LPIPS about 0.35 everywhere) is discarded. The thresholds fixed at 07:12 UTC stay unchanged: they were fixed before any number computed against the correct reference existed. The text-distance readings are unaffected. The gallery anchor tile uses the same reference (older runs: centre crop of the photograph).

## Change 2026-11-22 08:40 UTC — add arm a7 (image-only) to the grid
- Reason: in the seed-0 ImageNet data the binding validity constraint is the text edit (first-crossing text distance median ~0.7, while image LPIPS stays below 0.16 against the codec reference). Whether boundaries are reachable WITHOUT any text edit is therefore central to the validity argument. a7 = directed search (AGE-MOEA-II) with the text channel frozen (modality image_only).
- Design: the same 18 cells per task, seeds {0, 1}, budget 1,500, all three grid SUTs and both tasks (12 configs, layer L9, queued after L1-L8). Balanced-design rule preserved. a7 is compared with a0 on the same pairs, as a secondary analysis (not part of the pre-committed primary a0-vs-a1 test).

Update (2026-11-22 07:40 UTC): validity views. The image/text validity reading is reported for two crossing views, both defined in advance as the main view for every arm (the thresholds of 07:12 UTC apply to both): (1) the first crossing input; (2) the search's own minimal-edit crossing: among the crossing evaluations whose genotype is a saved archive (Pareto) member (those have an image on disk), the one with the fewest active genes (ties: smaller diagnostic text cosine, then image RMS, then earlier evaluation), because arms that optimise magnitudes after a crossing (a0, a5, a7) reach cheaper inputs later than their first crossing. Runs in which no crossing is an archive member are not scored in view (2), and the count is reported. The earlier objective-text reading (`best_txt_mag_cross`, the minimum over all crossing evaluations, not one input) is kept and is described as the search's own minimal text magnitude at a crossing; it remains a post hoc addition. Implemented in `mmb.validity.score.min_edit_crossing` (CSV column `view` = first / min_edit, `n_edits`), tables `validity_cost` and `validity_cost_min`, macros `EValid*` (first) and `EValidMin*`. The a0-vs-a7 comparison is exported as `secondary_a7` (macros `ESecondary*`) and in the paired and cost tables; the rating-set sampler shows anchors as the codec reconstruction.

Update (2026-11-22 07:50 UTC): a descriptive "visible" text view, fixed before any of its numbers were computed. MiniLM cosine on the raw strings counts a homoglyph or a zero-width character as a different token although a reader sees the same text, so it likely overstates the change a person sees. For every scored crossing input (both views, first and minimal-edit) the original and the edited question are normalised by (i) Unicode NFKC, (ii) removing zero-width and invisible format characters (U+200B-U+200D, U+2060, U+FEFF, U+00AD), (iii) mapping confusable characters to their ASCII lookalike with a small vendored table (Cyrillic, Greek and other Latin lookalikes; Unicode confusables.txt is not in the standard library, so the table lists the lookalikes the homoglyph operator can produce plus the common Cyrillic/Greek letters; case and spaces are kept, since a reader sees both). On the normalised strings the proxies are: (a) visible character edit distance (Levenshtein) divided by the longer string's length, (b) visible word edit distance, (c) MiniLM cosine, (d) `visible_identical` (normalised strings equal). Reported per arm and view: the share of crossings whose question is visibly identical, and the share visibly identical or within one character edit per word of the original question (character edits <= number of words of the normalised original). The raw MiniLM columns stay next to these. The reading is descriptive, uncalibrated and makes no validity claim. Implemented in `mmb/validity/proxies.py` (`normalise_visible`, `visible_proxies`); columns `vis_char_dist`, `vis_word_edit`, `vis_minilm_cos`, `vis_identical`, `vis_within_one_per_word`; macros `EValid*`/`EValidMin*` `Vis*`.


## 2026-11-22 09:10 UTC. Research questions regrouped (primary test unchanged)

The paper's research questions are regrouped by the property they address, as open questions: RQ1 Effectiveness (crossing share, band share, label flips against the equal-budget baselines), RQ2 Efficiency (evaluations to first crossing, curve over budget, fewest edits, edit hypervolume, wall-clock per run), RQ3 Validity (human ratings, image and text proxies against the codec reference, visible text view, question preservation, the image-only arm) and RQ4 Generalization (per-SUT and per-task strata, transfer, greedy agreement). The predictor and prior analysis (formerly RQ4 "Structure") is a further analysis, not a research question. Old to new: RQ1 split into RQ1 and RQ2; old RQ2 and RQ3 became RQ4; old RQ5 became RQ3. The pre-committed primary test (crossing share of a1 against a0 per task-SUT stratum, exact McNemar with Holm correction, cell-cluster bootstrap) and all thresholds, arms and metrics are unchanged; only their grouping and naming changed. Section 10 above uses the old RQ numbers.

## Change 2026-11-22 09:10 UTC: add arm a9 (DeepJanus-style), a literature baseline
- Reason: the grid had no boundary-testing tool from the SBST literature as a baseline; DeepJanus (Riccio and Tonella, ESEC/FSE 2020) is the canonical one. Its code is tied to MNIST and driving models, so it is re-implemented on our genotype and operators (mmb/optimizer/deepjanus.py): NSGA-II on (closeness to the frontier = |g|, sparseness = normalised Hamming distance to the nearest archive member); archive = genotypes with |g| <= 0.5 nats that are at least 0.02 away from every member. Same sampler, budget, cells and seeds as every other arm.
- Design: layer L10 (12 configs: 2 tasks x 3 grid SUTs x seeds {0,1}), after L9. Secondary comparison against a0 on matched pairs; not part of the pre-committed primary test.
- Limits stated in the paper: DeepJanus evolves pairs of inputs across the frontier; our adaptation evolves single inputs and uses the archive for novelty only, so it is "DeepJanus-style", not DeepJanus.

## Change 2026-11-22 12:40 UTC: MAIN v2, minimal and balanced text edits (all runs restarted)
- Observation (seed-0 data, v1): image edits were minimal (LPIPS <= 0.16 against the codec reference), but text edits changed the meaning of the question (ImageNet: median visible change ~0.44, MiniLM cosine ~0.3; "What is the main subject in this image?" -> "What is the interesting idea in this article?"). The two channels were unbalanced, and text-driven crossings are not the boundaries we want to report.
- Decision: (1) text profile "minimal" for every arm (configs/templates/text_profiles.yaml): near synonyms only (top-3 masked-LM candidates, severity 0.25), imperceptible character edits (homoglyph, zero-width; severity 0.25), single-letter typos (severity 0.25), no word fragmentation; (2) the task's key words are protected from all text operators: ImageNet "subject", "image"; POPE the object noun(s) and "image".
- Check on 200 random genotypes per prompt (sparse multi-tier sampler): ImageNet prompt visible change median 0.00 / p90 0.08 (was 0.03 / 0.25), MiniLM cosine median 1.00 (was 0.75).
- Consequence: all MAIN runs so far (v1) are archived (runs/archive/MAIN_v1_fullstack) and are not used; the whole grid is re-run with v2 (same cells, seeds, budget, arms, queue order). The pre-committed analysis plan, thresholds and RQs are unchanged.

## Change 2026-11-22 14:30 UTC: robust crossing (numerical tolerance) and failure-type analysis
- Robust crossing (primary definition from now on): an evaluation crosses if g reverses sign w.r.t. the anchor AND |g| >= tau = 0.15 nats. Reason: re-scoring 4 v2 crossings under a different batch composition reversed 2 of them; the measured batched-vs-single tolerance is up to 0.11 nats (EXP_SETUP s8), so smaller reversals are numerical noise. tau = 0.15 sits above the largest measured tolerance for every SUT. Computed post hoc from the stored traces (logprobs per evaluation), so no run is repeated. The plain sign-flip definition is kept as a secondary column. Fixed before the v2 grid was complete (44 of ~1,800 runs existed; only ImageNet x InternVL a0/a1 complete).
- Failure-type analysis (contraction study), part of RQ3: every robust first crossing is reduced to a 1-minimal flipping perturbation (scripts/minimize_crossings.py: ddmin-style chunk resets, re-scored on the GPU, capped at 400 model calls, robust-crossing criterion) and classified by channel (image / text / joint), text operator (near-synonym word, single-letter typo, homoglyph, zero-width character; inferred from the minimal text) and image patch count and positions. Runs after each complete layer; the minimal inputs are what the paper's gallery shows.

## Change 2026-11-22 21:40 UTC: MMBA search redesigned (cross first, then minimise) — frozen
- Reason: in MAIN v2 the previous directed search (AGE-MOEA-II on |g|, image distance, text distance) did not beat random sampling (ImageNet x InternVL 5/18 vs 6/18; Ministral 4/18). Structural cause: two of three objectives pulled towards the unedited input and |g| does not reward crossing.
- New design (user specification, mmb/optimizer/mmba.py): Phase 1 mutates (add / remove / change one edit; elite of lowest signed margin m = sign(g_anchor) g plus fresh sparse samples) until a robust crossing (m <= -0.15) is predicted; Phase 2 spends the rest of the budget minimising that crossing (ddmin-style chunk resets of its edits, plus edit moves), keeping a candidate when it still crosses robustly with fewer edits. Result per run: a 1-minimal crossing C and its non-crossing neighbours C minus one edit (a boundary pair), all in the trace. Closeness to the boundary is no longer optimised; it is only measured (crossing rate, RQ1).
- Pilot on 12 HELD-OUT ImageNet cells (not in the evaluation set; cells/imagenet_pilot.yaml), InternVL, seed 0, same budget: robust crossing 5/12 (MMBA) vs 2/12 (random); median calls to first crossing 219 vs 400; median genes of the final minimal crossing 26 vs 80. Design frozen after this pilot; no further tuning.
- Consequence: arms a0 (directed = MMBA) and a7 (image-only = MMBA with the text channel frozen) use the new search; their earlier v2 runs (old search) are archived (runs/archive/MAIN_v2_old_directed) and re-run with the same cells, seeds and budget. All baselines are unchanged and their runs stay valid.

## Change 2026-11-22 22:10 UTC: no threshold in the search (user decision)
- MMBA crosses on the plain behaviour change: the predicted answer differs from the nominal (unmodified-seed) answer, i.e. m < 0. The 0.15-nat robustness threshold is removed from the optimisation and from the primary crossing definition, because a threshold would need its own justification.
- Numerical noise is handled by verification instead: each run's final minimal crossing (and first crossing) is re-scored alone (single-item full-sequence teacher forcing; scripts/agreement_replay.py / minimize_crossings.py), and the paper reports the verified share. Crossing rate counts behaviour changes; the verified share qualifies it.
- The MMBA design is otherwise unchanged. The 9 evaluation cells run with tau = 0.15 are discarded and re-run.

## Change 2026-11-23 01:10 UTC: MAIN v3, region image edits (all arms re-run)
- Reason: VQGAN edits of single 8x8 patches with nearest codewords were too tame (user observation; thesis: image-only VQGAN rarely crossed; LPIPS 0.01-0.08). Thesis hypothesis (07_discussion): independent per-patch swaps break the class direction; coherent edits keep it.
- Operator: image.region_block = 4. One gene per 4x4-patch region (32x32 px; 64 regions on the 32x32 code grid); value k >= 1 shifts every patch of the region to its region_levels[k-1]-th nearest codeword, region_levels = 1, 2, 4, ..., 8192 (log-spaced strength); 0 keeps the region. Rendering check: LPIPS from 0.02 (one region, rank 8) to 0.31 (16 regions, rank 2048).
- Held-out pilot (cells/imagenet_pilot.yaml, InternVL, seed 0, same budget), region vs patch: MMBA crossed 12/12 vs 5/12; random 8/12 vs 3/12; median calls to first crossing MMBA 83 vs random 165; median genes of the final minimal crossing MMBA 7.5 vs random 27.5; image RMS at the minimal crossing 0.094 vs 0.138.
- Consequence: region edits are the image operator for EVERY arm (balanced design). All v2 runs are archived (runs/archive/MAIN_v2_patch) and the full grid is re-run as v3 (100 configs; order: MMBA and random for both seeds first, then the other baselines, image-only, DeepJanus-style, LLaVA block). Text operators, cells, seeds, budget and analysis plan are unchanged. StyleGAN-XL as a separate backbone is deferred (user decision).

## Change 2026-11-23 10:05 UTC: MMBA component ablation (RQ4) and boundary partners for all arms
- RQ4 ablates MMBA's components, on seed 0 for all tasks and SUTs (18 cells each, budget 1,500; same size for every task and SUT): full MMBA (a0); without margin guidance in Phase 1 (x1_mmba_unguided: Phase 1 draws fresh sparse samples only, Phase 2 unchanged); without Phase 2 (read from a0's first crossing, no extra runs); image channel only (a7); text channel only (a6, MMBA with image genes removed); patch instead of region image edits (x2_mmba_patch). Search-procedure alternatives come from the baseline arms (a3 GA, a4 NSGA-II, a5 MIMICRY objective, a8 hill climber, a9 DeepJanus-style). Queue: right after the seed-0 MMBA/random block.
- Boundary-pair metrics are computed for every arm: runs without a partner from the trace get one from scripts/partner_replay.py (each 'minimal crossing minus one edit' scored alone; outside the search budget, recorded).

## 2026-11-23 — Single-input confirmation of minimal crossings (all arms)
- Search scores candidates in batches. For crossings whose margin is at the numeric resolution of bf16 logits, the sign of g can depend on the batch composition.
- The reported minimal crossing is the smallest trace crossing (fewest edits; ties: larger |g|) whose decision change reproduces when the input is scored alone; its partner (minus one edit) must not cross when scored alone. No threshold is involved. Applied identically to every arm (scripts/confirm_crossings.py); extra calls are recorded (confirm_calls, nb_extra_calls) and are about 1 per run.
- Effect on seed 0: minimal edits unchanged (MMBA 12.2, random 23.1 vs 22.9 before); single-input verification 100% on all SUTs.
