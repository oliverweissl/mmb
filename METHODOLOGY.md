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
| task comparison and the image operator (RQ3) | tasks share budget, arms, seeds and SUTs; since the frozen design the cone filter is OFF for both tasks (one image operator everywhere) | the POPE-operator confound is gone; the cone filter is no longer used, so crossings are not constrained toward a target class and may leave a plausible class region |
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
