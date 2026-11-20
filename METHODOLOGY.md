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
| channel attribution (A7) | a6/a7 arms, magnitudes reported per channel in their own units | image and text magnitudes remain incommensurable by design |
| validity (G4) | automatic proxy on every crossing input (E7) | a proxy alone is not validity; the human study stays at 35 raters |
| task comparison confounded by the image operator (RQ3) | tasks share budget, arms, seeds and SUTs | POPE uses the KNN operator without the cone filter, ImageNet with it, so a difference between tasks mixes task and operator; the cone ablation exists only for ImageNet (thesis Exp-27) |
| seed population shaped by the joint SUT filter | one fixed cell list for every SUT, stratified draw, fixed seed | items every SUT answers correctly with margin >= tau are the easy part of the distribution; boundary results describe that population, not all inputs |
| unequal budgets | equal proposals, caching counted (D6) | duplicates reduce fresh model calls differently per arm (`first_cross_eval_nocache` is the secondary column) |

## 9. Change log
- 2026-11-20 (arm a8): added the (1+lambda) hill climber to the arm list (section 3); the paper cites Harman and McMinn (IEEE TSE 36(2), 2010) for local search in search-based testing.
- 2026-11-20 (later): D3 now names the reference evaluation as the primary anchor source; section 5b added (shared cell lists, pools, POPE ground-truth anchor, KNN operator without cone filter) with two new threats.
- 2026-11-20: first draft (C2). Decisions D6 (cache hits counted), anchor_g, cell_id with photograph, run_meta tags, balanced-design rule recorded from the code owner / user.
