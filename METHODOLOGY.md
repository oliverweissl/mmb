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
| D3 | **Anchor-sign rule.** The anchor sign is the sign of `g` on the unmodified seed: the first evaluation whose genotype is all zeros. If the trace has none, `stats["anchor_g"]` is used (the code owner records it from an explicit unmodified-seed evaluation, decision of 2026-11-20). Only if both are missing is the anchor assumed to be class_a (sign +1); the column `anchor_source` records `zero_genotype`, `stats` or `assumed_class_a`. | The 5% zero-anchor initialisation can leave the unmodified seed out of a trace, and a wrong anchor sign silently inverts "crossed". The explicit `anchor_g` removes the dependence on the sampler. A run with `assumed_class_a` should be rare and is reported. |
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
| unequal budgets | equal proposals, caching counted (D6) | duplicates reduce fresh model calls differently per arm (`first_cross_eval_nocache` is the secondary column) |

## 9. Change log
- 2026-11-20: first draft (C2). Decisions D6 (cache hits counted), anchor_g, cell_id with photograph, run_meta tags, balanced-design rule recorded from the code owner / user.
