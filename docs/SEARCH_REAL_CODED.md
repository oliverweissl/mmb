# Real-coded search (CStud2, 2026-11-28)

Scope: the search side of the continuous genotype of METHODOLOGY_REVIEW_2026-11-28.md.
CPU only; the manipulator is coded against the C2 contract (stub in the tests).

## Genotype seen by the optimizer
`[c in [-1,1]^d_img | t in [0,1], u in [0,1)]`, all-zero = unedited seed.
`mmb/optimizer/genotype.py: GenotypeSpace` holds the bounds, `image_slice`, `text_slice`,
which genes `scale(g, lam)` multiplies (c and t; u unchanged) and which upper bounds are open (u).
`GenotypeSpace.from_manipulator(m)` reads `m.xl/m.xu` (or `m.bounds`), the two slices and `m.scale`.
Magnitudes (C2 decision 2026-11-28, w-weighted): image = w_img * RMS(c), text = w_txt * t,
genotype = L2 of the two weighted block magnitudes. w_img / w_txt are read from the manipulator
(attributes `w_img`, `w_txt`; 1.0 if missing); the optimizer never applies them to the genes.

## Shared variation (every method)
| piece | setting | where |
|---|---|---|
| initial sampler | `MagnitudeStratifiedSampling`: magnitude r_k = r_min + (1 - r_min)((k + U)/n)^2 (one draw per stratum, dense at small r), modes image-only / text-only / both cycled over the strata, own seeded RNG | `sparse_sampling.py` |
| crossover | SBX, eta = 15, prob = 0.9 per mating, prob_var = 0.5 per gene | `variation.build_crossover` |
| mutation | PM, eta = 20; per offspring a fair coin picks ONE block, then each gene of the block mutates with p = 1/len(block); `at_least_once` inside the chosen block (kept by C2 decision 2026-11-28, same for every method: an applied mutation always changes the offspring) | `variation.BlockPolynomialMutation` |

Every arm builds these through `RealPymooOptimizer` (agemoea2, nsga2, ga = MIMICRY-style with the
softmax-balance objective, random, hillclimb), `DeepJanusOptimizer` and `MMBAOptimizer`, so
generation 0 is identical across methods for a seed (test `test_generation0_shared_across_methods`).

### pymoo good practice
- Real-coded GA default in pymoo is SBX(eta = 15, prob = 0.9) + PM(eta = 20) with float random sampling:
  installed pymoo 0.6.1.6, `algorithms/moo/nsga2.py:84-87`, `algorithms/moo/age2.py:26-29`
  (also https://pymoo.org/algorithms/moo/nsga2.html, https://pymoo.org/algorithms/moo/age2.html).
- Per-variable mutation probability 1/n_var: `core/mutation.py`, `get_prob_var` = min(0.5, 1/n_var);
  we apply the same rule inside the chosen block (1/len(block)).
- No rounding repair on real genes (the integer code used `RoundingRepair`; dropped). SBX and PM clamp to
  the box themselves (`repair_clamp`, `set_to_bounds_if_outside`); we additionally keep u < 1.
- Seeding: `algorithm.setup(problem, seed=...)` fixes pymoo's `random_state` stream (0.6.1.x API).
- Sources: Deb and Agrawal 1995 (SBX), Deb and Goyal 1996 (PM), Deb et al. 2002 (NSGA-II),
  Panichella 2022 (AGE-MOEA-II), Blank and Deb 2020 (pymoo, IEEE Access 8).
- Note: PM at a bound keeps the bound with probability 1/2 (Deb's bounded PM); e.g. a t = 0 gene
  stays 0 in half of its mutations. This is standard behaviour, not a bug.

## MMBA (`mmba.py`)
Phase 1 unchanged in spirit: margin-elite + SBX/PM offspring + 20 % fresh samples, until the first
crossing (m < -tau). Phase 2 minimises magnitude:
1. global line: scale(C, lam), lam in [0, 1]; 2. block lines: scale one block by mu (mu = 0 tried first:
an unneeded block is dropped), alternating image/text until a full round gains <= eps;
3. a line stops when hi - lo <= eps; the last line's ON/OFF pair is `boundary_pair`
(`on`, `off`, `gap`, margins, line kind).
Lines use batched bisection: a generation evaluates pop_size equally spaced points in the bracket,
so the bracket shrinks by pop_size + 1 per generation (two generations reach eps = 1e-3 at pop 50).
Budget left after convergence: guided Phase-1 restart restricted to weighted magnitude < |C|
(margin elite + SBX/PM + 20 % fresh samples; anything at or above |C| is rescaled by scale(g, lam)
to [0.5, 1) * |C|); a crossing below |C| becomes the incumbent and Phase 2 runs again. The budget
stays exactly generations x pop_size.
`boundary_pair` also carries `line` (global | image | text), `mag_on/mag_off` and, after the
tester renders the pair, `kind` = `text_step` (OFF/ON texts differ: the text output stepped) or
`continuous` (same text, image changes continuously), `text_differs`.

## Tester / runner / config (integration, 2026-11-28)
- `run_boundary_test._build_tester`: every arm is `RealPymooOptimizer` (MMBA, DeepJanus subclass it),
  `optimizer.mmba.eps/tau` go to MMBA; numba warm-up on the real genotype.
- `VLMBoundaryTester`: `set_space(manipulator)` per seed (legacy integer optimizers keep
  `update_gene_bounds`), float all-zero anchor, float genotypes in first_cross / pareto / trace,
  stats.json records `genotype`, `xl`, `xu`, `image_slice`, `text_slice`, `w_img`, `w_txt`.
- `min_cross.json`: the smallest crossing by weighted magnitude evaluated WITHIN the budget
  (`within_budget`, `eval_idx`); budget-bound metrics use only this and the trace.
- `boundary_pair.json` (+ `min_cross_neighbour.json/.png`): the ON/OFF pair, decided on the CANONICAL path
  (rendered alone, single SUT call; docs/DETERMINISM_2026-11-28.md). `pair_source` = `own_phase2` (MMBA's
  Phase-2 pair, both sides confirmed) | `posthoc_bisection` (canonical bisection on scale(C, lam) to the same
  eps from the smallest confirmed in-budget crossing; all arms) | `none`. Extra calls in `nb_extra_calls`;
  refined ON image `boundary_on.png`. Fields: gap, line, lam_on/off, kind, text_differs, confirmed, per side g,
  magnitude, mag_image, mag_text, text, genotype. min_cross.json adds g_canonical / confirmed.

## Continuous image backend (`image.backend: vqgan_latent`)
`mmb/manipulator/image/latent.py`: z = z0 + w_img * m_max * ||z0|| * (B c) / sqrt(d), z0 = quantised latent,
B = d orthonormal directions (`latent_bases.py`: dct | dctmid | bump, moved from the pilot script), decoded
without re-quantisation, batched; c = 0 returns the cached codec reconstruction. Config `image.latent`
(basis dct, d 16, m_max 0.2, w_img 1.0). `VLMManipulator` exposes the contract (xl, xu, slices, scale,
w_img / w_txt forwarded, magnitude_mask, upper_open) and keeps float genes end to end.
- Config: `MutationConfig.eta = 20`, `CrossoverConfig.eta = 15`, `SamplingConfig.mode =
  stratified_magnitude` (+ r_min, r_max, power), `OptimizerConfig.mmba = MMBAConfig(eps=1e-3, tau=0)`.
  A real-coded optimizer rejects the legacy integer samplers (sparse, sparse_multitier, ...).
- Template: `configs/templates/continuous_template.yaml`. Early stop was removed (every run spends the full
  budget); the integer optimizer, integer samplers, codebook / StyleGAN backends were removed (2026-11-28 trim).

## Tests (`tests/test_real_search.py`, `tests/test_algorithms.py`)
Synthetic SUT (`mmb/optimizer/synthetic_sut.py`): m = base - k_img mean(c d) - k_txt floor(4t)/4
(smooth image part, piecewise-constant text part). Checked: boundary within eps (smooth image
boundary, closed-form lam, text step at t = 0.75 with the unneeded image block dropped), ON crosses / OFF
does not, magnitude non-increasing, exactly 30 x 50 = 1500 calls for every method, determinism per seed.

## Figure
`docs/figs/search_real/mmba_synthetic_trace.{png,pdf,json}` (`scripts/analysis/search_real_fig.py`):
A Phase-1 margin, B Phase-2 brackets, C bracket width vs eps, D incumbent weighted magnitudes with
the guided-restart crossings marked.
