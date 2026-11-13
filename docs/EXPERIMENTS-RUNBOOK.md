# Experiments runbook (GPU box required)

No result in the paper's new tables (E1-E10) exists yet; this is how to produce them.

## E1/E2 — equal-budget search arms
```
pip install -e .[dev]
python scripts/make_arm_configs.py configs/templates/evolutionary_template.yaml configs/paper/arms   # regenerates the 8 arms
for a in configs/paper/arms/*.yaml; do mmb-evolve $a --device cuda; done
```
Arms: a0 directed (AGE-MOEA-II, 3 objectives) · a1 random, tiered init · a2 random, uniform init ·
a3 single-objective GA on proximity only · a4 NSGA-II · a5 MIMICRY-style softmax-balance objective ·
a6 text-only · a7 image-only. a5 changes only the boundary objective (`optimizer.boundary_objective`),
so it isolates the log-likelihood-margin contribution.

Before running: set the same `seeds:` block, `generations`, `pop_size` in the base config (arms share
them), **disable early stop** (`optimizer.early_stop`) so "crossing share vs budget" curves are not truncated,
and use identical `seed` integers across arms (paired tests). Mapping-only arm: `mmb-pdq`.
Proximity band for "reached boundary": |g| <= 1e-2 nats (the paper's criterion).

## Still to implement (see PROGRESS.md)
Diffusion image backend, LLM-paraphrase text backend, task protocol, validity scorer, API SUT.
