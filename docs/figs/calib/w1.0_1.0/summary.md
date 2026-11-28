# Calibration round w1.0_1.0: quick summary

Crossing = a crossing within the budget (canonical scoring). Cells: held-out calibration cells.

margin closed = 1 - min(sign(g0) g) / |g0| over the run (>= 100 % = crossed); median over cells.

| SUT | task | arm | crossed / n | share | median margin closed | median min magnitude |
|---|---|---|---|---|---|---|
| qwen | imagenet | MMBA (a0) | 1 / 8 | 12% | 63% | 0.479 |
| qwen | imagenet | text-only (a6) | 0 / 8 | 0% | 55% | nan |
| qwen | imagenet | image-only (a7) | 0 / 8 | 0% | 37% | nan |
| qwen | pope | MMBA (a0) | 1 / 8 | 12% | 56% | 0.113 |
| qwen | pope | text-only (a6) | 0 / 8 | 0% | 40% | nan |
| qwen | pope | image-only (a7) | 2 / 8 | 25% | 20% | 0.298 |

## Text-only vs image-only (exact McNemar on matched cells)

| SUT | n matched | both | text only | image only | neither | p |
|---|---|---|---|---|---|---|
| qwen | 16 | 0 | 0 | 2 | 14 | 0.5 |
| all | 16 | 0 | 0 | 2 | 14 | 0.5 |

Canonical consistency: 0 run(s) with canonical_consistent = false.
Pairs: {'own_phase2': 4}; kinds: {'continuous': 3, 'text_step': 1}.
