# Experiments runbook (GPU box)

Order of a campaign in the continuous design. Details: [EXP_SETUP.md](../EXP_SETUP.md); decisions:
[METHODOLOGY.md](../METHODOLOGY.md); design: [METHODOLOGY_REVIEW_2026-11-28.md](METHODOLOGY_REVIEW_2026-11-28.md),
[SEARCH_REAL_CODED.md](SEARCH_REAL_CODED.md), [DETERMINISM_2026-11-28.md](DETERMINISM_2026-11-28.md).

1. **Question pool** (only when cells change): `python scripts/build_question_pool.py` (GPU, ~7 min), then
   `scripts/plot_question_pool.py` and `scripts/review_question_pool.py` for the review figures and sheet. Writes
   `configs/text/question_pool.json` (what runs read) and `question_pool_rejected.json.gz` (review only).
2. **Smoke**: one small config (e.g. `configs/generated/SMOKE_continuous/`) through `mmb-evolve <cfg> --device cuda`;
   check `min_cross.json` and `boundary_pair.json`, then `scripts/analysis/run_analysis.sh runs/SMOKE/continuous SMOKE`.
3. **Calibration of (w_img, w_txt)**: per step `python -m mmb.analysis.calibration configs ...`, run the step's configs
   with `scripts/run_queue.sh`, then `python -m mmb.analysis.calibration analyze ...` (trajectory table + figure).
   Stop at the weakest balanced step (McNemar p >= 0.05, both modalities mostly crossing).
4. **MAIN**: set the adopted weights in the base configs, regenerate `configs/generated/MAIN`
   (`scripts/make_arm_configs.py`), run `scripts/run_queue.sh configs/generated/MAIN` (strictly serial, resumable).
   Never edit code while a queue runs.
5. **Replays**: `scripts/replay_maintenance.sh` (greedy agreement, modality swap, transfer).
6. **Export**: `scripts/analysis/export_to_paper.sh <paper_repo>`, review the diff in the paper repo.
