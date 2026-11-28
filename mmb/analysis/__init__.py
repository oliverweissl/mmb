"""Analysis pipeline (continuous genotype): run directories -> per-run CSV -> statistics, LaTeX tables, figures.

* :mod:`mmb.analysis.collect`     -- one row per run from trace.parquet, stats.json, min_cross.json, boundary_pair.json
                                    (budget-bound metrics from in-budget evaluations only)
* :mod:`mmb.analysis.stats`       -- Wilson CIs, paired tests vs a reference arm (incl. smallest-crossing magnitude),
                                    bootstrap bands, LaTeX tables
* :mod:`mmb.analysis.paired`      -- cell-cluster bootstrap, primary comparisons (MMBA vs MIMICRY-style / DeepJanus-style)
* :mod:`mmb.analysis.plots`       -- crossing-share-vs-budget figure
* :mod:`mmb.analysis.calibration` -- (w_img, w_txt) calibration: step configs, trajectory table and figure
* :mod:`mmb.analysis.export`      -- results -> paper (tables, figures, numbers.tex)

All of it is CPU-only (numpy / pandas / scipy / matplotlib).
"""
