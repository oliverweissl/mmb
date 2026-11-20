"""Analysis pipeline: run directories -> per-run CSV -> statistics, LaTeX tables, figures.

* :mod:`mmb.analysis.collect` -- one row per run from ``trace.parquet`` + ``stats.json``
* :mod:`mmb.analysis.stats`   -- Wilson CIs, paired tests vs a reference arm, bootstrap bands, LaTeX tables
* :mod:`mmb.analysis.plots`   -- crossing-share-vs-budget figure

All of it is CPU-only (numpy / pandas / scipy / matplotlib).
"""
