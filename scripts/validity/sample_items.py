#!/usr/bin/env python
"""RQ3 human study: sample the minimal crossings and write the rating page (thin CLI over mmb.validity.rating_set).

    python scripts/validity/sample_items.py results/MAIN.csv --out study/rq3 [--n 120] [--controls 24] \
        [--arms a0_directed,a1_random_tiered,a8_hillclimb] [--proxies results/validity_MAIN.csv] [--seed 42]

Give study/rq3/rate.html (self-contained) to raters; manifest.csv is the private key and stays with the analyst.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mmb.validity.rating_set import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
