#!/usr/bin/env python
"""RQ3 human study: validity shares, rater agreement and proxy calibration (thin CLI over mmb.validity.analyze_ratings).

    python scripts/validity/analyze_ratings.py study/rq3/manifest.csv ratings_*.json --out results/validity_human.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mmb.validity.analyze_ratings import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
