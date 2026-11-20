"""Automatic validity proxies (E7): similarity of a crossing input to its seed.

* :mod:`mmb.validity.proxies` -- image and text proxy functions and the lazily loaded CPU models
* :mod:`mmb.validity.score`   -- CLI: run directories -> one CSV row per crossing run
* :mod:`mmb.validity.calibrate` -- agreement of each proxy with human validity labels (needs HS-01 stimuli)

The proxies are *descriptive*: none of them is validity. They become evidence of validity only
after calibration against human judgments (see calibrate.py).
"""
