"""Offline analysis of experiment artifacts.

Nothing here calls a model, a judge, or a GPU. Every module reads committed
artifacts and writes committed artifacts, so any conclusion drawn from it can be
recomputed by a grader with `python -m analysis.<module>`.
"""
