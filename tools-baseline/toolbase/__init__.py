"""Platon and MOB-suite as baselines: shard, parse, apply the published rules, score.

Unlike every other baseline in this project these two are not retrained. They are
database- and rule-based predictors run exactly as published, so the only code
here is the harness around them: splitting the test split into inputs they were
designed for, parsing their outputs without guessing at column positions, and
re-deriving their documented decision rules so that one run yields all three of
Platon's filter modes plus a continuous score.
"""
