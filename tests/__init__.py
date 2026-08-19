"""Test package.

This file exists to make `tests` an unambiguous regular package. Several tests
share helpers via `from tests.test_filtering import make_example`, and without
`__init__.py` that import resolves through namespace-package rules — which means
a top-level `tests` module anywhere else on `sys.path` can shadow this one. That
is not hypothetical: the suite passed locally and failed on a fresh Colab clone
with `ModuleNotFoundError: No module named 'tests.test_filtering'`, because
something in that image ships its own `tests`.

A regular package plus `pythonpath = ["."]` in `pyproject.toml` settles it: the
repository root goes on the front of `sys.path`, and this package wins.
"""
