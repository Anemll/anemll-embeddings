"""Pytest fixtures so script-style tests collect under pytest.

The suite is also runnable as ``python tests/test_*.py``. Those files take a
``tmp`` (or ``monkey_home``) path argument from ``main()``. This aliases the
standard ``tmp_path`` fixture so ``python -m pytest tests`` works too.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def monkey_home(tmp_path: Path) -> Path:
    return tmp_path
