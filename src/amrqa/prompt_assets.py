"""Packaged prompt assets used to keep long demonstrations out of control-flow code."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def load_react_demonstrations() -> str:
    """Load the eight trajectories retained from the original AMRQA ReAct prompt."""
    return files("amrqa").joinpath("assets/react_fewshot.txt").read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def load_atomic_fact_demonstrations() -> str:
    """Load the four atomic-fact examples retained from the original prompt."""
    return (
        files("amrqa")
        .joinpath("assets/atomic_fact_fewshot.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
