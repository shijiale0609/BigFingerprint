"""Public Python API for BigFingerprint."""

from .bigfingerprint import generate_bigfingerprint_vector
from .bigfingerprint_preprocess import (
    add_missing_h_endgroups,
    expand_h_endgroups,
    needs_h_endgroup_completion,
    preprocess_bigsmiles,
)

__version__ = "0.2.0"

__all__ = [
    "add_missing_h_endgroups",
    "expand_h_endgroups",
    "generate_bigfingerprint_vector",
    "needs_h_endgroup_completion",
    "preprocess_bigsmiles",
]
