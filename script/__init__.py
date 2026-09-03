"""Public Python API for BigFingerprint."""

from .api import FingerprintResult, generate_bigfingerprint_vectors
from .bigfingerprint import generate_bigfingerprint_vector
from .bigfingerprint_preprocess import (
    add_missing_h_endgroups,
    expand_h_endgroups,
    needs_h_endgroup_completion,
    preprocess_bigsmiles,
)

__version__ = "0.1.0"

__all__ = [
    "FingerprintResult",
    "add_missing_h_endgroups",
    "expand_h_endgroups",
    "generate_bigfingerprint_vector",
    "generate_bigfingerprint_vectors",
    "needs_h_endgroup_completion",
    "preprocess_bigsmiles",
]
