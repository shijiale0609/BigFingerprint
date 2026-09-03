"""High-level, side-effect-free BigFingerprint workflows."""

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

from .bigfingerprint import generate_bigfingerprint_vector
from .bigfingerprint_preprocess import preprocess_bigsmiles


@dataclass(frozen=True)
class FingerprintResult:
    """Fingerprint candidates generated from one input BigSMILES string.

    Hydrogen end-group preprocessing can produce more than one chemically valid
    candidate.  Keeping the candidates next to their vectors avoids silently
    choosing or averaging an ambiguous structure.
    """

    input_bigsmiles: str
    candidates: Tuple[str, ...]
    vectors: np.ndarray

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        return iter(zip(self.candidates, self.vectors))

    @property
    def vector(self) -> np.ndarray:
        """Return the sole vector, raising if H expansion was ambiguous."""
        if len(self) != 1:
            raise ValueError(
                f"Expected one fingerprint candidate, but preprocessing produced {len(self)}. "
                "Use '.vectors' or iterate over the result instead."
            )
        return self.vectors[0]


def generate_bigfingerprint_vectors(
    bigsmiles: str,
    *,
    preprocess_h: bool = True,
    pecfp_iterations: int = 2,
    fp_size: int = 2048,
    verbose: bool = False,
) -> FingerprintResult:
    """Preprocess one BigSMILES string and fingerprint every resulting candidate.

    By default, explicit and missing ``[H]`` end groups are expanded first.  A
    two-dimensional array is always returned in ``result.vectors``; its rows
    correspond one-to-one with ``result.candidates``.
    """
    if not isinstance(bigsmiles, str) or not bigsmiles.strip():
        raise ValueError("bigsmiles must be a non-empty string")

    candidates = (
        tuple(preprocess_bigsmiles(bigsmiles))
        if preprocess_h
        else (bigsmiles,)
    )
    vectors = np.stack(
        [
            generate_bigfingerprint_vector(
                candidate,
                pecfp_iterations=pecfp_iterations,
                fp_size=fp_size,
                verbose=verbose,
            )
            for candidate in candidates
        ],
        axis=0,
    )
    return FingerprintResult(bigsmiles, candidates, vectors)
