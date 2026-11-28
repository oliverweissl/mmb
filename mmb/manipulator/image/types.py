"""Image data types kept after the 2026-11-28 trim: the VQGAN code grid used by the codec.

(The discrete patch-swap types PatchStrategy / CandidateStrategy / PatchSelection / ManipulationContext were
removed with the integer image operator; the continuous backend lives in latent.py.)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CodeGrid:
    """A VQGAN-encoded image: a spatial grid of codebook indices.

    Shape (H, W), typically (16, 16) for f16 downsampling of 256×256 input.
    Each cell value ∈ [0, n_codes) indexes a codeword in the VQGAN codebook.
    """

    indices: NDArray[np.int64]  # (H, W), read-only after construction

    def __post_init__(self) -> None:
        if self.indices.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {self.indices.shape}")
        # Defensive copy: break aliasing with caller's array
        owned = self.indices.copy()
        owned.flags.writeable = False
        object.__setattr__(self, "indices", owned)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.indices.shape[0], self.indices.shape[1])

    @property
    def n_tokens(self) -> int:
        return int(self.indices.size)

    @property
    def fingerprint(self) -> str:
        """Stable 16-char hex digest of the grid content."""
        return hashlib.sha256(self.indices.tobytes()).hexdigest()[:16]

    def replace(
        self,
        rows: NDArray[np.intp],
        cols: NDArray[np.intp],
        codes: NDArray[np.int64],
    ) -> CodeGrid:
        """Return a new grid with specified positions overwritten."""
        out = self.indices.copy()
        out.flags.writeable = True
        out[rows, cols] = codes
        return CodeGrid(out)
