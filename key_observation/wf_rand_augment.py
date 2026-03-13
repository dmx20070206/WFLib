from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wf_primitives import PRIMITIVES, WFPrimitive


class RandAugment:
    # n = number of primitives to apply
    # m = magnitude (strength) of each primitive
    # n ∈ {1, 2, 3}, m ∈ {1, 2, 3, 4, 5}
    N_MIN: int = 1
    N_MAX: int = 3
    M_MIN: int = 1
    M_MAX: int = 5

    def __init__(
        self,
        n: int,
        m: int,
        primitives: Optional[dict[str, WFPrimitive]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not (self.N_MIN <= n <= self.N_MAX):
            raise ValueError(f"n must be in [{self.N_MIN}, {self.N_MAX}], got {n}")
        if not (self.M_MIN <= m <= self.M_MAX):
            raise ValueError(f"m must be in [{self.M_MIN}, {self.M_MAX}], got {m}")

        self.n = n
        self.m = m
        self.primitives: dict[str, WFPrimitive] = primitives if primitives is not None else PRIMITIVES

        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Core augmentation API
    # ------------------------------------------------------------------

    def augment_trace(self, trace: np.ndarray) -> np.ndarray:
        ops = self._rng.sample(list(self.primitives.values()), k=self.n)
        out = trace.copy()
        for op in ops:
            out = op(out, self.m)
        return out

    def augment_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_aug = np.stack(
            [self.augment_trace(X[i]) for i in range(len(X))],
            axis=0,
        )
        return X_aug, y.copy()

    def augment_npz(self, src_path: str | Path, dst_path: str | Path) -> None:
        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        data = dict(np.load(src_path))
        X_aug, y_aug = self.augment_dataset(data["X"], data["y"])
        data["X"] = X_aug
        data["y"] = y_aug
        np.savez(dst_path, **data)

    # ------------------------------------------------------------------
    # Search-space helpers  (used by the bash search pipeline)
    # ------------------------------------------------------------------

    @classmethod
    def search_space(cls) -> list[tuple[int, int]]:
        return [(n, m) for n in range(cls.N_MIN, cls.N_MAX + 1) for m in range(cls.M_MIN, cls.M_MAX + 1)]

    @classmethod
    def from_config(
        cls,
        config: dict,
        **kwargs,
    ) -> "RandAugment":
        return cls(n=config["n"], m=config["m"], **kwargs)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ops = ", ".join(self.primitives.keys())
        return f"RandAugment(n={self.n}, m={self.m}, " f"primitives=[{ops}])"


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point  (used by the bash search loop)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply RandAugment to an NPZ traffic dataset.")
    parser.add_argument("--src", default=None, help="Source .npz file path")
    parser.add_argument("--dst", default=None, help="Destination .npz file path")
    parser.add_argument("--n", default=3, type=int, help="Number of primitives (1-3)")
    parser.add_argument("--m", default=5, type=int, help="Magnitude (1-5)")
    parser.add_argument("--seed", default=20070206, type=int, help="Random seed")
    args = parser.parse_args()

    ra = RandAugment(n=args.n, m=args.m, seed=args.seed)
    ra.augment_npz(args.src, args.dst)
    print(f"[RandAugment] n={args.n} m={args.m}  {args.src} -> {args.dst}")
