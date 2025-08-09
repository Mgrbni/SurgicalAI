# SPDX-License-Identifier: Apache-2.0
"""Synthetic sample data generator."""

from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh


def make_sample(out: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        case = out / f"case_{i:03d}"
        case.mkdir(exist_ok=True)
        img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(case / "image.png")
        mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=1)
        np.savez(case / "mesh.npz", vertices=mesh.vertices, faces=mesh.faces)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    make_sample(args.out, args.n, args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
