from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

def feature_vector_from_abcde(abcde) -> np.ndarray:
    return np.array([[abcde.asymmetry, abcde.border_irregularity, abcde.color_variegation, abcde.diameter_px, abcde.elevation_satellite]], dtype=np.float32)

class SimpleAnn:
    def __init__(self):
        self.vectors = None
        self.meta = []

    def fit(self, vectors: np.ndarray, meta: list[dict]):
        self.vectors = vectors.astype(np.float32)
        self.meta = meta
        self.nn = NearestNeighbors(n_neighbors=min(6, len(meta)), metric="cosine").fit(self.vectors)

    def query(self, v: np.ndarray, k=6) -> list[dict]:
        if self.vectors is None or len(self.meta)==0: return []
        d, idx = self.nn.kneighbors(v, n_neighbors=min(k, len(self.meta)))
        return [self.meta[i] | {"dist": float(d[0][j])} for j,i in enumerate(idx[0])]

def save_thumbnails(hits: list[dict], out_dir: Path, size=(256,256)):
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, h in enumerate(hits):
        p = Path(h["path"])
        if p.exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail(size)
            im.save(out_dir / f"nn_{j+1}.jpg")
