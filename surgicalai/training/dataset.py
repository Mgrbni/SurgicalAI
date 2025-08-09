# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import random

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from surgicalai.config import DataConfig, TrainConfig


class LesionDataset(Dataset):
    """Simple dataset reading images from disk."""

    def __init__(self, items: List[Tuple[Path, int]], transform: transforms.Compose):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


def _load_from_folder(cfg: DataConfig, class_to_idx: dict[str, int]):
    items: List[Tuple[Path, int, str]] = []
    for cls_name, idx in class_to_idx.items():
        for p in (Path(cfg.root) / cls_name).glob("*.jpg"):
            items.append((p, idx, "train"))
    return items


def _load_from_csv(cfg: DataConfig, class_to_idx: dict[str, int]):
    df = pd.read_csv(cfg.csv)
    items: List[Tuple[Path, int, str]] = []
    for _, row in df.iterrows():
        path = Path(row["path"])
        label = class_to_idx[str(row["label"])]
        split = row.get("split", "train")
        items.append((path, label, split))
    return items


def _load_from_db(cfg: DataConfig, class_to_idx: dict[str, int]):
    from sqlalchemy import create_engine, text

    engine = create_engine(cfg.db_dsn)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT uri_or_path,label,split FROM images")
        ).fetchall()
    items: List[Tuple[Path, int, str]] = []
    for r in rows:
        items.append((Path(r[0]), class_to_idx[str(r[1])], r[2] or "train"))
    return items


def _split_items(items: List[Tuple[Path, int, str]]):
    train, val, test = [], [], []
    for p, lbl, split in items:
        if split == "val":
            val.append((p, lbl))
        elif split == "test":
            test.append((p, lbl))
        else:
            train.append((p, lbl))
    if not val or not test:
        random.shuffle(train)
        n = len(train)
        val = train[int(0.8 * n) : int(0.9 * n)] or train
        test = train[int(0.9 * n) :] or train
        train = train[: int(0.8 * n)] or train
    return train, val, test


def create_dataloaders(train_cfg: TrainConfig, data_cfg: DataConfig):
    class_to_idx = {n: i for i, n in enumerate(train_cfg.class_names)}
    if data_cfg.source == "folder":
        items = _load_from_folder(data_cfg, class_to_idx)
    elif data_cfg.source == "csv":
        items = _load_from_csv(data_cfg, class_to_idx)
    elif data_cfg.source == "db":
        items = _load_from_db(data_cfg, class_to_idx)
    else:  # pragma: no cover - invalid source
        raise ValueError("unknown data source")
    train_items, val_items, test_items = _split_items(items)

    tf_train = transforms.Compose(
        [
            transforms.Resize((train_cfg.img_size, train_cfg.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
        ]
    )
    tf_eval = transforms.Compose(
        [
            transforms.Resize((train_cfg.img_size, train_cfg.img_size)),
            transforms.ToTensor(),
        ]
    )

    train_ds = LesionDataset(train_items, tf_train)
    val_ds = LesionDataset(val_items, tf_eval)
    test_ds = LesionDataset(test_items, tf_eval)

    # Weighted sampler for class imbalance
    counts = torch.bincount(torch.tensor([lbl for _, lbl in train_items]))
    weights = 1.0 / (counts.float() + 1e-6)
    sample_weights = [weights[lbl] for _, lbl in train_items]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )
    return train_loader, val_loader, test_loader
