# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch import nn

from surgicalai.config import load_config, Config
from surgicalai.training.dataset import create_dataloaders
from surgicalai.training.train import _build_model


def retrain(from_csv: Path, epochs: int, config: Config | None = None) -> None:
    cfg = config or load_config()
    # base loaders
    train_loader, val_loader, _ = create_dataloaders(cfg.train, cfg.data)
    # oversample misclassified
    df = pd.read_csv(from_csv)
    extra_items = [(Path(p), int(lbl)) for p, lbl in zip(df["path"], df["label"])]
    if extra_items:
        # simple oversampling by extending dataset
        for _ in range(5):
            train_loader.dataset.items.extend(extra_items)  # type: ignore[attr-defined]
    device = torch.device("cpu")
    model = _build_model(len(cfg.train.class_names))
    state = torch.load(Path("models/resnet50_best.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), Path("models/resnet50_best.pt"))


def main(from_csv: str, epochs: int = 3) -> None:  # pragma: no cover
    retrain(Path(from_csv), epochs)
