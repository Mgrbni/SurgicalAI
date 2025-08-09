# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch import nn
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

from surgicalai.config import load_config, Config
from surgicalai.training.dataset import create_dataloaders
from surgicalai.training.gradcam_tools import max_gradcam
from surgicalai.utils.io import write_json


def _load_model(num_classes: int, checkpoint: Path) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(
    checkpoint: Path, config: Config | None = None, csv_out: Path | None = None
) -> None:
    cfg = config or load_config()
    _, _, test_loader = create_dataloaders(cfg.train, cfg.data)
    model = _load_model(len(cfg.train.class_names), checkpoint)
    device = torch.device("cpu")
    model.to(device)

    y_true: List[int] = []
    y_pred: List[int] = []
    probs: List[float] = []
    records: List[str] = []
    with torch.no_grad():
        for imgs, labels, paths in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            p = torch.softmax(logits, dim=1)
            prob, pred = torch.max(p, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.cpu().tolist())
            probs.extend(prob.cpu().tolist())
            for path, lbl, pr, pb in zip(
                paths, labels.tolist(), pred.cpu().tolist(), prob.cpu().tolist()
            ):
                gc = max_gradcam(model, imgs[0:1])
                if pr != lbl or pb < 0.6:
                    records.append(f"{path},{lbl},{pr},{pb:.4f},{gc:.4f}")
    if csv_out:
        header = "path,label,pred,prob,max_gradcam"
        csv_out.write_text("\n".join([header] + records))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else 0.0,
        "f1": f1_score(y_true, y_pred, average="macro"),
        "sensitivity": recall_score(y_true, y_pred, average="macro"),
        "specificity": precision_score(y_true, y_pred, average="macro"),
    }
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(cfg.train.class_names)))
    )
    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "confusion_matrix.json", cm.tolist())


def main(checkpoint: str, csv_out: str | None = None) -> None:  # pragma: no cover
    evaluate(Path(checkpoint), csv_out=Path(csv_out) if csv_out else None)
