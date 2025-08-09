from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
)

from surgicalai.config import load_config, Config
from surgicalai.training.dataset import create_dataloaders
from surgicalai.utils.io import write_json


def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _metrics(y_true: List[int], y_pred: List[int], probs: List[float]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, probs)
    except Exception:
        auroc = 0.0
    f1 = f1_score(y_true, y_pred, average="macro")
    sens = recall_score(y_true, y_pred, average="macro")
    spec = precision_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "auroc": auroc, "f1": f1, "sensitivity": sens, "specificity": spec}


def train(config: Config | None = None) -> Path:
    cfg = config or load_config()
    train_loader, val_loader, _ = create_dataloaders(cfg.train, cfg.data)
    device = torch.device("cpu")
    model = _build_model(len(cfg.train.class_names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 0
    history: Dict[str, List[float]] = {"accuracy": []}
    out_dir = Path("outputs/train")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    for epoch in range(cfg.train.max_epochs):
        model.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
        # validation
        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        probs: List[float] = []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                p = torch.softmax(logits, dim=1)
                prob, pred = torch.max(p, 1)
                y_true.extend(labels.tolist())
                y_pred.extend(pred.cpu().tolist())
                probs.extend(prob.cpu().tolist())
        m = _metrics(y_true, y_pred, probs)
        history["accuracy"].append(m["accuracy"])
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            torch.save(model.state_dict(), model_dir / "resnet50_best.pt")
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                break
    # export
    dummy = torch.randn(1, 3, cfg.train.img_size, cfg.train.img_size)
    if cfg.export.to_torchscript:
        traced = torch.jit.trace(model.cpu(), dummy)
        traced.save(model_dir / "resnet50_traced.pt")
    if cfg.export.to_onnx:
        torch.onnx.export(model.cpu(), dummy, model_dir / "resnet50.onnx", opset_version=12)

    # metrics
    write_json(out_dir / "metrics.json", {"accuracy": history["accuracy"]})
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(cfg.train.class_names))))
    write_json(out_dir / "confusion_matrix.json", cm.tolist())

    # misclassified examples
    rows = ["path,label,pred,prob,fold,epoch"]
    for (path, lbl), pred, prob in zip(val_loader.dataset.items, y_pred, probs):  # type: ignore[attr-defined]
        if pred != lbl or prob < 0.6:
            rows.append(f"{path},{lbl},{pred},{prob:.4f},val,{len(history['accuracy'])}")
    (out_dir / "misclassified.csv").write_text("\n".join(rows))

    return model_dir / "resnet50_best.pt"


def main() -> None:  # pragma: no cover - CLI entry
    train()
