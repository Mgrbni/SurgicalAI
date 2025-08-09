from __future__ import annotations

from pathlib import Path
import subprocess

import typer

from surgicalai.demo import run as demo_run
from surgicalai.training import train as train_mod
from surgicalai.training import evaluate as eval_mod
from surgicalai.training import retrain as retrain_mod
from surgicalai.datastore import db as db_mod
from surgicalai.config import load_config

app = typer.Typer(help="SurgicalAI research prototype")
dataset_app = typer.Typer(help="Dataset utilities")
app.add_typer(dataset_app, name="dataset")


@app.callback()
def root() -> None:
    """SurgicalAI command line."""


@app.command()
def demo(
    out: Path = typer.Option(..., "--out", help="Output directory"),
    with_llm: bool = typer.Option(False, help="Use LLM"),
    model: str | None = None,
) -> None:
    demo_run(out, with_llm=with_llm, model=model)


@dataset_app.command("sync")
def dataset_sync(
    dsn: str = typer.Option("", help="Database DSN"),
    root: Path | None = typer.Option(None, help="Dataset root"),
    csv: Path | None = typer.Option(None, help="CSV manifest"),
) -> None:
    if not dsn:
        typer.echo("DSN required")
        raise typer.Exit(code=1)
    db_mod.sync_from_folder_or_csv(dsn, str(root) if root else None, str(csv) if csv else None)


@app.command()
def train(
    data_source: str = typer.Option("folder"),
    root: Path = typer.Option(Path("data/lesions")),
    csv: Path = typer.Option(Path("data/lesions/labels.csv")),
    dsn: str = typer.Option(""),
    epochs: int = typer.Option(None),
    lr: float = typer.Option(None),
) -> None:
    cfg = load_config()
    cfg.data.source = data_source
    cfg.data.root = str(root)
    cfg.data.csv = str(csv)
    if dsn:
        cfg.data.db_dsn = dsn
    if epochs:
        cfg.train.max_epochs = epochs
    if lr:
        cfg.train.lr = lr
    train_mod.train(cfg)


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(Path("models/resnet50_best.pt")),
    csv_out: Path | None = typer.Option(None),
) -> None:
    eval_mod.evaluate(checkpoint, csv_out=csv_out)


@app.command()
def retrain(
    from_csv: Path = typer.Option(..., help="misclassified CSV"),
    epochs: int = typer.Option(3),
) -> None:
    retrain_mod.retrain(from_csv, epochs)


@app.command()
def package() -> None:
    """Build Windows executable via PyInstaller."""
    subprocess.run(["pyinstaller", "surgicalai.spec", "--clean"], check=True)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
