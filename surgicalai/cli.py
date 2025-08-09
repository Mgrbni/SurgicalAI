from __future__ import annotations

from pathlib import Path
import typer

from surgicalai.demo import run as demo_run

app = typer.Typer(help="SurgicalAI research prototype")


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


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
