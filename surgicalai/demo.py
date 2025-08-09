"""Module to run demo via python -m surgicalai.demo."""
from .cli import demo, app
import typer

if __name__ == '__main__':
    typer.run(demo)
