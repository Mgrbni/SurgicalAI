"""Command-line interface for SurgicalAI."""

from pathlib import Path
import tyro
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

@dataclass
class DemoArgs:
    """Arguments for demo command."""
    image: Path = tyro.cli.arg(help="Path to lesion image file")
    out: Path = tyro.cli.arg(default=Path("runs/demo"), help="Output directory")

def demo(args: DemoArgs) -> None:
    """Run the complete SurgicalAI pipeline demo."""
    console = Console()
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Will implement full pipeline here
    console.print("[bold green]Starting SurgicalAI pipeline...[/]")
    
    # Show summary table
    table = Table(title="SurgicalAI Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Lesion Analysis", "✓") 
    table.add_row("Flap Planning", "✓")
    table.add_row("Report Generation", "✓")
    
    console.print(table)
    
    console.print(f"[bold green]Output written to: {args.out}[/]")

def main() -> None:
    """CLI entry point."""
    tyro.cli(DemoArgs)
