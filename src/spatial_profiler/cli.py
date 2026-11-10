"""Command-line interface for spatial-microenvironment-profiler."""

from pathlib import Path

import typer
from rich.console import Console

from spatial_profiler import __version__
from spatial_profiler.config import load_config
from spatial_profiler.qc_preprocess import run_spatial_qc_pipeline

app = typer.Typer(
    name="spatial-profiler",
    help="End-to-end spatial transcriptomics (Visium HD) and morphology analysis toolkit.",
    add_completion=False,
)
console = Console()


@app.callback()
def main_callback(
    version: bool | None = typer.Option(None, "--version", "-v", help="Show software version."),
) -> None:
    if version:
        console.print(f"[bold green]spatial-profiler[/bold green] version: {__version__}")
        raise typer.Exit()


@app.command("qc")
def run_qc(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Path to raw Visium Space Ranger directory or .h5ad file."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Path to write preprocessed AnnData (.h5ad)."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to custom YAML parameters file."
    ),
) -> None:
    """Run spatial quality control, spot filtering, normalization, and HVG selection."""
    console.print("[bold blue]Loading configuration...[/bold blue]")
    cfg = load_config(config_path)

    console.print(f"[bold cyan]Running QC on {input_path}...[/bold cyan]")
    adata = run_spatial_qc_pipeline(input_path, config=cfg.qc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    console.print(f"[bold green]✓ Preprocessed dataset written to {output_path}[/bold green]")


@app.command("info")
def show_info() -> None:
    """Display system environment and package metadata."""
    import importlib.metadata

    import torch

    from spatial_profiler.r_bridge import is_rscript_available

    console.print("[bold yellow]spatial-microenvironment-profiler System Info[/bold yellow]")
    console.print(f"Version: {__version__}")
    console.print(f"Scanpy: {importlib.metadata.version('scanpy')}")
    console.print(f"AnnData: {importlib.metadata.version('anndata')}")
    console.print(f"PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    console.print(
        f"Rscript on PATH: {'✓ YES' if is_rscript_available() else '✗ NO (run inside Docker for RCTD)'}"
    )


if __name__ == "__main__":
    app()
