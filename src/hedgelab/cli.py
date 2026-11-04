"""Command-line entry point (``uv run hedgelab ...``).

Pipeline phases (calibrate, price, backtest, report) are added here as subcommands as
they are implemented; ``info`` is the environment/data smoke check.
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from hedgelab.checkpoints import Stage, save_checkpoint
from hedgelab.config import get_settings
from hedgelab.eda_report import run_eda, write_eda_report
from hedgelab.loaders import load_futures_series, load_option_quotes
from hedgelab.log import setup_logging

app = typer.Typer(
    help="Brent Asian collar: calibration, pricing and hedging.", no_args_is_help=True
)
console = Console()
logger = logging.getLogger(__name__)


@app.callback()
def _main() -> None:
    """Configure logging before any subcommand runs."""
    setup_logging(get_settings().log_level)


@app.command()
def info() -> None:
    """Load the dataset, write raw checkpoints and print a summary table."""
    settings = get_settings()
    options = load_option_quotes(settings.options_path)
    futures = load_futures_series(settings.futures_path)
    save_checkpoint(options, settings.checkpoint_dir, Stage.RAW, "options")

    table = Table(title="Option chains in dataset", header_style="bold cyan")
    for col in ("contract", "future", "expiry", "first date", "last date", "quoted %"):
        table.add_column(col)
    grouped = options.groupby("contract_code", sort=False)
    for code, chain in sorted(grouped, key=lambda kv: kv[1]["expiration"].iloc[0]):
        quoted = chain[["option_bid", "option_ask"]].notna().all(axis=1).mean()
        table.add_row(
            str(code),
            str(chain["future_contract_code"].iloc[0]),
            f"{chain['expiration'].iloc[0]:%Y-%m-%d}",
            f"{chain['date'].min():%Y-%m-%d}",
            f"{chain['date'].max():%Y-%m-%d}",
            f"{100 * quoted:.0f}",
        )
    console.print(table)
    for code, series in futures.items():
        console.print(
            f"[bold]{code}[/] futures: {len(series)} days, "
            f"{series.index.min():%Y-%m-%d} -> {series.index.max():%Y-%m-%d}"
        )


@app.command()
def eda() -> None:
    """Phase 0: run data-quality checks and write the EDA report (outputs/eda/)."""
    settings = get_settings()
    result = run_eda(settings)

    table = Table(title="Data-quality checks", header_style="bold cyan")
    for col in ("", "check", "failed / tested", "note"):
        table.add_column(col)
    for _, row in result.checks.iterrows():
        if row["passed"]:
            mark = "[green]PASS[/]"
        elif row["severity"] == "error":
            mark = "[bold red]FAIL[/]"
        else:
            mark = "[dim]info[/]"
        table.add_row(
            mark, str(row["check"]), f"{row['failed']:,} / {row['tested']:,}", str(row["comment"])
        )
    console.print(table)
    console.print(result.return_stats.round(3).to_string())

    path = write_eda_report(result, settings.output_dir / "eda" / "eda_report.html")
    console.print(f"[bold]Report:[/] {path}")
    blocking = ~result.checks["passed"] & (result.checks["severity"] == "error")
    if blocking.any():
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
