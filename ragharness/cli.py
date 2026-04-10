import click

from ragharness._version import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """ragharness — pluggable RAG evaluation framework."""


@main.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Print run plan without executing.")
@click.option("--output-dir", type=click.Path(), default=None, help="Override output directory.")
@click.option("--filter", "filter_str", default=None, help='Filter configs (e.g. "top_k=5").')
@click.option("--no-confirm", is_flag=True, help="Skip cost confirmation prompt.")
@click.option("--verbose", is_flag=True, help="Show per-question results as they complete.")
def run(
    config: str,
    dry_run: bool,
    output_dir: str | None,
    no_confirm: bool,
    filter_str: str | None,
    verbose: bool,
) -> None:
    """Run an evaluation sweep from a config file."""
    click.echo("run: not yet implemented")


@main.command()
@click.argument("config", type=click.Path(exists=True))
def validate(config: str) -> None:
    """Validate a config file without running."""
    click.echo("validate: not yet implemented")


@main.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path(), default=None, help="Output directory for charts.")
def report(csv_path: str, output_dir: str | None) -> None:
    """Re-generate reports from an existing CSV result file."""
    click.echo("report: not yet implemented")
