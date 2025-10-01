import os
import sys

import click


@click.command()
@click.option("--intervals", default="60m,120m", show_default=True)
@click.option("--cost_bps", default=2, show_default=True)
@click.option("--commission", default=2.0, show_default=True)
@click.option("--slippage", default=1.0, show_default=True)
@click.option("--vol_target", default=0.10, show_default=True)
@click.option("--max_dd", default=0.12, show_default=True)
@click.option("--per_trade", default=0.005, show_default=True)
@click.option("--daily_stop", default=0.01, show_default=True)
@click.option("--features_dir", default="data/datasets", show_default=True)
@click.option("--models_dir", default="models", show_default=True)
@click.option("--report", default="reports/paper.parquet", show_default=True)
def main(intervals: str, cost_bps: int, commission: float, slippage: float, vol_target: float, max_dd: float, per_trade: float, daily_stop: float, features_dir: str, models_dir: str, report: str):
    # Fetch data
    import subprocess

    subprocess.check_call([sys.executable, "scripts/fetch_data.py"])

    # Build features
    subprocess.check_call([sys.executable, "scripts/make_dataset.py", "--intervals", intervals])

    # Train
    subprocess.check_call([sys.executable, "scripts/train_model.py", "--intervals", intervals])

    # Backtest 60m by default
    subprocess.check_call([
        sys.executable,
        "scripts/run_backtest.py",
        "--interval",
        intervals.split(",")[0],
        "--commission",
        str(commission),
        "--slippage",
        str(slippage),
        "--vol_target",
        str(vol_target),
        "--max_dd",
        str(max_dd),
        "--per_trade",
        str(per_trade),
        "--daily_stop",
        str(daily_stop),
        "--out",
        report,
    ])

    click.echo(f"Paper run completed. Report at {report}")


if __name__ == "__main__":
    main()