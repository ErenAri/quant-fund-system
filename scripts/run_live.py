"""
Run live/paper trading.

Before running:
1. Sign up for Alpaca (https://alpaca.markets) - free paper trading
2. Get API keys from dashboard
3. Set environment variables:
   export ALPACA_API_KEY="your_key"
   export ALPACA_SECRET_KEY="your_secret"

Usage:
    python scripts/run_live.py --mode paper
    python scripts/run_live.py --mode live  # Real money!
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

from quantfund.execution.live_engine import LiveTradingEngine
from quantfund.utils.log import get_logger

# Load environment variables from .env file
# Look for .env in the project root (parent of scripts/)
# override=True ensures .env takes precedence over shell environment
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

logger = get_logger(__name__)


@click.command()
@click.option("--mode", type=click.Choice(["paper", "live"]), default="paper", help="Trading mode")
@click.option("--config", default="configs/live_trading.yaml", help="Config file path")
@click.option("--symbols", default="SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLY,XLP,XLV,TLT,HYG", help="Comma-separated symbols")
@click.option("--dry-run", is_flag=True, help="Compute signals but don't execute orders")
def main(mode: str, config: str, symbols: str, dry_run: bool):
    """Run daily trading strategy."""

    if mode == "live":
        click.confirm("⚠️  You are about to trade with REAL MONEY. Are you sure?", abort=True)

    symbol_list = [s.strip() for s in symbols.split(",")]

    logger.info(f"Starting live trading engine in {mode} mode")
    logger.info(f"Symbols: {symbol_list}")

    try:
        # Initialize engine
        engine = LiveTradingEngine(config_path=config)

        # Run strategy
        result = engine.run_daily_strategy(symbol_list)

        # Log result
        logger.info("Strategy execution complete")
        print(json.dumps(result, indent=2))

        if dry_run:
            logger.info("DRY RUN: No orders were actually executed")

        # Summary
        print("\n" + "="*60)
        print(f"Timestamp: {result['timestamp']}")
        print(f"Account Value: ${result['account_value']:,.2f}")
        print(f"Orders Generated: {len(result['orders'])}")
        print(f"Target Positions: {len([w for w in result['target_weights'].values() if w > 0])}")
        print("="*60)

    except Exception as e:
        logger.error(f"Error running live trading: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
