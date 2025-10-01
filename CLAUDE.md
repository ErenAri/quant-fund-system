# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantFund MVP is a medium-frequency ETF quantitative trading system combining momentum and mean-reversion strategies with calibrated probabilities and event-driven backtesting. The system operates on multiple timeframes (daily, 60m, 120m) and includes risk controls, regime filtering, and ML-based signal generation.

## Development Commands

### Installation
```bash
python -m pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests with coverage (80% minimum required)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_strategies.py -v

# Run specific test method
python -m pytest tests/test_strategies.py::TestMomentumStrategy::test_momo_signals_basic_functionality -v

# Run with custom coverage threshold
python -m pytest tests/ -v --cov-fail-under=50
```

### Data Pipeline Workflow
```bash
# 1. Fetch raw OHLCV data from yfinance
python scripts/fetch_data.py --start 2012-01-01 --end 2025-09-27

# 2. Compute features (technical indicators)
python scripts/make_dataset.py

# 3. Train ML model with time-series CV
python scripts/train_model.py --start 2012-01-01 --end 2025-09-27 --interval 60m

# 4. Run backtest
python scripts/run_backtest.py --interval 60m --use_model

# Optional: Run paper trading simulation
python scripts/run_paper.py
```

### Configuration Files
- `configs/universe.yaml` - Trading universe symbols
- `configs/train.yaml` - Training parameters (cost_bps, cv_splits, purge/embargo days)
- `configs/data.yaml` - Data fetch parameters (start/end dates, intervals)

## Architecture

### Data Flow
```
Raw OHLCV (yfinance) → Parquet (data/parquet/) →
Features (data/datasets/) → ML Model (models/) →
Signals → Backtest Engine → Performance Reports (reports/)
```

### Core Components

**Data Layer** (`src/quantfund/data/`)
- `ingest.py`: yfinance ingestion, handles 1d/60m/120m intervals, auto-adjusted OHLC, partitioned Parquet storage

**Feature Engineering** (`src/quantfund/features/`)
- `core.py`: Computes 20+ technical indicators (momentum: r_5/10/20, MACD, ADX; mean-reversion: RSI, EMA z-score, Bollinger %B; microstructure: autocorr, volume spike, overnight gaps)
- All features strictly use information available at bar close (no look-ahead)

**Strategy Layer** (`src/quantfund/strategies/`)
- `momo.py`: Momentum signals from calibrated p_up probabilities (ML model or fallback sigmoid)
- `meanrev.py`: Mean-reversion signals from z-score and RSI oversold conditions
- `filters.py`: Regime detection combining SPY trend (EMA50>200, MACD, r_20>0), volatility filter (vol20 < rolling median), credit spread (HYG/TLT ratio), and VIX threshold

**ML Models** (`src/quantfund/models/`)
- `train.py`: XGBoost binary classifier with time-series cross-validation, purge & embargo for leak prevention, isotonic calibration for probability calibration, walk-forward retraining by quarter
- Labels: next-bar return > transaction costs (default 2-3 bps)

**Backtest Engine** (`src/quantfund/backtest/`)
- `engine.py`: Event-driven backtester with next-open fills (realistic execution), position sizing via Kelly fraction capped at 15%, portfolio volatility targeting (default 10% annual), ATR-based stops (3x multiplier), per-trade risk limits, max drawdown cutoff (default 12%), daily loss stop (default 1%)

**Risk Controls** (`src/quantfund/risk/`)
- `controls.py`: Volatility scaling, position sizing, drawdown computation, daily loss circuit breakers

### Key Design Patterns

**Partitioned Parquet Storage**: All data stored as `interval=XX/symbol=YYY/data.parquet` for efficient parallel processing and symbol-level updates

**Signal Composition**: Strategies produce normalized signals [0,1]; final signal averages selected strategies (momo, meanrev, or both)

**Regime Gating**: All signals multiplied by `regime_ok` boolean mask to avoid trading in adverse market conditions

**Time-Series CV**: Custom fold generation with purge (5 days before validation) and embargo (2 days after validation) to prevent data leakage

**Cost-Aware Labeling**: Binary classification targets whether next-bar return exceeds transaction costs, not just direction

**Multi-Interval Support**: Same pipeline works for daily (1d), hourly (60m), and 2-hour (120m) bars with interval-specific annualization factors

### Testing Notes
- Tests use `hypothesis` for property-based testing
- `conftest.py` provides common fixtures (sample OHLCV DataFrames)
- Coverage requirement is 80% (configurable via `--cov-fail-under`)
- Test structure: `test_*.py` files in `tests/` directory

### Important Constraints
- **No Look-Ahead**: All feature computation and signal generation strictly use information available at bar close
- **Timezone Handling**: All timestamps normalized to UTC for consistency across data sources
- **Missing Data**: Forward-fill for price data, regime filters default to `True` if components missing
- **Empty DataFrames**: Functions return empty DataFrames with preserved schema rather than raising errors

## Recent Improvements (2025-10-01)

### Logging & Observability
- **Structured logging**: `src/quantfund/utils/log.py` provides `get_logger()` for all modules
- **Data ingestion logging**: All fetch operations logged with INFO/WARNING/ERROR levels
- **Exception tracking**: No more silent failures - all exceptions logged with type and message

### Configuration Validation
- **CostModel validation**: commission_bps and slippage_bps must be in [0, 100]
- **RiskLimits validation**: All risk parameters validated in `__post_init__`:
  - `annual_vol_target` in (0, 1]
  - `max_drawdown` in (0, 1]
  - `per_trade_risk` in (0, 1]
  - `daily_loss_stop` in (0, 1]
  - `atr_stop_mult` >= 0
  - `kelly_cap` in [0, 1]
  - `max_position_concentration` in (0, 1] (NEW - prevents >30% allocation to single asset)

### Data Validation
- **New module**: `src/quantfund/data/validators.py` provides:
  - `validate_ohlcv()`: Checks price relationships, NaN/inf, extreme moves, volume issues, data gaps
  - `validate_features()`: Checks for inf values, excessive NaN, unsorted/duplicate timestamps
  - `validate_signals()`: Ensures signals in [0,1] range, no NaN/inf
- **Usage**: Wrap data loads with validators before passing to strategies/backtest

### Bug Fixes
- **Fixed**: `per_trade_weight_cap` now handles negative volatility (returns 0.0)
- **Fixed**: `daily_loss_stopped` uses `<=` threshold (more conservative)
- **Fixed**: `TrainConfig` now frozen (immutable)
- **Fixed**: Empty Series/DataFrame now preserve index and name
- **Fixed**: Hypothesis tests properly suppress function-scoped fixture warnings
