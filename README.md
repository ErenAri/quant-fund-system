# 🚀 Quantitative Trading System

An automated quantitative trading system using machine learning and technical strategies for ETF portfolio management.

## 📊 Performance

**Backtest Results (2014-2025):**
- **Sharpe Ratio:** 0.68
- **Annual Return:** 3.4%
- **Max Drawdown:** 12%
- **Strategy:** 30% Momentum / 70% Mean Reversion

**Status:** ✅ Live in paper trading mode with cloud automation

## 🎯 Features

### Trading Strategy
- **ML-Enhanced Signals** - XGBoost classification for regime prediction
- **Technical Indicators** - RSI, MACD, ADX, Bollinger Bands
- **Dual Strategy** - Momentum + Mean Reversion combination
- **Risk Management** - Kelly sizing, ATR stops, position limits

### Execution
- **Broker Integration** - Alpaca API (paper & live trading)
- **Automated Trading** - Cloud VM with cron scheduling
- **Portfolio Rebalancing** - Daily at 9:50 AM ET
- **Trade Logging** - Complete audit trail

### Infrastructure
- **Docker Container** - Portable deployment
- **Cloud Ready** - AWS, GCP, Azure support
- **Monitoring** - Health checks & error notifications
- **Cost Effective** - ~$0-15/month to run

## 🚀 Quick Start

### Local Testing

```bash
# 1. Clone repository
cd C:\projects\quant-fund-mvp

# 2. Configure API keys (.env already created)
# ALPACA_API_KEY and ALPACA_SECRET_KEY

# 3. Run paper trading
python scripts/run_live.py --mode paper
```

### Cloud Deployment (10 Minutes)

```bash
# Complete guide in deploy/QUICKSTART.md
# Automated setup for AWS EC2, runs daily

cd deploy
cat QUICKSTART.md
```

## 📈 What Just Happened

We built a complete production-ready trading system:

1. ✅ **Backtested strategy** (0.68 Sharpe)
2. ✅ **Alpaca integration** (paper trading)
3. ✅ **Cloud deployment** (Docker + cron)
4. ✅ **Automated execution** (Mon-Fri 9:50 AM ET)
5. ✅ **Monitoring** (logs + health checks)

## 🏗️ Project Structure

```
quant-fund-mvp/
├── src/quantfund/          # Core trading library
├── scripts/                # Executable scripts
│   ├── run_live.py        # Main trading script
│   ├── health_check.py    # System monitoring
│   └── run_scheduled.sh   # Cron wrapper
├── deploy/                 # Cloud deployment
│   ├── QUICKSTART.md      # 10-min deploy
│   └── DEPLOYMENT.md      # Full guide
├── configs/                # Configuration
├── artifacts/              # Trained models
├── logs/                   # Trade logs
├── Dockerfile             # Container
└── .env                   # API keys (gitignored)
```

## 🔧 Configuration

All settings in `configs/live_trading.yaml`:

```yaml
strategy:
  interval: "1d"           # Daily rebalancing
  momo_weight: 0.3         # 30% momentum, 70% mean-reversion

risk:
  annual_vol_target: 0.20  # 20% target volatility
  per_trade_risk: 0.015    # 1.5% risk per trade

limits:
  max_position_size: 0.30  # Max 30% in single ETF
  max_num_positions: 12    # Max 12 positions
```

## 📊 Universe

12 Liquid ETFs:
- **Indices:** SPY, QQQ, IWM, DIA
- **Sectors:** XLK, XLF, XLE, XLY, XLP, XLV
- **Bonds:** TLT, HYG

## 💰 Costs

### Cloud VM
- **Free Tier:** $0/month (AWS t3.micro, first 12 months)
- **After:** $7-15/month depending on instance size
- **Total:** ~$0-16/month including data transfer

### Broker
- **Alpaca:** Commission-free, free API

## 🔐 Security

- ✅ API keys in `.env` (gitignored)
- ✅ SSH key authentication for VM
- ✅ Firewall: SSH only from your IP
- ✅ Never commit credentials

## 🧪 Testing

```bash
# Test broker connection
python scripts/test_broker_integration.py

# Health check
python scripts/health_check.py

# Manual trade run
python scripts/run_live.py --mode paper
```

## 📊 Monitoring

```bash
# View logs
tail -f logs/live/trading_*.log

# Check system health
python scripts/health_check.py

# Alpaca dashboard
# Visit: https://app.alpaca.markets
```

## 🗺️ Roadmap

### Phase 1: ETF Strategy ✅ COMPLETE
- [x] Backtest optimization (0.68 Sharpe)
- [x] Paper trading integration
- [x] Cloud deployment automation
- [x] Monitoring & logging

### Phase 2: Options Strategy (Next)
- [ ] Adaptive iron condors
- [ ] Volatility regime trading
- [ ] Earnings cycle positioning

### Phase 3: Production Scaling
- [ ] Real-time monitoring dashboard
- [ ] Multiple strategies in parallel
- [ ] Advanced risk analytics

## 📚 Documentation

- **Quick Deploy:** `deploy/QUICKSTART.md` - Deploy in 10 minutes
- **Full Guide:** `deploy/DEPLOYMENT.md` - Complete production setup
- **Summary:** `DEPLOYMENT_SUMMARY.md` - What we built

## ⚠️ Disclaimer

**Educational purposes only. Not financial advice.**

- Trading involves substantial risk
- Past performance ≠ future results
- You can lose money
- Test thoroughly with paper trading first
- Use at your own risk

## 📜 License

MIT License

---

**Status:** ✅ Operational | **Mode:** Paper Trading | **Updated:** October 2025
