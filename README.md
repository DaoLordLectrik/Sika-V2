# Sikanonti - Crypto/Commodity Trading Signal Bot

A rule-based trading signal bot that watches crypto and tokenized commodity pairs, computes technical indicators every 15 minutes, and sends Telegram alerts with BUY/SELL recommendations.

## Architecture

- **Live Data**: Kraken API (US-accessible, no IP blocking)
- **Historical Data**: Binance API (for backtesting)
- **Scheduling**: GitHub Actions + external cron trigger
- **Health Check**: Separate heartbeat workflow

## Setup

1. Create Telegram bot via @BotFather
2. Add secrets to GitHub repo
3. Configure external cron trigger
4. Deploy via GitHub Actions

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your credentials
python scripts/run_signal_check.py