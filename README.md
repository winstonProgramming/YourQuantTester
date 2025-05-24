<h1 align="center">YourQuantTester</h1>

YourQuantTester is a comprehensive backtesting system that evaluates quantitative trading strategies. It is capable of datascraping OHLCV data, generating indicators and signals, and backtesting strategies, providing detailed performance metrics.

## 📈 Features

- Backtests long and/or short strategies
- Customizable technical indicators (RSI, EMA, Stochastic, etc.)
- Supports multiple signal types: divergences, breakouts, candlestick patterns
- Multi-layered sell criteria (indicators, support/resistance, margin, time-based)
- Portfolio management and dynamic position sizing
- Sharpe ratio and probability-of-profit simulation (only supported for long strategies)

## 🧠 Strategy Overview

The strategy uses technical indicators to generate buy signals based on the chronological order of signals. It evaluates trades using quality scores and triggers sell conditions via nested rules involving indicators, artificial margins, resistance levels, and time constraints.

## 🔧 Configuration

Settings are controlled via the `main.py` module, including:

- Essential:
  - **Data path**: Path to store data

- Non-essential:
  - **Date range**: Within the past year
  - **Universe**: Up to 500 international stocks
  - **Candle interval**: `1h`
  - **Indicators**: Customizable RSIs, EMAs, and Stochastics
  - **Signals**: Customizable divergences and breakouts; non-customizable candlesticks
  - **Buy signals**: Customizable chronologically-ordered signals
  - **Sell signals**: Customizable nested rules involving indicators, artificial margins, resistance levels, and time constraints
  - **Position sizing**: Based on portfolio money/value (with option for quality-based sizing)
  - **Performance metrics**: Sharpe ratio, trade win/loss stats, volatility

## ⚙️ Key Modules
- `get_tickers.py` – Generates tickers list and scrapes OHLCV data
- `indicators.py` – RSI, EMA, Stochastic, volatility, and candlestick processing
- `signals.py` – Highs and lows, supports and resistances, breakouts, divergences, and Stochastic crossovers processing
- `trade_signals.py` – Trade construction logic
- `backtest.py` – Simulates order execution and equity tracking

## ▶️ Usage

1. Install required packages:
    ```bash
    pip install pandas numpy matplotlib plotly subplots graph_objects yfinance
from itertools import permutations, product
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
talib
    ```
    Install TA-Lib as well https://www.youtube.com/watch?v=hZIZMMcTQ8c
2. Ensure you set `config.scrape_data_bool = True` on first run to download fresh data.
3. Run the `main.py` module.

## 📊 Backtest Results


Start date: 2024-10-01
End date: 2025-04-01

Performance Summary

Annualized Return (Model): 16.05%

Sharpe Ratio (Hourly): 6.57

Volatility: 4.90%

Average % of Portfolio in Market: 28.75%

Trade Statistics

Total Trades: 1217

Trades Won: 740

Trades Lost: 477

Avg Profit per Trade (raw): 0.37%

Avg Profit per Trade (size-adjusted): 0.012%

Avg Winning Trade: 1.46%

Avg Losing Trade: -1.31%

Avg Trade Duration: 6.75 hours

Max Concurrent Positions: 42

Weekly Outcome

Weeks Won: 15

Weeks Lost: 10

Weeks Uninvested: 0

Probability of Profit in 1 Week: 64.01%

## 📁 File Organization

```plaintext
.
├── config.py
├── calibrate_config.py
├── file_management.py
├── get_tickers.py
├── indicators.py
├── signals.py
├── trade_signals.py
├── backtest.py
├── graph.py
└── main.py (the script you provided)
```

<p>For an in depth explanation of the simulator, watch my YouTube video on it: https://youtu.be/mg2qNVv_FS8?si=Td6XSo1LUoIVoerH</p>

<div>
  <img src="Stock Market Day Trading Simulator Images/img 1.png" width=650px>
  <img src="Stock Market Day Trading Simulator Images/img 2.png" width=650px>
  <img src="Stock Market Day Trading Simulator Images/img 3.png" width=650px>
</div>
