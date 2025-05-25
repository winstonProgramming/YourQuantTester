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
    pip install numpy matplotlib pandas plotly yfinance
    ```
    Install TA-Lib as well: https://www.youtube.com/watch?v=hZIZMMcTQ8c.
2. Ensure you set `config.scrape_data_bool = True` on the first run to download fresh data.
3. Run the `main.py` module.

## 📊 Backtest Results
- Start date
- End date
- Model return
- Average percentage of portfolio in the market
- Trades
- Trades won
- Trades lost
- Average profit per trade (discounting trade size)
- Average profit per trade (accounting trade size)
- Average profit of winning trade
- Average profit of losing trade
- Average trade duration
- Most positions at once
- Sharpe ratio
- Volatility
- Volatility multiplied by average percentage of portfolio in the market

## 📁 File Organization
```plaintext
YourQuantTester/
├── main.py                                # Main script for running and configuring
├── config.py                              # Stores variables
├── calibrate_config.py                    # Adjusts configurations
├── file_management.py                     # Handles data storage and file organization
├── get_tickers.py                         # Retrieves tickers
├── scrape_data.py                         # Scrapes OHLCV data
├── indicators.py                          # Generates indicators
├── signals.py                             # Generates signals
├── get_time_delta.py                      # Calculates time difference between two dates
├── trade_signals.py                       # Generates buy and/or sell signals
├── backtest.py                            # Executes trades and computes metrics
├── graph.py                               # Creates graphs
├── tickers.csv                            # Records tickers
├── order list.csv                         # Records orders
├── portfolio holdings hourly.csv          # Records portfolio value every hour
├── portfolio holdings daily.csv           # Records portfolio value every day
└── portfolio holdings weekly.csv          # Records portfolio value every week

Downloads/
├── stocks_csv/
│   ├── 1h/
│   │   ├── raw data/
│   │   ├── rsi/
│   │   ├── ema/
│   │   ├── stochastic/
│   │   ├── volatility/
│   │   ├── candle stick/
│   │   ├── highs and lows/
│   │   │   ├── 1/
│   │   │   │   ├── highs price 1/
│   │   │   │   ├── lows price 1/
│   │   │   │   ├── highs rsi 1/
│   │   │   │   └── lows rsi 1/
│   │   │   ├── 2/
│   │   │   │   ├── highs price 2/
│   │   │   │   ├── lows price 2/
│   │   │   │   ├── highs rsi 2/
│   │   │   │   └── lows rsi 2/
│   │   │   ├── 3/
│   │   │   │   ├── highs price 3/
│   │   │   │   └── lows price 3/
│   │   │   ├── 4/
│   │   │   │   ├── highs price 4/
│   │   │   │   └── lows price 4/
│   │   ├── supports and resistances/
│   │   ├── breakouts/
│   │   ├── divergences/
│   │   ├── stochastic crossover divergence/
│   │   ├── buy signals/
│   │   ├── sell signals/
└── └── └── order list/
```
<br>
For an in depth explanation of YourQuantTester, watch my YouTube video on it: https://youtu.be/mg2qNVv_FS8?si=Td6XSo1LUoIVoerH.
<br>
<br>
Image of order list.csv:
<div><img src="Stock Market Day Trading Simulator Images/img 1.png" width=650px></div>
<br>
Image of backtest results:
<div><img src="Stock Market Day Trading Simulator Images/img 2.png" width=650px></div>
<br>
Image of expected distribution of profits:
<div><img src="Stock Market Day Trading Simulator Images/img 3.png" width=650px></div>
