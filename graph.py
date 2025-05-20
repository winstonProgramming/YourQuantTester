import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import config

def graph_stock(ticker):
    df_raw = pd.read_csv(
        config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv', header=0)
    df_rsi = pd.read_csv(
        config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv', header=0)
    df_highs_price = pd.read_csv(
        config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/2/highs price 2/{ticker}_highs price 2.csv', header=0)
    df_lows_price = pd.read_csv(
        config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/2/lows price 2/{ticker}_lows price 2.csv', header=0)
    df_highs_rsi = pd.read_csv(
        config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/2/highs rsi 2/{ticker}_highs rsi 2.csv', header=0)
    df_lows_rsi = pd.read_csv(
        config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/2/lows rsi 2/{ticker}_lows rsi 2.csv', header=0)

    dropna_df_highs_price = df_highs_price.dropna()
    dropna_df_lows_price = df_lows_price.dropna()
    dropna_df_highs_rsi = df_highs_rsi.dropna()
    dropna_df_lows_rsi = df_lows_rsi.dropna()

    fig = make_subplots(rows=2, cols=1)

    fig.append_trace(go.Candlestick(x=df_raw.index, open=df_raw['open'], high=df_raw['high'], low=df_raw['low'], close=df_raw['close']), row=1, col=1)
    fig.append_trace(go.Scatter(x=dropna_df_highs_price.index, y=dropna_df_highs_price['prices'], type='scatter', mode='lines', line_color='green'), row=1, col=1)
    fig.append_trace(go.Scatter(x=dropna_df_lows_price.index, y=dropna_df_lows_price['prices'], type='scatter', mode='lines', line_color='red'), row=1, col=1)
    fig.add_scatter(x=df_raw.index, y=df_highs_price['point_position'], mode='markers', marker=dict(size=4, color='Green'), row=1, col=1)
    fig.add_scatter(x=df_raw.index, y=df_lows_price['point_position'], mode='markers', marker=dict(size=4, color='Red'), row=1, col=1)

    fig.append_trace(go.Scatter(x=df_rsi.index, y=df_rsi[str(config.rsi_length)], line_color='purple'), row=2, col=1)
    fig.append_trace(go.Scatter(x=dropna_df_highs_rsi.index, y=dropna_df_highs_rsi['rsis'], type='scatter', mode='lines', line_color='green'), row=2, col=1)
    fig.append_trace(go.Scatter(x=dropna_df_lows_rsi.index, y=dropna_df_lows_rsi['rsis'], type='scatter', mode='lines', line_color='red'), row=2, col=1)
    fig.add_scatter(x=df_raw.index, y=df_highs_rsi['point_position'], mode='markers', marker=dict(size=4, color='Green'), row=2, col=1)
    fig.add_scatter(x=df_raw.index, y=df_lows_rsi['point_position'], mode='markers', marker=dict(size=4, color='Red'), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.show()

def graph_profits():
    df_order = pd.read_csv('order list.csv', index_col=False, header=0)
    profits_list = df_order['profit_of_trade_discounting_trade_size'].tolist()
    datetime_list = df_order[str(config.datetime_str)].tolist()

    new_profits_list = []
    new_datetime_list = []
    for counter, profit in enumerate(profits_list):
        if type(profit) is str:
            new_profits_list.append(float(profit[0:-1]))
            new_datetime_list.append(datetime_list[counter])

    new_profits_list.sort()

    x = new_datetime_list
    y = new_profits_list

    plt.figure(figsize=(0.1, 0.1))
    plt.scatter(x, y)
    plt.show()
