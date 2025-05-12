import time
from datetime import datetime, timedelta
import math
import statistics
import re
import os

import yfinance as yf
import talib
import pandas as pd
import numpy as np
# import multiprocessing
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from tvDatafeed import TvDatafeed, Interval
# import pylab
import urllib.request
import urllib.parse
# from collections import OrderedDict
# import ast
# import itertools
from itertools import permutations, product

import config
import process_variables
import get_time_delta
import export_csv

time1 = time.time()

# settings
config.stocks_csv_file_path = 'D:/Winston\'s Data/Downloads/stocks_csv'

config.equity = 'stocks'
config.nation = 'international'  # usa or international
config.stock_number = 10
config.candle_length = '1h'
config.date_or_datetime = 'datetime'
config.start_date = '2025-1-1'
config.end_date = '2025-2-1'
config.long = True
config.short = True

config.rsi_length = 12
config.ema_length = 12
config.fastk_period = 5
config.slowk_period = 3
config.slowd_period = 3
config.rolling_volatility_length = 30

config.high_low_range_b1 = 5
config.high_low_range_a1 = 5
config.high_low_range_b2 = 5
config.high_low_range_a2 = 1
config.high_low_range_3 = 8
config.high_low_range_4 = 3

config.divergence_expiration = 30

config.rsi_quality_multiplier = 0.1  # 10
config.ema_quality_exponential_multiplier = 0.8  # infinite
config.ema_quality_multiplier = 100  # infinite
config.stochastic_quality_multiplier = 0.1  # 10

# config.rsi_quality_multiplier = 0
# config.ema_quality_exponential_multiplier = 0
# config.ema_quality_multiplier = 0
# config.stochastic_quality_multiplier = 0

config.breakout_pattern_formation_expiration = 5
config.head_and_shoulders_multiplier = 1.2  # higher value = more restrictive
config.zone_multiplier = 0.25  # higher value = less restrictive
config.breakout_pattern_breakout_expiration = 10

config.price_difference_quality_multiplier = 3  # infinite
config.rsi_difference_quality_multiplier = 1.2  # infinite
config.rsi_level_quality_multiplier = 0.02  # 4

config.stochastic_crossover = True
config.stochastic_maximum = 50  # only is applied if definitive_stochastic == True
config.flexible_stochastic_cross_level = True  # only is applied if definitive_stochastic == True
config.stochastic_cross_level = 10  # only is applied if definitive_stochastic == True
config.stochastic_cross_expiration = 10  # only is applied if definitive_stochastic == True

config.buy_signal_order_dict = {'divergences': 0,
                         'breakouts': 1,
                         'candle sticks': 1}

# order_dependant_buy_expirations = False

config.buy_signal_expiration_dict = {'divergences': 5,  # if order_dependant_buy_expirations is True
                              'breakouts': 8,
                              'candle sticks': math.nan}

config.buy_signal_expiration_list = [10, 5]  # len(buy_signal_expiration_list) = len(buy_signal_order_dict) - 1

# config.pre_earnings_date_omission = 1  # set to -1 to be inactive, omits buy signal if buy signal is before earnings date
# config.post_earnings_date_omission = 2  # set to -1 to be inactive, omits buy signal if buy signal is after earnings date

config.quality_minimum = 0

config.sell_signals_nested_list = [['sell signal indicator 1', 'artificial margin 1'], ['support resistance 1'], ['sell time 1']]  # sell signal indicator, support resistance, artificial margin, sell time

# sell signal indicator 1
config.sell_signal_indicator_type_1 = 'rsi'  # rsi, k, d
config.sell_signal_indicator_flexible_1 = False  # True, False
config.sell_signal_indicator_value_1 = 50  # 0-100
config.sell_signal_simultaneous_fulfillment_1 = True  # requires the signal to be fulfilled at the time of the sell signal; cannot be complete then incomplete

# support resistance 1
config.support_resistance_resistance_minimum_distance_1 = 1  # 1-infinity
config.support_resistance_support_minimum_distance_1 = 1  # 0-1
config.support_resistance_resistance_high_1 = True  # True, False
config.support_resistance_simultaneous_fulfillment_1 = True  # requires the signal to be fulfilled at the time of the sell signal; cannot be complete then incomplete

# artificial margin 1
config.artificial_margin_take_profit_1 = 1.03  # 1-infinity
config.artificial_margin_stop_loss_1 = 0.99  # 0-1
config.artificial_margin_take_profit_high_1 = True  # True, False
config.artificial_margin_simultaneous_fulfillment_1 = True  # requires the signal to be fulfilled at the time of the sell signal; cannot be complete then incomplete

# sell time 1
config.sell_time_value_1 = 60

config.a = 5  # a >= 1
config.order_size_based_on_money = False
config.b = 1.3  # b >= 1
config.order_size_based_on_quality = False
config.lowest_order_quality = 10
config.highest_order_quality = 30
config.c = 20  # c > 0
config.d = 0.1  # 1 > d > 0

config.calculate_sharpe_ratio = False
config.risk_free_rate = 0.0425

config.calc_profit_odds = False
config.sims = 10000
config.hours = 8*21
config.profit_minimum = 1

# -------------------------------------------------

process_variables.process_variables()


def raw_scraping_func():
    if equity == 'stocks':
        for ticker in tickers:
            df = pd.DataFrame(yf.download(str(ticker), start=start_date, end=end_date, interval=candle_length))
            df.index.name = 'datetime'
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
            export_csv.export_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), df, 1)

    print('raw_func complete')


def rsi_func():
    rsi_lengths = [rsi_length]
    for ticker in tickers:
        df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)
        close = pd.Series(df_raw.close)
        rsi_nested_list = []
        for length in rsi_lengths:
            rsi = talib.RSI(close, timeperiod=length)
            rsi_nested_list.append(rsi.tolist())
        df = pd.DataFrame(list(zip(*rsi_nested_list)), columns=rsi_lengths)
        df.index = df_raw[date_or_datetime]
        # print(df_raw)
        # print(df_raw[date_or_datetime])
        export_csv.export_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), df, 1)

    print('rsi_func complete')


def ema_func():
    ema_lengths = [ema_length]
    for ticker in tickers:
        df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)
        close = df_raw['close'].tolist()
        ema_nested_list = []
        for length in ema_lengths:
            sma_list = []
            for i in range(length):
                sma_list.append(close[i])
            starting_sma = sum(sma_list)/length
            ema_list_price = [None]*(length-1)
            ema_list_price.append(starting_sma)
            for day in range(len(close)-length):
                ema_list_price.append(close[day+length]*(2/(length+1)) + ema_list_price[-1]*(1-(2/(length+1))))
            ema_list_price[length-1] = None
            ema_list = []
            for day in range(len(close)):
                if ema_list_price[day] is None:
                    ema_list.append(None)
                else:
                    ema_list.append(close[day]/ema_list_price[day])
            ema_nested_list.append(ema_list)
        df = pd.DataFrame(list(zip(*ema_nested_list)), columns=ema_lengths)
        df.index = df_raw[date_or_datetime]
        export_csv.export_csv(stocks_csv_file_path + '/{}/ema/{}_ema.csv'.format(candle_length, ticker), df, 1)
    print('ema_func complete')


def stochastic_func():
    for ticker in tickers:
        df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)
        high = pd.Series(df_raw.high)
        low = pd.Series(df_raw.low)
        close = pd.Series(df_raw.close)
        stochastic = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
        stochastic_nested_list = [stochastic[0], stochastic[1]]
        df = pd.DataFrame(list(zip(*stochastic_nested_list)), columns=['k', 'd'])
        df.index = df_raw[date_or_datetime]
        export_csv.export_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), df, 1)
    print('stochastic_func complete')


def volatility_func():
    for ticker in tickers:
        df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)

        pct_change_df = pd.DataFrame()
        pct_change_df['pct'] = df_raw['close'].pct_change()

        pct_list = pct_change_df['pct'].iloc[1:]
        volatility_series = pct_list.rolling(rolling_volatility_length).std(ddof=1)

        volatility_list = volatility_series.tolist()
        volatility_list.insert(0, math.nan)
        volatility_series = pd.Series(volatility_list)

        volatility_df = pd.DataFrame({date_or_datetime: df_raw[date_or_datetime], 'volatility': volatility_series.values})
        export_csv.export_csv(stocks_csv_file_path + '/{}/volatility/{}_volatility.csv'.format(candle_length, ticker), volatility_df, 1)

    print('volatility_func complete')


def candle_stick_func():
    for ticker in tickers:
        df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)

        df = pd.DataFrame()

        if long:
            hammer = talib.CDLHAMMER(df_raw.open, df_raw.high, df_raw.low, df_raw.close)
            engulfing = talib.CDLENGULFING(df_raw.open, df_raw.high, df_raw.low, df_raw.close)
            morning_star = talib.CDLMORNINGSTAR(df_raw.open, df_raw.high, df_raw.low, df_raw.close)
            three_soldiers = talib.CDL3WHITESOLDIERS(df_raw.open, df_raw.high, df_raw.low, df_raw.close)

            bullish_candle_stick = (hammer == 100) | (engulfing == 100) | (morning_star == 100) | (three_soldiers == 100)

            bullish_candle_stick_list = []
            for candle in bullish_candle_stick:
                if candle:
                    bullish_candle_stick_list.append('long')
                else:
                    bullish_candle_stick_list.append(None)

            df = df.assign(bullish_candles=bullish_candle_stick_list)

        if short:
            shooting_star = talib.CDLSHOOTINGSTAR(df_raw.open, df_raw.high, df_raw.low, df_raw.close)
            engulfing = talib.CDLENGULFING(df_raw.open, df_raw.high, df_raw.low, df_raw.close)
            evening_star = talib.CDLEVENINGSTAR(df_raw.open, df_raw.high, df_raw.low, df_raw.close)
            three_black_crows = talib.CDL3BLACKCROWS(df_raw.open, df_raw.high, df_raw.low, df_raw.close)

            bearish_candle_stick = (shooting_star == -100) | (engulfing == -100) | (evening_star == -100) | (three_black_crows == -100)

            bearish_candle_stick_list = []
            for candle in bearish_candle_stick:
                if candle:
                    bearish_candle_stick_list.append('short')
                else:
                    bearish_candle_stick_list.append(None)

            df = df.assign(bearish_candles=bearish_candle_stick_list)

        df.index = df_raw[date_or_datetime]

        # candle_stick_list = [bullish_candle_stick_list, bearish_candle_stick_list]
        # df = pd.DataFrame(list(zip(*candle_stick_list)), columns=['bullish candle stick', 'bearish candle stick'])

        export_csv.export_csv(stocks_csv_file_path + '/{}/candle stick/{}_candle stick.csv'.format(candle_length, ticker), df, 1)

    print('candle_stick_func complete')


def rsi_quality_func():
    for ticker in tickers:
        df_rsi = pd.read_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), index_col=False, header=0)
        rsi_list = df_rsi[str(rsi_length)].tolist()

        if long:
            bullish_rsi_quality_list = []
            for rsi in rsi_list:
                bullish_rsi_quality_list.append((100-rsi)*rsi_quality_multiplier)

            df_rsi = df_rsi.assign(bullish_quality=bullish_rsi_quality_list)

        if short:
            bearish_rsi_quality_list = []
            for rsi in rsi_list:
                bearish_rsi_quality_list.append(rsi*rsi_quality_multiplier)

            df_rsi = df_rsi.assign(bearish_quality=bearish_rsi_quality_list)

        df_rsi.to_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), mode='w')
    print('rsi_quality_func complete')


def ema_quality_func():
    for ticker in tickers:
        df_ema = pd.read_csv(stocks_csv_file_path + '/{}/ema/{}_ema.csv'.format(candle_length, ticker), index_col=False, header=0)
        ema_list = df_ema[str(ema_length)].tolist()

        if long:
            ema_quality_list = []
            for ema in ema_list:
                if ema > 1:
                    ema_quality_list.append(ema_quality_multiplier*(abs((ema-1)**ema_quality_exponential_multiplier)))
                else:
                    ema_quality_list.append(ema_quality_multiplier*(-1*abs((ema-1)**ema_quality_exponential_multiplier)))

            df_ema = df_ema.assign(bullish_quality=ema_quality_list)

        if short:
            ema_quality_list = []
            for ema in ema_list:
                if ema > 1:
                    ema_quality_list.append(
                        ema_quality_multiplier * (-1 * abs((ema - 1) ** ema_quality_exponential_multiplier)))
                else:
                    ema_quality_list.append(
                        ema_quality_multiplier * (abs((ema - 1) ** ema_quality_exponential_multiplier)))

            df_ema = df_ema.assign(bearish_quality=ema_quality_list)

        df_ema.to_csv(stocks_csv_file_path + '/{}/ema/{}_ema.csv'.format(candle_length, ticker), mode='w')
    print('ema_quality_func complete')


def stochastic_quality_func():
    for ticker in tickers:
        df_stochastic = pd.read_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), index_col=False, header=0)
        k_list = df_stochastic['k'].tolist()
        # d_list = df_stochastic['d'].tolist()

        if long:
            stochastic_quality_list = []
            for k in k_list:
                stochastic_quality_list.append((100-k)*stochastic_quality_multiplier)

            df_stochastic = df_stochastic.assign(bullish_quality=stochastic_quality_list)

        if short:
            stochastic_quality_list = []
            for k in k_list:
                stochastic_quality_list.append(k * stochastic_quality_multiplier)

            df_stochastic = df_stochastic.assign(bearish_quality=stochastic_quality_list)

        df_stochastic.to_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), mode='w')
    print('stochastic_quality_func complete')


def highs_lows_func():
    def highs_price_func(before_range, after_range, version):
        for ticker in tickers:
            df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)
            high = df_raw['high'].tolist()
            high_list = []
            price_list = []
            position_list = []
            for day in range(len(high)):
                if day < before_range:
                    high_list.append(None)
                    price_list.append(np.nan)
                    position_list.append(np.nan)
                elif day > len(high) - after_range - 1:
                    high_list.append(None)
                    price_list.append(np.nan)
                    position_list.append(np.nan)
                else:
                    high_violation = False
                    is_high = False
                    for range_counter in range(before_range):
                        if high[day] > high[day - range_counter - 1]:
                            pass
                        else:
                            high_violation = True
                    for range_counter in range(after_range):
                        if high[day] > high[day + range_counter + 1]:
                            pass
                        else:
                            high_violation = True
                    if high_violation is False:
                        is_high = True
                    if is_high is False:
                        position_list.append(np.nan)
                        price_list.append(np.nan)
                        high_list.append(None)
                    if is_high is True:
                        position_list.append(df_raw['high'][day]+0.008)
                        price_list.append(df_raw['high'][day])
                        high_list.append(True)
            high_low_nested_list = [high_list, price_list, position_list]
            df = pd.DataFrame(list(zip(*high_low_nested_list)), columns=['high', 'prices', 'point_position'])
            df.index = df_raw[date_or_datetime]
            export_csv.export_csv(stocks_csv_file_path + '/{}/highs and lows/{}/highs price {}/{}_highs price {}.csv'.format(candle_length, version, version, ticker, version), df, 2)

    def lows_price_func(before_range, after_range, version):
        for ticker in tickers:
            df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)
            low = df_raw['low'].tolist()
            low_list = []
            price_list = []
            position_list = []
            for day in range(len(low)):
                if day < before_range:
                    low_list.append(None)
                    price_list.append(np.nan)
                    position_list.append(np.nan)
                elif day > len(low) - after_range - 1:
                    low_list.append(None)
                    price_list.append(np.nan)
                    position_list.append(np.nan)
                else:
                    low_violation = False
                    is_low = False
                    for range_counter in range(before_range):
                        if low[day] < low[day - range_counter - 1]:
                            pass
                        else:
                            low_violation = True
                    for range_counter in range(after_range):
                        if low[day] < low[day + range_counter + 1]:
                            pass
                        else:
                            low_violation = True
                    if low_violation is False:
                        is_low = True
                    if is_low is False:
                        position_list.append(np.nan)
                        price_list.append(np.nan)
                        low_list.append(None)
                    if is_low is True:
                        position_list.append(df_raw['low'][day]-0.008)
                        price_list.append(df_raw['low'][day])
                        low_list.append(True)
            low_low_nested_list = [low_list, price_list, position_list]
            df = pd.DataFrame(list(zip(*low_low_nested_list)), columns=['low', 'prices', 'point_position'])
            df.index = df_raw[date_or_datetime]
            export_csv.export_csv(stocks_csv_file_path + '/{}/highs and lows/{}/lows price {}/{}_lows price {}.csv'.format(candle_length, version, version, ticker, version), df, 2)

    def highs_rsi_func(before_range, after_range, version):
        for ticker in tickers:
            df_rsi = pd.read_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), index_col=False, header=0)
            rsi = df_rsi[str(rsi_length)].tolist()
            high_list = []
            price_list = []
            position_list = []
            for day in range(len(rsi)):
                if day < before_range:
                    high_list.append(None)
                    price_list.append(np.nan)
                    position_list.append(np.nan)
                elif day > len(rsi) - after_range - 1:
                    high_list.append(None)
                    price_list.append(np.nan)
                    position_list.append(np.nan)
                else:
                    high_violation = False
                    is_high = False
                    for range_counter in range(before_range):
                        if rsi[day] > rsi[day - range_counter - 1]:
                            pass
                        else:
                            high_violation = True
                    for range_counter in range(after_range):
                        if rsi[day] > rsi[day + range_counter + 1]:
                            pass
                        else:
                            high_violation = True
                    if high_violation is False:
                        is_high = True
                    if is_high is False:
                        position_list.append(np.nan)
                        price_list.append(np.nan)
                        high_list.append(None)
                    if is_high is True:
                        position_list.append(df_rsi[str(rsi_length)][day]+1.4)
                        price_list.append(df_rsi[str(rsi_length)][day])
                        high_list.append('True')
            high_high_nested_list = [high_list, price_list, position_list]
            df = pd.DataFrame(list(zip(*high_high_nested_list)), columns=['high', 'rsis', 'point_position'])
            df.index = df_rsi[date_or_datetime]
            export_csv.export_csv(stocks_csv_file_path + '/{}/highs and lows/{}/highs rsi {}/{}_highs rsi {}.csv'.format(candle_length, version, version, ticker, version), df, 2)

    def lows_rsi_func(before_range, after_range, version):
        for ticker in tickers:
            df_rsi = pd.read_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), index_col=False, header=0)
            rsi = df_rsi[str(rsi_length)].tolist()
            low_list = []
            price_list = []
            position_list = []
            for day in range(len(rsi)):
                if day < before_range:
                    low_list.append(None)
                    position_list.append(np.nan)
                    price_list.append(np.nan)
                elif day > len(rsi) - after_range - 1:
                    low_list.append(None)
                    position_list.append(np.nan)
                    price_list.append(np.nan)
                else:
                    low_violation = False
                    is_low = False
                    for range_counter in range(before_range):
                        if rsi[day] < rsi[day - range_counter - 1]:
                            pass
                        else:
                            low_violation = True
                    for range_counter in range(after_range):
                        if rsi[day] < rsi[day + range_counter + 1]:
                            pass
                        else:
                            low_violation = True
                    if low_violation is False:
                        is_low = True
                    if is_low is False:
                        position_list.append(np.nan)
                        price_list.append(np.nan)
                        low_list.append(None)
                    if is_low is True:
                        position_list.append(df_rsi[str(rsi_length)][day]-1.4)
                        price_list.append(df_rsi[str(rsi_length)][day])
                        low_list.append('True')
            low_low_nested_list = [low_list, price_list, position_list]
            df = pd.DataFrame(list(zip(*low_low_nested_list)), columns=['low', 'rsis', 'point_position'])
            df.index = df_rsi[date_or_datetime]
            export_csv.export_csv(stocks_csv_file_path + '/{}/highs and lows/{}/lows rsi {}/{}_lows rsi {}.csv'.format(candle_length, version, version, ticker, version), df, 2)

    # 1, first divergence low
    highs_price_func(high_low_range_b1, high_low_range_a1, 1)
    lows_price_func(high_low_range_b1, high_low_range_a1, 1)
    highs_rsi_func(high_low_range_b1, high_low_range_a1, 1)
    lows_rsi_func(high_low_range_b1, high_low_range_a1, 1)
    # 2, second divergence low
    highs_price_func(high_low_range_b2, high_low_range_a2, 2)
    lows_price_func(high_low_range_b2, high_low_range_a2, 2)
    highs_rsi_func(high_low_range_b2, high_low_range_a2, 2)
    lows_rsi_func(high_low_range_b2, high_low_range_a2, 2)
    # 3, supports and resistances for sell signals
    highs_price_func(high_low_range_3, high_low_range_3, 3)
    lows_price_func(high_low_range_3, high_low_range_3, 3)
    # 4, breakouts and reversals
    highs_price_func(high_low_range_4, high_low_range_4, 4)
    lows_price_func(high_low_range_4, high_low_range_4, 4)

    print('high_lows_func complete')


def support_and_resistance_func():
    for ticker in tickers:
        df_highs_price = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/3/highs price 3/{}_highs price 3.csv'.format(candle_length, ticker), index_col=False, header=0)
        df_lows_price = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/3/lows price 3/{}_lows price 3.csv'.format(candle_length, ticker), index_col=False, header=0)

        highs_list = df_highs_price['prices'].tolist()
        lows_list = df_lows_price['prices'].tolist()

        supports_and_resistances_list_nested_list = []

        for day in range(len(df_highs_price[date_or_datetime])):
            new_highs_list = highs_list[0:day+1]
            new_lows_list = lows_list[0:day+1]

            new_highs_list.reverse()
            new_lows_list.reverse()

            newer_highs_list = [x for x in new_highs_list if not math.isnan(x)]
            newer_lows_list = [x for x in new_lows_list if not math.isnan(x)]

            supports_and_resistances_list = []
            try:
                supports_and_resistances_list.append(newer_highs_list[0])
            except IndexError:
                supports_and_resistances_list = []
            try:
                supports_and_resistances_list.append(newer_lows_list[0])
            except IndexError:
                supports_and_resistances_list = []
            if not supports_and_resistances_list:
                supports_and_resistances_list = [math.nan]
            supports_and_resistances_list.sort()

            for x in newer_highs_list:
                if x > supports_and_resistances_list[-1]:
                    supports_and_resistances_list.append(x)
                    supports_and_resistances_list.sort()

            for x in newer_lows_list:
                if x < supports_and_resistances_list[0]:
                    supports_and_resistances_list.append(x)
                    supports_and_resistances_list.sort()

            supports_and_resistances_value = ''
            for val in supports_and_resistances_list:
                supports_and_resistances_value = supports_and_resistances_value + '_' + str(val)

            supports_and_resistances_list_nested_list.append(supports_and_resistances_value)
        df = pd.DataFrame(supports_and_resistances_list_nested_list, columns=['supports and resistances'])
        df.index = df_highs_price[date_or_datetime]
        export_csv.export_csv(stocks_csv_file_path + '/{}/supports and resistances/{}_supports and resistances.csv'.format(candle_length, ticker), df, 1)
    print('support_and_resistance_func complete')


def breakout_func():
    for ticker in tickers:
        df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)
        dates = df_raw[str(date_or_datetime)].tolist()
        close = df_raw['close'].tolist()

        df_volatility = pd.read_csv(stocks_csv_file_path + '/{}/volatility/{}_volatility.csv'.format(candle_length, ticker), index_col=False, header=0)
        volatility = df_volatility['volatility'].tolist()

        df_highs_price = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/4/highs price 4/{}_highs price 4.csv'.format(candle_length, ticker), index_col=False, header=0)
        df_lows_price = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/4/lows price 4/{}_lows price 4.csv'.format(candle_length, ticker), index_col=False, header=0)
        highs_list = df_highs_price['prices'].tolist()
        lows_list = df_lows_price['prices'].tolist()

        highs_lows_dict = {}
        key_counter = 0
        for count in range(len(dates)):
            if not math.isnan(highs_list[count]) or not math.isnan(lows_list[count]):
                highs_lows_dict[str(key_counter)] = [dates[count], close[count], volatility[count], highs_list[count], lows_list[count]]
                key_counter += 1

        df = pd.DataFrame()

        if long:
            unverified_breakout_data = []  # format: [date, breakout price, type]

            for count in range(len(highs_lows_dict) - 5):  # inverted head and shoulders
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count)][0]) < breakout_pattern_formation_expiration * 5:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4] > highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * head_and_shoulders_multiplier:  # checking 1 >> 3
                                                    if not math.isnan(highs_lows_dict[str(count + 4)][3]):  # checking 4 == high
                                                        if abs(highs_lows_dict[str(count + 2)][3] - highs_lows_dict[str(count + 4)][3]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * zone_multiplier:  # checking 2 == 4
                                                            if not math.isnan(highs_lows_dict[str(count + 5)][4]):  # checking that 5 == low
                                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 4)][3]:
                                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count + 2)][3], 'inverted head and shoulders'])
                                                                else:
                                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count + 4)][3], 'inverted head and shoulders'])

            for count in range(len(highs_lows_dict) - 3):  # double bottom
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 3)][0], highs_lows_dict[str(count)][0]) < breakout_pattern_formation_expiration * 3:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 3)][4]:  # checking 2 > 3
                                                    if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * zone_multiplier:  # checking 1 == 3
                                                        unverified_breakout_data.append([highs_lows_dict[str(count + 3)][0], highs_lows_dict[str(count + 2)][3], 'double bottom'])

            for count in range(len(highs_lows_dict) - 5):  # triple bottom
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count)][0]) < breakout_pattern_formation_expiration * 5:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 3)][4]:  # checking 2 > 3
                                                    if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * zone_multiplier:  # checking 1 == 3
                                                        if not math.isnan(highs_lows_dict[str(count + 4)][3]):  # checking 4 == high
                                                            if abs(highs_lows_dict[str(count + 2)][3] - highs_lows_dict[str(count + 4)][3]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * zone_multiplier:  # checking 1 == 3
                                                                if not math.isnan(highs_lows_dict[str(count + 5)][4]):  # checking that 5 == low
                                                                    if abs(highs_lows_dict[str(count + 3)][4] - highs_lows_dict[str(count + 5)][4]) < highs_lows_dict[str(count + 5)][1] * highs_lows_dict[str(count + 5)][2] * zone_multiplier:  # checking 3 == 5
                                                                        if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 5)][4]) < highs_lows_dict[str(count + 5)][1] * highs_lows_dict[str(count + 5)][2] * zone_multiplier:  # checking 1 == 5
                                                                            if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 4)][3]:
                                                                                unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count + 2)][3], 'triple bottom'])
                                                                            else:
                                                                                unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count + 4)][3], 'triple bottom'])

            for count in range(len(highs_lows_dict) - 4):  # bullish rectangle
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 4)][0], highs_lows_dict[str(count)][0]) < breakout_pattern_formation_expiration * 4:
                    if not math.isnan(highs_lows_dict[str(count)][4]):  # checking that 0 == low
                        if not math.isnan(highs_lows_dict[str(count + 1)][3]):  # checking that 1 == high
                            if not math.isnan(highs_lows_dict[str(count + 2)][4]):  # checking that 2 == low
                                if not math.isnan(highs_lows_dict[str(count + 3)][3]):  # checking that 3 == high
                                    if abs(highs_lows_dict[str(count + 1)][3] - highs_lows_dict[str(count + 3)][3]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * zone_multiplier:  # checking 1 == 3
                                        if not math.isnan(highs_lows_dict[str(count + 4)][4]):  # checking that 4 == low
                                            if abs(highs_lows_dict[str(count + 2)][4] - highs_lows_dict[str(count + 4)][4]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * zone_multiplier:  # checking 2 == 4
                                                if highs_lows_dict[str(count + 1)][3] > highs_lows_dict[str(count + 3)][3]:
                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 4)][0], highs_lows_dict[str(count + 1)][3], 'bullish rectangle'])
                                                else:
                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 4)][0], highs_lows_dict[str(count + 3)][3], 'bullish rectangle'])

            if len(unverified_breakout_data) != 0:
                unverified_breakout_data.sort(key=lambda var: var[0])

                breakout_check_list_counter = 0
                breakout_data_list = []
                breakout_dates_list = []
                bug_fix_counter = 0
                for x in range(len(dates) - breakout_pattern_breakout_expiration):
                    if dates[x - bug_fix_counter] == unverified_breakout_data[breakout_check_list_counter][0]:
                        for date in dates[x - bug_fix_counter:x - bug_fix_counter + breakout_pattern_breakout_expiration]:
                            if close[dates.index(date)] > unverified_breakout_data[breakout_check_list_counter][1]:
                                breakout_data_list.append(unverified_breakout_data[breakout_check_list_counter])
                                breakout_dates_list.append(dates[dates.index(date)])
                                break
                        breakout_check_list_counter += 1
                        if breakout_check_list_counter == len(unverified_breakout_data):
                            break
                        if dates[x - bug_fix_counter] == unverified_breakout_data[breakout_check_list_counter][0]:
                            bug_fix_counter += 1

                bullish_breakout_type_list = []
                for counter, date in enumerate(dates):
                    bullish_breakout_type_list_appender = []
                    times = len([y for y in breakout_dates_list if y == date])
                    if times == 0:
                        bullish_breakout_type_list.append(None)
                    elif times > 0:
                        indices = [ind for ind, ele in enumerate(breakout_dates_list) if ele == date]
                        for ind in indices:
                            bullish_breakout_type_list_appender.append(breakout_data_list[ind][2])
                        bullish_breakout_type_list.append(bullish_breakout_type_list_appender)

                bullish_breakout_list = []
                for breakout_type in bullish_breakout_type_list:
                    if breakout_type is None:
                        bullish_breakout_list.append(None)
                    else:
                        bullish_breakout_list.append('long')

                df = df.assign(bullish_breakout=bullish_breakout_list)
            else:
                empty_list = [None]*len(dates)
                df = df.assign(bullish_breakout=empty_list)

        if short:
            highs_lows_dict = {}
            key_counter = 0
            for count in range(len(dates)):
                if not math.isnan(highs_list[count]) or not math.isnan(lows_list[count]):
                    highs_lows_dict[str(key_counter)] = [dates[count], close[count], volatility[count], -lows_list[count], -highs_list[count]]
                    key_counter += 1

            unverified_breakout_data = []  # format: [date, breakout price, type]

            for count in range(len(highs_lows_dict) - 5):  # head and shoulders
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count)][0]) < breakout_pattern_formation_expiration * 5:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4] > highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * head_and_shoulders_multiplier:  # checking 1 >> 3
                                                    if not math.isnan(highs_lows_dict[str(count + 4)][3]):  # checking 4 == high
                                                        if abs(highs_lows_dict[str(count + 2)][3] - highs_lows_dict[str(count + 4)][3]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * zone_multiplier:  # checking 2 == 4
                                                            if not math.isnan(highs_lows_dict[str(count + 5)][4]):  # checking that 5 == low
                                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 4)][3]:
                                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], -highs_lows_dict[str(count + 2)][3], 'head and shoulders'])
                                                                else:
                                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], -highs_lows_dict[str(count + 4)][3], 'head and shoulders'])

            for count in range(len(highs_lows_dict) - 3):  # double top
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 3)][0], highs_lows_dict[str(count)][0]) < breakout_pattern_formation_expiration * 3:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 3)][4]:  # checking 2 > 3
                                                    if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * zone_multiplier:  # checking 1 == 3
                                                        unverified_breakout_data.append([highs_lows_dict[str(count + 3)][0], -highs_lows_dict[str(count + 2)][3], 'double top'])

            for count in range(len(highs_lows_dict) - 5):  # triple top
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count)][0]) < breakout_pattern_formation_expiration * 5:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 3)][4]:  # checking 2 > 3
                                                    if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * zone_multiplier:  # checking 1 == 3
                                                        if not math.isnan(highs_lows_dict[str(count + 4)][3]):  # checking 4 == high
                                                            if abs(highs_lows_dict[str(count + 2)][3] - highs_lows_dict[str(count + 4)][3]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * zone_multiplier:  # checking 1 == 3
                                                                if not math.isnan(highs_lows_dict[str(count + 5)][4]):  # checking that 5 == low
                                                                    if abs(highs_lows_dict[str(count + 3)][4] - highs_lows_dict[str(count + 5)][4]) < highs_lows_dict[str(count + 5)][1] * highs_lows_dict[str(count + 5)][2] * zone_multiplier:  # checking 3 == 5
                                                                        if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 5)][4]) < highs_lows_dict[str(count + 5)][1] * highs_lows_dict[str(count + 5)][2] * zone_multiplier:  # checking 1 == 5
                                                                            if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 4)][3]:
                                                                                unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], -highs_lows_dict[str(count + 2)][3], 'triple top'])
                                                                            else:
                                                                                unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], -highs_lows_dict[str(count + 4)][3], 'triple top'])

            for count in range(len(highs_lows_dict) - 4):  # bearish rectangle
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 4)][0], highs_lows_dict[str(count)][0]) < breakout_pattern_formation_expiration * 4:
                    if not math.isnan(highs_lows_dict[str(count)][4]):  # checking that 0 == low
                        if not math.isnan(highs_lows_dict[str(count + 1)][3]):  # checking that 1 == high
                            if not math.isnan(highs_lows_dict[str(count + 2)][4]):  # checking that 2 == low
                                if not math.isnan(highs_lows_dict[str(count + 3)][3]):  # checking that 3 == high
                                    if abs(highs_lows_dict[str(count + 1)][3] - highs_lows_dict[str(count + 3)][3]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * zone_multiplier:  # checking 1 == 3
                                        if not math.isnan(highs_lows_dict[str(count + 4)][4]):  # checking that 4 == low
                                            if abs(highs_lows_dict[str(count + 2)][4] - highs_lows_dict[str(count + 4)][4]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * zone_multiplier:  # checking 2 == 4
                                                if highs_lows_dict[str(count + 1)][3] > highs_lows_dict[str(count + 3)][3]:
                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 4)][0], -highs_lows_dict[str(count + 1)][3], 'bearish rectangle'])
                                                else:
                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 4)][0], -highs_lows_dict[str(count + 3)][3], 'bearish rectangle'])

            if len(unverified_breakout_data) != 0:
                unverified_breakout_data.sort(key=lambda var: var[0])

                breakout_check_list_counter = 0
                breakout_data_list = []
                breakout_dates_list = []
                bug_fix_counter = 0
                for x in range(len(dates) - breakout_pattern_breakout_expiration):
                    if dates[x - bug_fix_counter] == unverified_breakout_data[breakout_check_list_counter][0]:
                        for date in dates[x - bug_fix_counter:x - bug_fix_counter + breakout_pattern_breakout_expiration]:
                            if close[dates.index(date)] < unverified_breakout_data[breakout_check_list_counter][1]:
                                breakout_data_list.append(unverified_breakout_data[breakout_check_list_counter])
                                breakout_dates_list.append(dates[dates.index(date)])
                                break
                        breakout_check_list_counter += 1
                        if breakout_check_list_counter == len(unverified_breakout_data):
                            break
                        if dates[x - bug_fix_counter] == unverified_breakout_data[breakout_check_list_counter][0]:
                            bug_fix_counter += 1

                bearish_breakout_type_list = []
                for counter, date in enumerate(dates):
                    bearish_breakout_type_list_appender = []
                    times = len([y for y in breakout_dates_list if y == date])
                    if times == 0:
                        bearish_breakout_type_list.append(None)
                    elif times > 0:
                        indices = [ind for ind, ele in enumerate(breakout_dates_list) if ele == date]
                        for ind in indices:
                            bearish_breakout_type_list_appender.append(breakout_data_list[ind][2])
                        bearish_breakout_type_list.append(bearish_breakout_type_list_appender)

                bearish_breakout_list = []
                for breakout_type in bearish_breakout_type_list:
                    if breakout_type is None:
                        bearish_breakout_list.append(None)
                    else:
                        bearish_breakout_list.append('short')

                df = df.assign(bearish_breakout=bearish_breakout_list)
            else:
                empty_list = [None] * len(dates)
                df = df.assign(bearish_breakout=empty_list)

        df.index = df_raw[date_or_datetime]
        export_csv.export_csv(stocks_csv_file_path + '/{}/breakouts/{}_breakouts.csv'.format(candle_length, ticker), df, 1)

    print('breakout_func complete')


def indentify_divergences_func():
    for ticker in tickers:
        df_rsi = pd.read_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), index_col=False, header=0)
        dates = df_rsi[date_or_datetime].tolist()

        df_volatility = pd.read_csv(stocks_csv_file_path + '/{}/volatility/{}_volatility.csv'.format(candle_length, ticker), index_col=False, header=0)
        volatility = df_volatility['volatility'].tolist()

        df = pd.DataFrame()

        if long:
            df_lows_price_1 = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/1/lows price 1/{}_lows price 1.csv'.format(candle_length, ticker), index_col=False, header=0)
            df_lows_rsi_1 = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/1/lows rsi 1/{}_lows rsi 1.csv'.format(candle_length, ticker), index_col=False, header=0)
            df_lows_price_2 = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/2/lows price 2/{}_lows price 2.csv'.format(candle_length, ticker), index_col=False, header=0)
            df_lows_rsi_2 = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/2/lows rsi 2/{}_lows rsi 2.csv'.format(candle_length, ticker), index_col=False, header=0)

            dropna_df_lows_price_1 = df_lows_price_1.dropna()
            dropna_df_lows_rsi_1 = df_lows_rsi_1.dropna()
            dropna_df_lows_price_2 = df_lows_price_2.dropna()
            dropna_df_lows_rsi_2 = df_lows_rsi_2.dropna()

            low_price_list_1 = dropna_df_lows_price_1['prices'].tolist()
            low_price_dates_list_1 = dropna_df_lows_price_1[date_or_datetime].tolist()
            low_rsi_list_1 = dropna_df_lows_rsi_1['rsis'].tolist()
            low_rsi_dates_list_1 = dropna_df_lows_rsi_1[date_or_datetime].tolist()
            low_price_list_2 = dropna_df_lows_price_2['prices'].tolist()
            low_price_dates_list_2 = dropna_df_lows_price_2[date_or_datetime].tolist()
            low_rsi_list_2 = dropna_df_lows_rsi_2['rsis'].tolist()
            low_rsi_dates_list_2 = dropna_df_lows_rsi_2[date_or_datetime].tolist()

            price_dict_1 = dict(zip(low_price_dates_list_1, low_price_list_1))
            price_dict_2 = dict(zip(low_price_dates_list_2, low_price_list_2))

            matching_low_prices_1 = []
            matching_low_rsis_1 = []
            matching_low_dates_1 = []
            matching_low_prices_2 = []
            matching_low_rsis_2 = []
            matching_low_dates_2 = []

            for x in low_price_dates_list_1:
                try:
                    matching_low_prices_1.append(price_dict_1[low_rsi_dates_list_1[low_rsi_dates_list_1.index(x)]])
                    matching_low_rsis_1.append(low_rsi_list_1[low_rsi_dates_list_1.index(x)])
                    matching_low_dates_1.append(low_rsi_dates_list_1[low_rsi_dates_list_1.index(x)])
                except ValueError:
                    pass
            for x in low_price_dates_list_2:
                try:
                    matching_low_prices_2.append(price_dict_2[low_rsi_dates_list_2[low_rsi_dates_list_2.index(x)]])
                    matching_low_rsis_2.append(low_rsi_list_2[low_rsi_dates_list_2.index(x)])
                    matching_low_dates_2.append(low_rsi_dates_list_2[low_rsi_dates_list_2.index(x)])
                except ValueError:
                    pass

            index_1 = []
            index_2 = []
            counter1 = 0

            for x, date2 in enumerate(matching_low_dates_2):
                try:
                    if date2 == matching_low_dates_1[counter1]:
                        index_1.append(x)
                        counter1 += 1
                except IndexError:
                    pass

            for x in range(len(matching_low_dates_2)):
                index_2.append(x)

            divergences = []
            divergences_list = []
            divergence_quality = []
            divergence_quality_list = []
            for x in range(len(matching_low_dates_2) - 1):
                time_difference = get_time_delta.get_time_delta(matching_low_dates_2[x + 1], matching_low_dates_2[x])
                if matching_low_prices_2[x] > matching_low_prices_2[x + 1] and matching_low_rsis_2[x] < matching_low_rsis_2[x + 1] and time_difference < divergence_expiration and index_2[x] in index_1 and not math.isnan(volatility[x + 1]):  # standard divergence
                    divergences.append(matching_low_dates_2[x + 1])
                    price_difference = ((matching_low_prices_2[x + 1] / matching_low_prices_2[x])-1)/volatility[x + 1]
                    rsi_difference = (matching_low_rsis_2[x + 1] - matching_low_rsis_2[x])
                    price_difference_quality = price_difference_quality_multiplier * -price_difference
                    rsi_difference_quality = rsi_difference_quality_multiplier * rsi_difference
                    rsi_level_quality = ((100-matching_low_rsis_2[x + 1]) + (100-matching_low_rsis_2[x])) * rsi_level_quality_multiplier
                    divergence_quality.append(price_difference_quality + rsi_difference_quality + rsi_level_quality)
                # hidden divergences
                # elif matching_low_prices_2[x] < matching_low_prices_2[x + 1] and matching_low_rsis_2[x] > matching_low_rsis_2[x + 1] and time_difference < divergence_expiration and index_2[x] in index_1:  # hidden divergence
                #     divergences.append(matching_low_dates_2[x + 1])
                #     price_slope = (matching_low_prices_2[x + 1] / matching_low_prices_2[x]-1)/time_difference
                #     rsi_slope = (matching_low_rsis_2[x + 1] - matching_low_rsis_2[x])/time_difference
                #     price_slope_quality = price_slope_quality_multiplier * price_slope
                #     rsi_slope_quality = rsi_slope_quality_multiplier * -rsi_slope
                #     rsi_level_quality = ((100-matching_low_rsis_2[x + 1]) + (100-matching_low_rsis_2[x])) * rsi_level_quality_multiplier
                #     divergence_quality.append(price_slope_quality + rsi_slope_quality + rsi_level_quality)

            counter = 0
            for day in dates:
                if day in divergences:
                    divergences_list.append('long')
                    divergence_quality_list.append(divergence_quality[counter])
                    counter += 1
                else:
                    divergences_list.append(None)
                    divergence_quality_list.append(None)

            df = df.assign(bullish_divergences=divergences_list)
            df = df.assign(bullish_quality=divergence_quality_list)

        if short:
            df_highs_price_1 = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/1/highs price 1/{}_highs price 1.csv'.format(candle_length, ticker), index_col=False, header=0)
            df_highs_rsi_1 = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/1/highs rsi 1/{}_highs rsi 1.csv'.format(candle_length, ticker), index_col=False, header=0)
            df_highs_price_2 = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/2/highs price 2/{}_highs price 2.csv'.format(candle_length, ticker), index_col=False, header=0)
            df_highs_rsi_2 = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/2/highs rsi 2/{}_highs rsi 2.csv'.format(candle_length, ticker), index_col=False, header=0)

            dropna_df_highs_price_1 = df_highs_price_1.dropna()
            dropna_df_highs_rsi_1 = df_highs_rsi_1.dropna()
            dropna_df_highs_price_2 = df_highs_price_2.dropna()
            dropna_df_highs_rsi_2 = df_highs_rsi_2.dropna()

            high_price_list_1 = dropna_df_highs_price_1['prices'].tolist()
            high_price_dates_list_1 = dropna_df_highs_price_1[date_or_datetime].tolist()
            high_rsi_list_1 = dropna_df_highs_rsi_1['rsis'].tolist()
            high_rsi_dates_list_1 = dropna_df_highs_rsi_1[date_or_datetime].tolist()
            high_price_list_2 = dropna_df_highs_price_2['prices'].tolist()
            high_price_dates_list_2 = dropna_df_highs_price_2[date_or_datetime].tolist()
            high_rsi_list_2 = dropna_df_highs_rsi_2['rsis'].tolist()
            high_rsi_dates_list_2 = dropna_df_highs_rsi_2[date_or_datetime].tolist()

            price_dict_1 = dict(zip(high_price_dates_list_1, high_price_list_1))
            price_dict_2 = dict(zip(high_price_dates_list_2, high_price_list_2))

            matching_high_prices_1 = []
            matching_high_rsis_1 = []
            matching_high_dates_1 = []
            matching_high_prices_2 = []
            matching_high_rsis_2 = []
            matching_high_dates_2 = []

            for x in high_price_dates_list_1:
                try:
                    matching_high_prices_1.append(price_dict_1[high_rsi_dates_list_1[high_rsi_dates_list_1.index(x)]])
                    matching_high_rsis_1.append(high_rsi_list_1[high_rsi_dates_list_1.index(x)])
                    matching_high_dates_1.append(high_rsi_dates_list_1[high_rsi_dates_list_1.index(x)])
                except ValueError:
                    pass
            for x in high_price_dates_list_2:
                try:
                    matching_high_prices_2.append(price_dict_2[high_rsi_dates_list_2[high_rsi_dates_list_2.index(x)]])
                    matching_high_rsis_2.append(high_rsi_list_2[high_rsi_dates_list_2.index(x)])
                    matching_high_dates_2.append(high_rsi_dates_list_2[high_rsi_dates_list_2.index(x)])
                except ValueError:
                    pass

            index_1 = []
            index_2 = []
            counter1 = 0

            for x, date2 in enumerate(matching_high_dates_2):
                try:
                    if date2 == matching_high_dates_1[counter1]:
                        index_1.append(x)
                        counter1 += 1
                except IndexError:
                    pass

            for x in range(len(matching_high_dates_2)):
                index_2.append(x)

            divergences = []
            divergences_list = []
            divergence_quality = []
            divergence_quality_list = []
            for x in range(len(matching_high_dates_2) - 1):
                time_difference = get_time_delta.get_time_delta(matching_high_dates_2[x + 1], matching_high_dates_2[x])
                if matching_high_prices_2[x] < matching_high_prices_2[x + 1] and matching_high_rsis_2[x] > matching_high_rsis_2[x + 1] and time_difference < divergence_expiration and index_2[x] in index_1 and not math.isnan(volatility[x + 1]):  # standard divergence
                    divergences.append(matching_high_dates_2[x + 1])
                    price_difference = ((matching_high_prices_2[x + 1] / matching_high_prices_2[x])-1)/volatility[x + 1]
                    rsi_difference = (matching_high_rsis_2[x + 1] - matching_high_rsis_2[x])
                    price_difference_quality = price_difference * price_difference_quality_multiplier
                    rsi_difference_quality = -rsi_difference * rsi_difference_quality_multiplier
                    rsi_level_quality = ((matching_high_rsis_2[x + 1]) + (matching_high_rsis_2[x])) * rsi_level_quality_multiplier
                    divergence_quality.append(price_difference_quality + rsi_difference_quality + rsi_level_quality)
            counter = 0
            for day in dates:
                if day in divergences:
                    divergences_list.append('short')
                    divergence_quality_list.append(divergence_quality[counter])
                    counter += 1
                else:
                    divergences_list.append(None)
                    divergence_quality_list.append(None)

            df = df.assign(bearish_divergences=divergences_list)
            df = df.assign(bearish_quality=divergence_quality_list)

        df.index = df_rsi[date_or_datetime]
        export_csv.export_csv(stocks_csv_file_path + '/{}/divergences/{}_divergences.csv'.format(candle_length, ticker), df, 1)

    print('identify_divergences_func complete')


def stochastic_crossover_func():
    for ticker in tickers:

        df = pd.DataFrame()

        if long:
            df_divergences = pd.read_csv(stocks_csv_file_path + '/{}/divergences/{}_divergences.csv'.format(candle_length, ticker), index_col=False, header=0)
            dates = df_divergences[date_or_datetime].tolist()
            divergences = df_divergences['bullish_divergences'].tolist()
            divergence_quality_list = df_divergences['bullish_quality'].tolist()

            df_stochastic = pd.read_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), index_col=False, header=0)
            k_list = df_stochastic['k'].tolist()
            d_list = df_stochastic['d'].tolist()

            crossover_list = []
            quality_list = []
            divergence_day = math.nan
            quality_day = math.nan
            if not flexible_stochastic_cross_level:
                for day in range(len(dates)):
                    quality = divergence_quality_list[day]
                    if divergences[day] == 'long' and k_list[day] < stochastic_maximum:
                        divergence_day = day
                        quality_day = quality
                    if divergence_day <= day:
                        if divergence_day + stochastic_cross_expiration > day:
                            if k_list[day] > d_list[day] and k_list[day] > stochastic_cross_level:
                                crossover_list.append('long')
                                quality_list.append(quality_day)
                                divergence_day = math.nan
                                quality_day = math.nan
                            else:
                                crossover_list.append(None)
                                quality_list.append(None)
                        else:
                            divergence_day = math.nan
                            quality_day = math.nan
                            crossover_list.append(None)
                            quality_list.append(None)
                    else:
                        crossover_list.append(None)
                        quality_list.append(None)
            else:
                inital_k = math.nan
                for day in range(len(dates)):
                    quality = divergence_quality_list[day]
                    if divergences[day] == 'long' and k_list[day] < stochastic_maximum:
                        divergence_day = day
                        quality_day = quality
                        inital_k = k_list[day]
                    if divergence_day <= day:
                        if divergence_day + stochastic_cross_expiration > day:
                            if k_list[day] > d_list[day] and k_list[day] > inital_k + stochastic_cross_level:
                                crossover_list.append('long')
                                quality_list.append(quality_day)
                                divergence_day = math.nan
                                quality_day = math.nan
                                inital_k = math.nan
                            else:
                                crossover_list.append(None)
                                quality_list.append(None)
                        else:
                            divergence_day = math.nan
                            quality_day = math.nan
                            inital_k = math.nan
                            crossover_list.append(None)
                            quality_list.append(None)
                    else:
                        crossover_list.append(None)
                        quality_list.append(None)

            df = df.assign(bullish_divergences=crossover_list)
            df = df.assign(bullish_quality=quality_list)

        if short:
            df_divergences = pd.read_csv(stocks_csv_file_path + '/{}/divergences/{}_divergences.csv'.format(candle_length, ticker), index_col=False, header=0)
            dates = df_divergences[date_or_datetime].tolist()
            divergences = df_divergences['bearish_divergences'].tolist()
            divergence_quality_list = df_divergences['bearish_quality'].tolist()

            df_stochastic = pd.read_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), index_col=False, header=0)
            k_list = df_stochastic['k'].tolist()
            d_list = df_stochastic['d'].tolist()

            crossover_list = []
            quality_list = []
            divergence_day = math.nan
            quality_day = math.nan
            if not flexible_stochastic_cross_level:
                for day in range(len(dates)):
                    quality = divergence_quality_list[day]
                    if divergences[day] == 'short' and k_list[day] > (100-stochastic_maximum):
                        divergence_day = day
                        quality_day = quality
                    if divergence_day <= day:
                        if divergence_day + stochastic_cross_expiration > day:
                            if k_list[day] < d_list[day] and k_list[day] < (100-stochastic_cross_level):
                                crossover_list.append('short')
                                quality_list.append(quality_day)
                                divergence_day = math.nan
                                quality_day = math.nan
                            else:
                                crossover_list.append(None)
                                quality_list.append(None)
                        else:
                            divergence_day = math.nan
                            quality_day = math.nan
                            crossover_list.append(None)
                            quality_list.append(None)
                    else:
                        crossover_list.append(None)
                        quality_list.append(None)
            else:
                inital_k = math.nan
                for day in range(len(dates)):
                    quality = divergence_quality_list[day]
                    if divergences[day] == 'short' and k_list[day] > (100-stochastic_maximum):
                        divergence_day = day
                        quality_day = quality
                        inital_k = k_list[day]
                    if divergence_day <= day:
                        if divergence_day + stochastic_cross_expiration > day:
                            if k_list[day] < d_list[day] and k_list[day] < inital_k - stochastic_cross_level:
                                crossover_list.append('short')
                                quality_list.append(quality_day)
                                divergence_day = math.nan
                                quality_day = math.nan
                                inital_k = math.nan
                            else:
                                crossover_list.append(None)
                                quality_list.append(None)
                        else:
                            divergence_day = math.nan
                            quality_day = math.nan
                            inital_k = math.nan
                            crossover_list.append(None)
                            quality_list.append(None)
                    else:
                        crossover_list.append(None)
                        quality_list.append(None)

            df = df.assign(bearish_divergences=crossover_list)
            df = df.assign(bearish_quality=quality_list)

        df.index = df_divergences[date_or_datetime]
        export_csv.export_csv(stocks_csv_file_path + '/{}/stochastic crossover divergence/{}_stochastic crossover divergence.csv'.format(candle_length, ticker), df, 1)

    print('stochastic_crossover_func complete')


def buy_signal_func():
    def flatten(container):
        for i2 in container:
            if isinstance(i2, (list, tuple)):
                for j in flatten(i2):
                    yield j
            else:
                yield i2

    buy_order_permutations = []
    product_list = []
    for unique_order in set(buy_signal_order_dict.values()):
        key_list = []
        for key, value in buy_signal_order_dict.items():
            if value == unique_order:
                key_list.append(key)
        product_list.append(permutations(key_list))
    products = product(*product_list)
    for perm in products:
        buy_order_permutations.append(list(flatten(perm)))

    # dates_list_compiled[-1]

    # holidays_url = 'https://www.nyse.com/markets/hours-calendars'
    # holidays_headers = {'User-Agent': 'Mozilla/5.0'}
    #
    # holidays_req = urllib.request.Request(holidays_url, headers=holidays_headers)
    # holidays_resp = urllib.request.urlopen(holidays_req)
    # holidays_resp_data = holidays_resp.read()
    #
    # holidays_days_list = re.findall(r'","text":"((\w{6,9}.\s\w{3,9}\s\d{1,2}.{0,3})|(\w{6,9}.\s\w{3,9}\s\d{1,2}\s[(]\w{12}\s\w{3}\s\w{8}[)]))"}', str(holidays_resp_data))
    # holidays_years_list = re.findall(r'All NYSE markets observe U.S. holidays as listed below for (.*?).</h4></div><div class="h-6"></div><div id="integration-id-a3eac97"', str(holidays_resp_data))
    #
    # holidays_days_list = [x[0].split(',')[1][1:].split(' ') for x in holidays_days_list]
    # for holiday in range(len(holidays_days_list)):
    #     print(holidays_days_list[holiday])
    #     if holidays_days_list[holiday][1][-3] == '*':
    #         holidays_days_list[holiday] = [holidays_days_list[holiday][0], holidays_days_list[holiday][1][:-3], 3]
    #         print(str(holidays_days_list[holiday]) + 'aa')
    #     elif holidays_days_list[holiday][1][-2] == '*':
    #         holidays_days_list[holiday] = [holidays_days_list[holiday][0], holidays_days_list[holiday][1][:-2], 2]
    #     elif holidays_days_list[holiday][1][-1] == '*':
    #         holidays_days_list[holiday] = [holidays_days_list[holiday][0], holidays_days_list[holiday][1][:-1], 1]
    #     else:
    #         holidays_days_list[holiday] = [holidays_days_list[holiday][0], holidays_days_list[holiday][1], 0]
    # holidays_years_list = holidays_years_list[0].split(',')
    # holidays_years_list = [holidays_years_list[0], holidays_years_list[1][1:], holidays_years_list[2][5:]]
    # print(holidays_days_list)
    # print(len(holidays_days_list))
    # print(holidays_years_list)
    #
    # holidays_datetimes_list = []
    # for num in range(len(holidays_days_list)):
    #     if len(holidays_days_list[num][1]) == 1:
    #         print(holidays_years_list[num % 3] + ' ' + holidays_days_list[num][0] + ' 0' + holidays_days_list[num][1])
    #         holidays_datetimes_list.append(holidays_years_list[num % 3] + ' ' + holidays_days_list[num][0] + ' 0' + holidays_days_list[num][1], '%Y %B %d')
    #     if len(holidays_days_list[num][1]) == 2:
    #         print(holidays_years_list[num % 3] + ' ' + holidays_days_list[num][0] + ' ' + holidays_days_list[num][1])
    #         holidays_datetimes_list.append(holidays_years_list[num % 3] + ' ' + holidays_days_list[num][0] + ' ' + holidays_days_list[num][1], '%Y %B %d')
    # print(holidays_datetimes_list)

    for ticker in tickers:
        df = pd.DataFrame()

        # ticker_obj = yf.Ticker(ticker)
        # earnings_df = ticker_obj.get_earnings_dates(limit=15)
        # earnings_dates_list0 = [str(x) for x in earnings_df.index]
        # earnings_dates_list0 = earnings_dates_list0[::-1]
        # earnings_dates_remove_list = []
        # for earnings_date in earnings_dates_list0:
        #     if earnings_date < dates_list_compiled[0]:
        #         earnings_dates_remove_list.append(earnings_date)
        # earnings_dates_list = []
        # for earnings_date in earnings_dates_list0:
        #     if earnings_date not in earnings_dates_remove_list:
        #         earnings_dates_list.append(earnings_date)

        if long:
            df_rsi = pd.read_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), index_col=False, header=0)
            rsi_quality_list = df_rsi['bullish_quality'].tolist()

            df_ema = pd.read_csv(stocks_csv_file_path + '/{}/ema/{}_ema.csv'.format(candle_length, ticker), index_col=False, header=0)
            ema_quality_list = df_ema['bullish_quality'].tolist()

            df_stochastic = pd.read_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), index_col=False, header=0)
            stochastic_quality_list = df_stochastic['bullish_quality'].tolist()

            if not stochastic_crossover:
                df_divergences = pd.read_csv(stocks_csv_file_path + '/{}/divergences/{}_divergences.csv'.format(candle_length, ticker), index_col=False, header=0)
                dates = df_divergences[date_or_datetime].tolist()
                divergences = df_divergences['bullish_divergences'].tolist()
                divergence_quality_list = df_divergences['bullish_quality'].tolist()
            else:
                df_divergences = pd.read_csv(stocks_csv_file_path + '/{}/stochastic crossover divergence/{}_stochastic crossover divergence.csv'.format(candle_length, ticker), index_col=False, header=0)
                dates = df_divergences[date_or_datetime].tolist()
                divergences = df_divergences['bullish_divergences'].tolist()
                divergence_quality_list = df_divergences['bullish_quality'].tolist()

            df_breakouts = pd.read_csv(stocks_csv_file_path + '/{}/breakouts/{}_breakouts.csv'.format(candle_length, ticker), index_col=False, header=0)
            breakouts = df_breakouts['bullish_breakout'].tolist()

            df_candle_stick = pd.read_csv(stocks_csv_file_path + '/{}/candle stick/{}_candle stick.csv'.format(candle_length, ticker), index_col=False, header=0)
            candle_sticks = df_candle_stick['bullish_candles'].tolist()

            buy_signal_list = []
            buy_signal_quality_list = []
            divergence_quality = 0
            bought = False
            quality = math.nan
            for day1 in range(len(dates)):
                # earnings_date_difference_dict = {}
                # for earnings_date2 in earnings_dates_list:
                #     earnings_date_difference = abs(datetime.strptime(earnings_date2[0:-6], '%Y-%m-%d %H:%M:%S') - datetime.strptime(dates[day1], '%Y-%m-%d %H:%M:%S'))
                #     earnings_date_difference_dict[earnings_date2[0:-6]] = earnings_date_difference.days + earnings_date_difference.seconds/60/60/24
                # earnings_date_difference_dict = dict(sorted(earnings_date_difference_dict.items(), key=lambda item: item[1]))
                # closest_earnings_date = list(earnings_date_difference_dict.keys())[0]
                #
                # for compiled_date in reversed(dates_list_compiled):
                #     if closest_earnings_date > compiled_date:
                #         closest_earnings_date = compiled_date
                #         break
                #
                omission = False
                # if pre_earnings_date_omission >= 0:
                #     for x in range(pre_earnings_date_omission+1):
                #         try:
                #             if closest_earnings_date == dates[day1+x]:
                #                 omission = True
                #         except IndexError:
                #             pass
                # if post_earnings_date_omission >= 0:
                #     for x in range(post_earnings_date_omission+1):
                #         if day1-x > 0:
                #             if closest_earnings_date == dates[day1-x]:
                #                 omission = True
                #         else:
                #             pass

                if bought and quality >= quality_minimum and not omission:
                    buy_signal_list.append('long')
                    buy_signal_quality_list.append(quality)
                else:
                    buy_signal_list.append(None)
                    buy_signal_quality_list.append(None)
                try:
                    bought_dict = {}
                    bought = False
                    for counter, buy_order_permutation in enumerate(buy_order_permutations):
                        bought_dict[counter] = True
                        buy_signals_dict = {}
                        reserve_days = 0
                        for count, buy_signal in reversed(list(enumerate(buy_order_permutation))):
                            buy_signals_dict[count] = False
                            if buy_signal == 'divergences':
                                buy_signal_data = divergences
                            if buy_signal == 'breakouts':
                                buy_signal_data = breakouts
                            if buy_signal == 'candle sticks':
                                buy_signal_data = candle_sticks
                            for day2 in range(buy_signal_expiration_list[count]+1):
                                buy_signal_expiration_modifier = sum(buy_signal_expiration_list[count+1:]) - reserve_days
                                if buy_signal_data[day1 - day2 - buy_signal_expiration_modifier] == 'long':
                                    reserve_days = buy_signal_expiration_list[count] - day2
                                    buy_signals_dict[count] = True
                                    if buy_signal == 'divergences':
                                        divergence_quality = divergence_quality_list[day1 - day2 - buy_signal_expiration_modifier]
                                    break
                            if not buy_signals_dict.get(count):
                                break

                        for val in buy_signals_dict.values():
                            if not val:
                                bought_dict[counter] = False

                    for val in bought_dict.values():
                        if val:
                            bought = True

                    quality = rsi_quality_list[day1] + ema_quality_list[day1] + stochastic_quality_list[day1] + divergence_quality

                except IndexError:
                    print(day1, ticker)
                    print('ERROR')

            df = df.assign(bullish_buy_signal=buy_signal_list)
            df = df.assign(bullish_quality=buy_signal_quality_list)

        if short:
            df_rsi = pd.read_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), index_col=False, header=0)
            rsi_quality_list = df_rsi['bearish_quality'].tolist()

            df_ema = pd.read_csv(stocks_csv_file_path + '/{}/ema/{}_ema.csv'.format(candle_length, ticker), index_col=False, header=0)
            ema_quality_list = df_ema['bearish_quality'].tolist()

            df_stochastic = pd.read_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), index_col=False, header=0)
            stochastic_quality_list = df_stochastic['bearish_quality'].tolist()

            if not stochastic_crossover:
                df_divergences = pd.read_csv(stocks_csv_file_path + '/{}/divergences/{}_divergences.csv'.format(candle_length, ticker), index_col=False, header=0)
                dates = df_divergences[date_or_datetime].tolist()
                divergences = df_divergences['bearish_divergences'].tolist()
                divergence_quality_list = df_divergences['bearish_quality'].tolist()
            else:
                df_divergences = pd.read_csv(stocks_csv_file_path + '/{}/stochastic crossover divergence/{}_stochastic crossover divergence.csv'.format(candle_length, ticker), index_col=False, header=0)
                dates = df_divergences[date_or_datetime].tolist()
                divergences = df_divergences['bearish_divergences'].tolist()
                divergence_quality_list = df_divergences['bearish_quality'].tolist()

            df_breakouts = pd.read_csv(stocks_csv_file_path + '/{}/breakouts/{}_breakouts.csv'.format(candle_length, ticker), index_col=False, header=0)
            breakouts = df_breakouts['bearish_breakout'].tolist()

            df_candle_stick = pd.read_csv(stocks_csv_file_path + '/{}/candle stick/{}_candle stick.csv'.format(candle_length, ticker), index_col=False, header=0)
            candle_sticks = df_candle_stick['bearish_candles'].tolist()

            buy_signal_list = []
            buy_signal_quality_list = []
            divergence_quality = 0
            bought = False
            quality = math.nan
            for day1 in range(len(dates)):
                # earnings_date_difference_dict = {}
                # for earnings_date2 in earnings_dates_list:
                #     earnings_date_difference = abs(datetime.strptime(earnings_date2[0:-6], '%Y-%m-%d %H:%M:%S') - datetime.strptime(dates[day1], '%Y-%m-%d %H:%M:%S'))
                #     earnings_date_difference_dict[earnings_date2[0:-6]] = earnings_date_difference.days + earnings_date_difference.seconds/60/60/24
                # earnings_date_difference_dict = dict(sorted(earnings_date_difference_dict.items(), key=lambda item: item[1]))
                # closest_earnings_date = list(earnings_date_difference_dict.keys())[0]
                #
                # for compiled_date in reversed(dates_list_compiled):
                #     if closest_earnings_date > compiled_date:
                #         closest_earnings_date = compiled_date
                #         break
                #
                omission = False
                # if pre_earnings_date_omission >= 0:
                #     for x in range(pre_earnings_date_omission+1):
                #         try:
                #             if closest_earnings_date == dates[day1+x]:
                #                 omission = True
                #         except IndexError:
                #             pass
                # if post_earnings_date_omission >= 0:
                #     for x in range(post_earnings_date_omission+1):
                #         if day1-x > 0:
                #             if closest_earnings_date == dates[day1-x]:
                #                 omission = True
                #         else:
                #             pass

                if bought and quality >= quality_minimum and not omission:
                    buy_signal_list.append('short')
                    buy_signal_quality_list.append(quality)
                else:
                    buy_signal_list.append(None)
                    buy_signal_quality_list.append(None)
                try:
                    bought_dict = {}
                    bought = False
                    for counter, buy_order_permutation in enumerate(buy_order_permutations):
                        bought_dict[counter] = True
                        buy_signals_dict = {}
                        reserve_days = 0
                        for count, buy_signal in reversed(list(enumerate(buy_order_permutation))):
                            buy_signals_dict[count] = False
                            if buy_signal == 'divergences':
                                buy_signal_data = divergences
                            if buy_signal == 'breakouts':
                                buy_signal_data = breakouts
                            if buy_signal == 'candle sticks':
                                buy_signal_data = candle_sticks
                            for day2 in range(buy_signal_expiration_list[count]+1):
                                buy_signal_expiration_modifier = sum(buy_signal_expiration_list[count+1:]) - reserve_days
                                if buy_signal_data[day1 - day2 - buy_signal_expiration_modifier] == 'short':
                                    reserve_days = buy_signal_expiration_list[count] - day2
                                    buy_signals_dict[count] = True
                                    if buy_signal == 'divergences':
                                        divergence_quality = divergence_quality_list[day1 - day2 - buy_signal_expiration_modifier]
                                    break
                            if not buy_signals_dict.get(count):
                                break

                        for val in buy_signals_dict.values():
                            if not val:
                                bought_dict[counter] = False

                    for val in bought_dict.values():
                        if val:
                            bought = True

                    quality = rsi_quality_list[day1] + ema_quality_list[day1] + stochastic_quality_list[day1] + divergence_quality

                except IndexError:
                    print(day1, ticker)
                    print('ERROR')

            df = df.assign(bearish_buy_signal=buy_signal_list)
            df = df.assign(bearish_quality=buy_signal_quality_list)

        df.index = df_divergences[date_or_datetime]
        export_csv.export_csv(stocks_csv_file_path + '/{}/buy signals/{}_buy signals.csv'.format(candle_length, ticker), df, 1)

    print('buy_signal_func complete')


def sell_signal_func():
    for ticker in tickers:
        df = pd.DataFrame()

        if long:
            sup_res_df = pd.read_csv(stocks_csv_file_path + '/{}/supports and resistances/{}_supports and resistances.csv'.format(candle_length, ticker), index_col=False, header=0)
            sup_res_raw = sup_res_df['supports and resistances'].tolist()

            df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)
            dates = df_raw[date_or_datetime].tolist()
            high_list = df_raw['high'].tolist()
            close_list = df_raw['close'].tolist()

            df_buy_signals = pd.read_csv(stocks_csv_file_path + '/{}/buy signals/{}_buy signals.csv'.format(candle_length, ticker), index_col=False, header=0)
            buy_signals = df_buy_signals['bullish_buy_signal'].tolist()

            if sell_signal_indicator_type_1 == 'rsi':
                df_rsi = pd.read_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), index_col=False, header=0)
                sell_signal_indicator_list = df_rsi[str(rsi_length)].tolist()
            elif sell_signal_indicator_type_1 == 'k':
                df_stochastic = pd.read_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), index_col=False, header=0)
                sell_signal_indicator_list = df_stochastic['k'].tolist()
            elif sell_signal_indicator_type_1 == 'd':
                df_stochastic = pd.read_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), index_col=False, header=0)
                sell_signal_indicator_list = df_stochastic['d'].tolist()

            sup_res = []
            for x in sup_res_raw:
                sup_res_appender = x.split('_')
                sup_res_appender.remove('')
                sup_res.append(sup_res_appender)

            sell_signals_dict = {}
            for x, sell_signals in enumerate(sell_signals_nested_list):
                sell_signals_dict[x] = {}
                for y in range(len(sell_signals)):
                    sell_signals_dict[x][y] = [False, None, math.nan]

            sell_signal_indicator_bought_1 = False
            sell_signal_indicator_buy_value_1 = math.nan
            artificial_margin_take_profit_price_1 = math.nan
            artificial_margin_stop_loss_price_1 = math.nan
            support_resistance_resistance_price_1 = math.nan
            support_resistance_support_price_1 = math.nan
            sell_time_buy_date_1 = math.nan
            sell_signal_list = []
            sell_price_list = []
            for day in range(len(dates)):
                sold = False
                for x, sell_signals in enumerate(sell_signals_nested_list):
                    for y, sell_signal in enumerate(sell_signals):
                        sell_signals_dict[x][y] = [False, None, math.nan]  # sell signal, sell price, sell day

                        if sell_signal == 'sell signal indicator 1':
                            sell_signal_active_today_1 = False
                            if not sell_signal_indicator_flexible_1:
                                if buy_signals[day] == 'long':
                                    sell_signal_indicator_bought_1 = True
                                if sell_signal_indicator_bought_1 and sell_signal_indicator_list[day] >= sell_signal_indicator_value_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif sell_signal_simultaneous_fulfillment_1:
                                    sell_signal_active_today_1 = True
                            else:
                                if buy_signals[day] == 'long':
                                    sell_signal_indicator_buy_value_1 = sell_signal_indicator_list[day]
                                if sell_signal_indicator_list[day] - sell_signal_indicator_buy_value_1 >= sell_signal_indicator_value_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif sell_signal_simultaneous_fulfillment_1:
                                    sell_signal_active_today_1 = True
                            if sell_signal_simultaneous_fulfillment_1 and not sell_signal_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'support resistance 1':
                            support_resistance_active_today_1 = False
                            if buy_signals[day] == 'long':
                                for sr in sup_res[day]:
                                    if float(sr) > close_list[day] * support_resistance_resistance_minimum_distance_1:
                                        support_resistance_resistance_price_1 = float(sr)
                                        break
                                for sr in reversed(sup_res[day]):
                                    if float(sr) < close_list[day] * support_resistance_support_minimum_distance_1:
                                        support_resistance_support_price_1 = float(sr)
                                        break
                            if not support_resistance_resistance_high_1:
                                if close_list[day] >= support_resistance_resistance_price_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            else:
                                if high_list[day] >= support_resistance_resistance_price_1:
                                    sell_signals_dict[x][y] = [True, str(support_resistance_resistance_price_1) + ' nc', day]
                                elif support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            if close_list[day] <= support_resistance_support_price_1:
                                sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                            elif support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            if support_resistance_simultaneous_fulfillment_1 and not support_resistance_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'artificial margin 1':
                            artificial_margin_active_today_1 = False
                            if buy_signals[day] == 'long':
                                artificial_margin_take_profit_price_1 = close_list[day] * artificial_margin_take_profit_1
                                artificial_margin_stop_loss_price_1 = close_list[day] * artificial_margin_stop_loss_1
                            if not artificial_margin_take_profit_high_1:
                                if close_list[day] >= artificial_margin_take_profit_price_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif artificial_margin_simultaneous_fulfillment_1:
                                        artificial_margin_active_today_1 = True
                            else:
                                if high_list[day] >= artificial_margin_take_profit_price_1:
                                    sell_signals_dict[x][y] = [True, str(artificial_margin_take_profit_price_1) + ' nc', day]
                                elif artificial_margin_simultaneous_fulfillment_1:
                                        artificial_margin_active_today_1 = True
                            if close_list[day] <= artificial_margin_stop_loss_price_1:
                                sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                            elif artificial_margin_simultaneous_fulfillment_1:
                                    artificial_margin_active_today_1 = True
                            if artificial_margin_simultaneous_fulfillment_1 and not artificial_margin_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'sell time 1':
                            if buy_signals[day] == 'long':
                                sell_time_buy_date_1 = day
                            if day - sell_time_buy_date_1 >= sell_time_value_1:
                                sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]

                cl_sell_price_list = []
                nc_sell_price_list = []
                for x, sell_signals in enumerate(sell_signals_dict.values()):
                    sell_price_dict = {}
                    sold_check_list = []
                    for y in range(len(sell_signals)):
                        if not ((sell_signals_dict.get(x)).get(y))[0]:
                            sold_check_list.append(False)
                        else:
                            sell_price_dict[str(((sell_signals_dict.get(x)).get(y))[2]) + ' ' + str(x) + ' ' + str(y)] = ((sell_signals_dict.get(x)).get(y))[1]  # will break if x or y is two digits
                    if len(sold_check_list) == 0:
                        sold = True
                        sell_price_dict = dict(sorted(sell_price_dict.items()))
                        key_to_use = 0
                        sell_price_keys_list = []
                        sell_price_vals_list = []
                        for key, val in sell_price_dict.items():
                            sell_price_keys_list.append(int(key[:-4]))
                            sell_price_vals_list.append(val)
                            if int(key[:-4]) > key_to_use:
                                key_to_use = int(key[:-4])
                        key_count = [k for k in sell_price_dict.keys() if int(k[:-4]) == key_to_use]
                        if len(key_count) == 1:
                            if sell_price_vals_list[sell_price_keys_list.index(key_to_use)][-2:] == 'cl':
                                cl_sell_price_list.append(float(sell_price_vals_list[sell_price_keys_list.index(key_to_use)][:-3]))  # appending the sell price for a group of sell signals that sold at a close value
                            elif sell_price_vals_list[sell_price_keys_list.index(key_to_use)][-2:] == 'nc':
                                nc_sell_price_list.append(float(sell_price_vals_list[sell_price_keys_list.index(key_to_use)][:-3]))  # appending the sell price for a group of sell signals that sold at a non-close value
                        elif len(key_count) > 1:
                            possible_keys_list = []
                            for count in range(len(key_count)):
                                sell_price_list_index = sell_price_keys_list.index(key_to_use)
                                possible_keys_list.append(sell_price_vals_list[sell_price_list_index])
                                sell_price_keys_list.pop(sell_price_list_index)
                                sell_price_vals_list.pop(sell_price_list_index)
                            cl_list = []
                            nc_list = []
                            for possible_key in possible_keys_list:
                                if possible_key[-2:] == 'cl':
                                    cl_list.append(float(possible_key[0:-3]))
                                if possible_key[-2:] == 'nc':
                                    nc_list.append(float(possible_key[0:-3]))
                            if len(nc_list) > 0:
                                nc_list.sort()
                                nc_sell_price_list.append(nc_list[0])  # appending the sell price for a group of sell signals that sold at a non-close value
                            else:
                                cl_sell_price_list.append(cl_list[0])  # appending the sell price for a group of sell signals that sold at a close value
                                if cl_list[0] != cl_list[-1]:
                                    print('ERROR')
                        else:
                            print('ERROR')

                if sold:
                    if len(nc_sell_price_list) > 0:
                        nc_sell_price_list.sort()
                        sell_price = nc_sell_price_list[0]
                    else:
                        sell_price = cl_sell_price_list[0]
                        if cl_sell_price_list[0] != cl_sell_price_list[-1]:
                            print('ERROR')

                if day == len(dates)-1:
                    sold = True
                    sell_price = close_list[day]

                if sold:
                    sell_signal_list.append('long')
                    sell_price_list.append(sell_price)
                    sell_signal_indicator_bought_1 = False
                    sell_signal_indicator_buy_value_1 = math.nan
                    artificial_margin_take_profit_price_1 = math.nan
                    artificial_margin_stop_loss_price_1 = math.nan
                    support_resistance_resistance_price_1 = math.nan
                    support_resistance_support_price_1 = math.nan
                    sell_time_buy_date_1 = math.nan
                else:
                    sell_signal_list.append(None)
                    sell_price_list.append(None)

            df = df.assign(bullish_sell_signal=sell_signal_list)
            df = df.assign(bullish_sell_price=sell_price_list)

        if short:
            sup_res_df = pd.read_csv(stocks_csv_file_path + '/{}/supports and resistances/{}_supports and resistances.csv'.format(candle_length, ticker), index_col=False, header=0)
            sup_res_raw = sup_res_df['supports and resistances'].tolist()

            df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)
            dates = df_raw[date_or_datetime].tolist()
            low_list = df_raw['low'].tolist()
            close_list = df_raw['close'].tolist()

            df_buy_signals = pd.read_csv(stocks_csv_file_path + '/{}/buy signals/{}_buy signals.csv'.format(candle_length, ticker), index_col=False, header=0)
            buy_signals = df_buy_signals['bearish_buy_signal'].tolist()

            if sell_signal_indicator_type_1 == 'rsi':
                df_rsi = pd.read_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), index_col=False, header=0)
                sell_signal_indicator_list = df_rsi[str(rsi_length)].tolist()
            elif sell_signal_indicator_type_1 == 'k':
                df_stochastic = pd.read_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), index_col=False, header=0)
                sell_signal_indicator_list = df_stochastic['k'].tolist()
            elif sell_signal_indicator_type_1 == 'd':
                df_stochastic = pd.read_csv(stocks_csv_file_path + '/{}/stochastic/{}_stochastic.csv'.format(candle_length, ticker), index_col=False, header=0)
                sell_signal_indicator_list = df_stochastic['d'].tolist()

            sup_res = []
            for x in sup_res_raw:
                sup_res_appender = x.split('_')
                sup_res_appender.remove('')
                sup_res.append(sup_res_appender)

            sell_signals_dict = {}
            for x, sell_signals in enumerate(sell_signals_nested_list):
                sell_signals_dict[x] = {}
                for y in range(len(sell_signals)):
                    sell_signals_dict[x][y] = [False, None, math.nan]

            sell_signal_indicator_bought_1 = False
            sell_signal_indicator_buy_value_1 = math.nan
            artificial_margin_take_profit_price_1 = math.nan
            artificial_margin_stop_loss_price_1 = math.nan
            support_resistance_resistance_price_1 = math.nan
            support_resistance_support_price_1 = math.nan
            sell_time_buy_date_1 = math.nan
            sell_signal_list = []
            sell_price_list = []
            for day in range(len(dates)):
                sold = False
                for x, sell_signals in enumerate(sell_signals_nested_list):
                    for y, sell_signal in enumerate(sell_signals):
                        sell_signals_dict[x][y] = [False, None, math.nan]  # sell signal, sell price, sell day

                        if sell_signal == 'sell signal indicator 1':
                            sell_signal_active_today_1 = False
                            if not sell_signal_indicator_flexible_1:
                                if buy_signals[day] == 'short':
                                    sell_signal_indicator_bought_1 = True
                                if sell_signal_indicator_bought_1 and sell_signal_indicator_list[day] <= (100 - sell_signal_indicator_value_1):
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif sell_signal_simultaneous_fulfillment_1:
                                    sell_signal_active_today_1 = True
                            else:
                                if buy_signals[day] == 'short':
                                    sell_signal_indicator_buy_value_1 = sell_signal_indicator_list[day]
                                if sell_signal_indicator_list[day] - sell_signal_indicator_buy_value_1 <= (100 - sell_signal_indicator_value_1):
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif sell_signal_simultaneous_fulfillment_1:
                                    sell_signal_active_today_1 = True
                            if sell_signal_simultaneous_fulfillment_1 and not sell_signal_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'support resistance 1':
                            support_resistance_active_today_1 = False
                            if buy_signals[day] == 'short':
                                for sr in sup_res[day]:
                                    if float(sr) > close_list[day] * (1/support_resistance_support_minimum_distance_1):
                                        support_resistance_resistance_price_1 = float(sr)
                                        break
                                for sr in reversed(sup_res[day]):
                                    if float(sr) < close_list[day] * (1/support_resistance_resistance_minimum_distance_1):
                                        support_resistance_support_price_1 = float(sr)
                                        break
                            if not support_resistance_resistance_high_1:
                                if close_list[day] <= support_resistance_support_price_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            else:
                                if low_list[day] <= support_resistance_support_price_1:
                                    sell_signals_dict[x][y] = [True, str(support_resistance_support_price_1) + ' nc', day]
                                elif support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            if close_list[day] >= support_resistance_resistance_price_1:
                                sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                            elif support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            if support_resistance_simultaneous_fulfillment_1 and not support_resistance_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'artificial margin 1':
                            artificial_margin_active_today_1 = False
                            if buy_signals[day] == 'short':
                                artificial_margin_take_profit_price_1 = close_list[day] * (1/artificial_margin_take_profit_1)
                                artificial_margin_stop_loss_price_1 = close_list[day] * (1/artificial_margin_stop_loss_1)

                            if not artificial_margin_take_profit_high_1:
                                if close_list[day] <= artificial_margin_take_profit_price_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif artificial_margin_simultaneous_fulfillment_1:
                                        artificial_margin_active_today_1 = True
                            else:
                                if low_list[day] <= artificial_margin_take_profit_price_1:
                                    sell_signals_dict[x][y] = [True, str(artificial_margin_take_profit_price_1) + ' nc', day]
                                elif artificial_margin_simultaneous_fulfillment_1:
                                    artificial_margin_active_today_1 = True
                            if close_list[day] >= artificial_margin_stop_loss_price_1:
                                sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                            elif artificial_margin_simultaneous_fulfillment_1:
                                artificial_margin_active_today_1 = True
                            if artificial_margin_simultaneous_fulfillment_1 and not artificial_margin_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'sell time 1':
                            if buy_signals[day] == 'short':
                                sell_time_buy_date_1 = day
                            if day - sell_time_buy_date_1 >= sell_time_value_1:
                                sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]

                cl_sell_price_list = []
                nc_sell_price_list = []
                for x, sell_signals in enumerate(sell_signals_dict.values()):
                    sell_price_dict = {}
                    sold_check_list = []
                    for y in range(len(sell_signals)):
                        if not ((sell_signals_dict.get(x)).get(y))[0]:
                            sold_check_list.append(False)
                        else:
                            sell_price_dict[str(((sell_signals_dict.get(x)).get(y))[2]) + ' ' + str(x) + ' ' + str(y)] = ((sell_signals_dict.get(x)).get(y))[1]  # will break if x or y is two digits
                    if len(sold_check_list) == 0:
                        sold = True
                        sell_price_dict = dict(sorted(sell_price_dict.items()))
                        key_to_use = 0
                        sell_price_keys_list = []
                        sell_price_vals_list = []
                        for key, val in sell_price_dict.items():
                            sell_price_keys_list.append(int(key[:-4]))
                            sell_price_vals_list.append(val)
                            if int(key[:-4]) > key_to_use:
                                key_to_use = int(key[:-4])
                        key_count = [k for k in sell_price_dict.keys() if int(k[:-4]) == key_to_use]
                        if len(key_count) == 1:
                            if sell_price_vals_list[sell_price_keys_list.index(key_to_use)][-2:] == 'cl':
                                cl_sell_price_list.append(float(sell_price_vals_list[sell_price_keys_list.index(key_to_use)][:-3]))  # appending the sell price for a group of sell signals that sold at a close value
                            elif sell_price_vals_list[sell_price_keys_list.index(key_to_use)][-2:] == 'nc':
                                nc_sell_price_list.append(float(sell_price_vals_list[sell_price_keys_list.index(key_to_use)][:-3]))  # appending the sell price for a group of sell signals that sold at a non-close value
                        elif len(key_count) > 1:
                            possible_keys_list = []
                            for count in range(len(key_count)):
                                sell_price_list_index = sell_price_keys_list.index(key_to_use)
                                possible_keys_list.append(sell_price_vals_list[sell_price_list_index])
                                sell_price_keys_list.pop(sell_price_list_index)
                                sell_price_vals_list.pop(sell_price_list_index)
                            cl_list = []
                            nc_list = []
                            for possible_key in possible_keys_list:
                                if possible_key[-2:] == 'cl':
                                    cl_list.append(float(possible_key[0:-3]))
                                if possible_key[-2:] == 'nc':
                                    nc_list.append(float(possible_key[0:-3]))
                            if len(nc_list) > 0:
                                nc_list.sort()
                                nc_sell_price_list.append(nc_list[-1])  # appending the sell price for a group of sell signals that sold at a non-close value
                            else:
                                cl_sell_price_list.append(cl_list[0])  # appending the sell price for a group of sell signals that sold at a close value
                                if cl_list[0] != cl_list[-1]:
                                    print('ERROR')
                        else:
                            print('ERROR')

                if sold:
                    if len(nc_sell_price_list) > 0:
                        nc_sell_price_list.sort()
                        sell_price = nc_sell_price_list[-1]
                    else:
                        sell_price = cl_sell_price_list[0]
                        if cl_sell_price_list[0] != cl_sell_price_list[-1]:
                            print('ERROR')

                if day == len(dates) - 1:
                    sold = True
                    sell_price = close_list[day]

                if sold:
                    sell_signal_list.append('short')
                    sell_price_list.append(sell_price)
                    sell_signal_indicator_bought_1 = False
                    sell_signal_indicator_buy_value_1 = math.nan
                    artificial_margin_take_profit_price_1 = math.nan
                    artificial_margin_stop_loss_price_1 = math.nan
                    support_resistance_resistance_price_1 = math.nan
                    support_resistance_support_price_1 = math.nan
                    sell_time_buy_date_1 = math.nan
                else:
                    sell_signal_list.append(None)
                    sell_price_list.append(None)

            df = df.assign(bearish_sell_signal=sell_signal_list)
            df = df.assign(bearish_sell_price=sell_price_list)

        df.index = df_raw[date_or_datetime]
        export_csv.export_csv(stocks_csv_file_path + '/{}/sell signals/{}_sell signals.csv'.format(candle_length, ticker), df, 1)
    print('sell_signal_func complete')


def back_test_func():
    benchmark_list = []

    def individual_stock_order_list_func():
        for ticker in tickers:
            # 1d
            df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)
            open_list = df_raw['open'].tolist()
            del open_list[0:longest_indicator_length]
            close_list = df_raw['close'].tolist()
            del close_list[0:longest_indicator_length]
            date_list = df_raw[date_or_datetime].tolist()
            del date_list[0:longest_indicator_length]

            df_buy = pd.read_csv(stocks_csv_file_path + '/{}/buy signals/{}_buy signals.csv'.format(candle_length, ticker), index_col=False, header=0)
            df_sell = pd.read_csv(stocks_csv_file_path + '/{}/sell signals/{}_sell signals.csv'.format(candle_length, ticker), index_col=False, header=0)

            if long:
                # buy
                bullish_buy_signal = df_buy['bullish_buy_signal'].tolist()
                del bullish_buy_signal[0:longest_indicator_length]
                bullish_quality = df_buy['bullish_quality'].tolist()
                del bullish_quality[0:longest_indicator_length]
                # sell
                bullish_sell_signal = df_sell['bullish_sell_signal'].tolist()
                del bullish_sell_signal[0:longest_indicator_length]
                bullish_sell_price = df_sell['bullish_sell_price'].tolist()
                del bullish_sell_price[0:longest_indicator_length]
            if short:
                # buy
                bearish_buy_signal = df_buy['bearish_buy_signal'].tolist()
                del bearish_buy_signal[0:longest_indicator_length]
                bearish_quality = df_buy['bearish_quality'].tolist()
                del bearish_quality[0:longest_indicator_length]
                # sell
                bearish_sell_signal = df_sell['bearish_sell_signal'].tolist()
                del bearish_sell_signal[0:longest_indicator_length]
                bearish_sell_price = df_sell['bearish_sell_price'].tolist()
                del bearish_sell_price[0:longest_indicator_length]

            order_list = []
            last_order = None
            ignore_sold = None
            for day in range(len(date_list)):
                if day != len(date_list) - 1:
                    if long and short:
                        if bullish_buy_signal[day] == 'long':  # buying long
                            if bullish_sell_signal[day] != 'long':
                                if last_order is None:
                                    order_list.append([date_list[day], 'long bought', open_list[day], bullish_quality[day]])
                                    last_order = 'long bought'
                                elif last_order == 'long bought':
                                    order_list.append([date_list[day], 'long hold', None, None])
                                    last_order = 'long bought'
                                elif last_order == 'short bought':
                                    order_list.append([date_list[day], 'short sold and long bought', [open_list[day], open_list[day]], bullish_quality[day]])
                                    last_order = 'long bought'
                                    ignore_sold = 'short'
                            else:
                                if last_order == '':
                                    order_list.append([date_list[day], 'long bought and long sold', [open_list[day], bullish_sell_price[day]], bullish_quality[day]])
                                    last_order = None
                                elif last_order == 'long bought':
                                    order_list.append([date_list[day], 'long sold', bullish_sell_price[day], None])
                                    last_order = None
                                elif last_order == 'short bought':
                                    order_list.append([date_list[day], 'short sold and long bought and long sold', [open_list[day], open_list[day], bullish_sell_price[day]], bullish_quality[day]])
                                    last_order = None
                                    ignore_sold = 'short'
                        elif bullish_sell_signal[day] == 'long':  # selling long
                            if ignore_sold != 'long':
                                order_list.append([date_list[day], 'long sold', bullish_sell_price[day], None])
                                last_order = None
                            else:
                                order_list.append([date_list[day], None, None, None])
                                ignore_sold = None
                        elif bearish_buy_signal[day] == 'short':  # buying short
                            if bearish_sell_signal[day] != 'short':
                                if last_order is None:
                                    order_list.append([date_list[day], 'short bought', open_list[day], bearish_quality[day]])
                                    last_order = 'short bought'
                                elif last_order == 'short bought':
                                    order_list.append([date_list[day], 'short hold', None, None])
                                    last_order = 'short bought'
                                elif last_order == 'long bought':
                                    order_list.append([date_list[day], 'long sold and short bought', [open_list[day], open_list[day]], bearish_quality[day]])
                                    last_order = 'short bought'
                                    ignore_sold = 'long'
                            else:
                                if last_order == '':
                                    order_list.append([date_list[day], 'short bought and short sold', [open_list[day], bearish_sell_price[day]], bearish_quality[day]])
                                    last_order = None
                                elif last_order == 'short bought':
                                    order_list.append([date_list[day], 'short sold', bearish_sell_price[day], None])
                                    last_order = None
                                elif last_order == 'long bought':
                                    order_list.append([date_list[day], 'long sold and short bought and short sold', [open_list[day], open_list[day], bearish_sell_price[day]], bearish_quality[day]])
                                    last_order = None
                                    ignore_sold = 'long'
                        elif bearish_sell_signal[day] == 'short':  # selling short
                            if ignore_sold != 'short':
                                order_list.append([date_list[day], 'short sold', bearish_sell_price[day], None])
                                last_order = None
                            else:
                                order_list.append([date_list[day], None, None, None])
                                ignore_sold = None
                        else:
                            if last_order == 'long bought':
                                order_list.append([date_list[day], 'long hold', None, None])
                                last_order = 'long bought'
                            elif last_order == 'short bought':
                                order_list.append([date_list[day], 'short hold', None, None])
                                last_order = 'short bought'
                            else:
                                order_list.append([date_list[day], None, None, None])
                                last_order = None
                    elif long:
                        if bullish_buy_signal[day] == 'long':  # buying long
                            if bullish_sell_signal[day] != 'long':
                                if last_order is None:
                                    order_list.append(
                                        [date_list[day], 'long bought', open_list[day], bullish_quality[day]])
                                    last_order = 'long bought'
                                elif last_order == 'long bought':
                                    order_list.append([date_list[day], 'long hold', None, None])
                                    last_order = 'long bought'
                                elif last_order == 'short bought':
                                    order_list.append(
                                        [date_list[day], 'short sold and long bought', [open_list[day], open_list[day]],
                                         bullish_quality[day]])
                                    last_order = 'long bought'
                                    ignore_sold = 'short'
                            else:
                                if last_order == '':
                                    order_list.append([date_list[day], 'long bought and long sold',
                                                       [open_list[day], bullish_sell_price[day]], bullish_quality[day]])
                                    last_order = None
                                elif last_order == 'long bought':
                                    order_list.append([date_list[day], 'long sold', bullish_sell_price[day], None])
                                    last_order = None
                                elif last_order == 'short bought':
                                    order_list.append([date_list[day], 'short sold and long bought and long sold',
                                                       [open_list[day], open_list[day], bullish_sell_price[day]],
                                                       bullish_quality[day]])
                                    last_order = None
                                    ignore_sold = 'short'
                        elif bullish_sell_signal[day] == 'long':  # selling long
                            if ignore_sold != 'long':
                                order_list.append([date_list[day], 'long sold', bullish_sell_price[day], None])
                                last_order = None
                            else:
                                order_list.append([date_list[day], None, None, None])
                                ignore_sold = None
                        else:
                            if last_order == 'long bought':
                                order_list.append([date_list[day], 'long hold', None, None])
                                last_order = 'long bought'
                            else:
                                order_list.append([date_list[day], None, None, None])
                                last_order = None
                    elif short:
                        if bearish_buy_signal[day] == 'short':  # buying short
                            if bearish_sell_signal[day] != 'short':
                                if last_order is None:
                                    order_list.append([date_list[day], 'short bought', open_list[day], bearish_quality[day]])
                                    last_order = 'short bought'
                                elif last_order == 'short bought':
                                    order_list.append([date_list[day], 'short hold', None, None])
                                    last_order = 'short bought'
                                elif last_order == 'long bought':
                                    order_list.append([date_list[day], 'long sold and short bought', [open_list[day], open_list[day]], bearish_quality[day]])
                                    last_order = 'short bought'
                                    ignore_sold = 'long'
                            else:
                                if last_order == '':
                                    order_list.append([date_list[day], 'short bought and short sold', [open_list[day], bearish_sell_price[day]], bearish_quality[day]])
                                    last_order = None
                                elif last_order == 'short bought':
                                    order_list.append([date_list[day], 'short sold', bearish_sell_price[day], None])
                                    last_order = None
                                elif last_order == 'long bought':
                                    order_list.append([date_list[day], 'long sold and short bought and short sold', [open_list[day], open_list[day], bearish_sell_price[day]], bearish_quality[day]])
                                    last_order = None
                                    ignore_sold = 'long'
                        elif bearish_sell_signal[day] == 'short':  # selling short
                            if ignore_sold != 'short':
                                order_list.append([date_list[day], 'short sold', bearish_sell_price[day], None])
                                last_order = None
                            else:
                                order_list.append([date_list[day], None, None, None])
                                ignore_sold = None
                        else:
                            if last_order == 'short bought':
                                order_list.append([date_list[day], 'short hold', None, None])
                                last_order = 'short bought'
                            else:
                                order_list.append([date_list[day], None, None, None])
                                last_order = None

                elif day == len(date_list) - 1:
                    if last_order == 'long bought':
                        order_list.append([date_list[day], 'long sold', bullish_sell_price[day], None])
                    elif last_order == 'short bought':
                        order_list.append([date_list[day], 'short sold', bearish_sell_price[day], None])
                    else:
                        order_list.append([date_list[day], None, None, None])

            benchmark_list.append(close_list[-1] / open_list[0])

            df = pd.DataFrame(order_list, columns=[date_or_datetime, 'type', 'price', 'quality'])
            export_csv.export_csv(stocks_csv_file_path + '/{}/order list/{}_order list.csv'.format(candle_length, ticker), df, 1)

    individual_stock_order_list_func()

    def order_list_compiler_func():
        order_date_list = []
        order_ticker_list = []
        order_type_list = []
        order_price_list = []
        order_quality_list = []
        stock_order_number_list = []
        for ticker in tickers:
            order_list_df_individual = pd.read_csv(stocks_csv_file_path + '/{}/order list/{}_order list.csv'.format(candle_length, ticker), index_col=False, header=0)
            order_date_list_individual = order_list_df_individual[date_or_datetime].tolist()
            order_type_list_individual = order_list_df_individual['type'].tolist()
            order_price_list_individual = order_list_df_individual['price'].tolist()
            order_quality_list_individual = order_list_df_individual['quality'].tolist()

            def order_list_append_func(order_type, stock_order_count_local, price_list_count_local, price_list_count_date_local):
                order_date_list.append(order_date_list_individual[x])
                order_ticker_list.append(ticker)
                order_type_list.append(order_type)
                if type(order_price_list_individual[x]) is float:
                    order_price_list.append(order_price_list_individual[x])
                elif type(order_price_list_individual[x]) is str:
                    if price_list_count_date_local == order_date_list_individual[x]:
                        price_list_count_date_local = order_date_list_individual[x]
                        res = order_price_list_individual[x].strip('][').split(', ')
                        order_price_list.append(res[price_list_count_local])
                        price_list_count_local += 1
                    else:
                        price_list_count_local = 0
                        price_list_count_date_local = order_date_list_individual[x]
                        res = order_price_list_individual[x].strip('][').split(', ')
                        order_price_list.append(res[price_list_count_local])
                        price_list_count_local += 1
                order_quality_list.append(order_quality_list_individual[x])
                stock_order_number_list.append(stock_order_count_local)
                stock_order_count_local += 1
                return stock_order_count_local, price_list_count_local, price_list_count_date_local

            stock_order_count = 0
            price_list_count = 0
            price_list_count_date = None
            for x in range(len(order_date_list_individual)):
                if order_type_list_individual[x] == 'long bought':
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('long bought', stock_order_count, price_list_count, price_list_count_date)
                elif order_type_list_individual[x] == 'long sold':
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('long sold', stock_order_count, price_list_count, price_list_count_date)
                elif order_type_list_individual[x] == 'short bought':
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('short bought', stock_order_count, price_list_count, price_list_count_date)
                elif order_type_list_individual[x] == 'short sold':
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('short sold', stock_order_count, price_list_count, price_list_count_date)
                elif order_type_list_individual[x] == 'short sold and long bought':
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('short sold', stock_order_count, price_list_count, price_list_count_date)
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('long bought', stock_order_count, price_list_count, price_list_count_date)
                elif order_type_list_individual[x] == 'long bought and long sold':
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('long bought', stock_order_count, price_list_count, price_list_count_date)
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('long sold', stock_order_count, price_list_count, price_list_count_date)
                elif order_type_list_individual[x] == 'short sold and long bought and long sold':
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('short sold', stock_order_count, price_list_count, price_list_count_date)
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('long bought', stock_order_count, price_list_count, price_list_count_date)
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('long sold', stock_order_count, price_list_count, price_list_count_date)
                elif order_type_list_individual[x] == 'long sold and short bought':
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('long sold', stock_order_count, price_list_count, price_list_count_date)
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('short bought', stock_order_count, price_list_count, price_list_count_date)
                elif order_type_list_individual[x] == 'short bought and short sold':
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('short bought', stock_order_count, price_list_count, price_list_count_date)
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('short sold', stock_order_count, price_list_count, price_list_count_date)
                elif order_type_list_individual[x] == 'long sold and short bought and short sold':
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('long sold', stock_order_count, price_list_count, price_list_count_date)
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('short bought', stock_order_count, price_list_count, price_list_count_date)
                    stock_order_count, price_list_count, price_list_count_date = order_list_append_func('short sold', stock_order_count, price_list_count, price_list_count_date)

        order_nested_list = [order_date_list, order_ticker_list, order_type_list, order_price_list, order_quality_list, stock_order_number_list]
        order_df = pd.DataFrame(list(zip(*order_nested_list)), columns=[date_or_datetime, 'ticker', 'type', 'price', 'quality', 'stock order number'])
        order_df = order_df.sort_values(by=[date_or_datetime, 'stock order number'], ascending=[True, True])
        order_df.to_csv('order list.csv', mode='w', index=False)

        if not order_size_based_on_money:
            order_date_list1 = order_df[date_or_datetime].tolist()
            order_ticker_list1 = order_df['ticker'].tolist()
            order_type_list1 = order_df['type'].tolist()
            order_price_list1 = order_df['price'].tolist()
            order_quality_list1 = order_df['quality'].tolist()
            stock_order_number_list1 = order_df['stock order number'].tolist()
            order_date_list2 = order_df[date_or_datetime].tolist()
            order_ticker_list2 = order_df['ticker'].tolist()
            order_type_list2 = order_df['type'].tolist()
            order_price_list2 = order_df['price'].tolist()
            order_quality_list2 = order_df['quality'].tolist()
            stock_order_number_list2 = order_df['stock order number'].tolist()
            dropped_ticker_list = []
            positions_counter = 0
            drop_counter = 0
            for enu, type_iterator in enumerate(order_type_list1):
                if order_ticker_list1[enu] in dropped_ticker_list and type_iterator == 'long sold' or order_ticker_list1[enu] in dropped_ticker_list and type_iterator == 'short sold':
                    order_date_list2.pop(enu - drop_counter)
                    order_ticker_list2.pop(enu - drop_counter)
                    order_type_list2.pop(enu - drop_counter)
                    order_price_list2.pop(enu - drop_counter)
                    order_quality_list2.pop(enu - drop_counter)
                    stock_order_number_list2.pop(enu - drop_counter)
                    dropped_ticker_list.remove(order_ticker_list1[enu])
                    drop_counter += 1
                    # print(order_ticker_list1[enu], type_iterator, 'remove', order_date_list1[enu])
                elif positions_counter == a and type_iterator == 'long bought' or positions_counter == a and type_iterator == 'short bought':
                    order_date_list2.pop(enu - drop_counter)
                    order_ticker_list2.pop(enu - drop_counter)
                    order_type_list2.pop(enu - drop_counter)
                    order_price_list2.pop(enu - drop_counter)
                    order_quality_list2.pop(enu - drop_counter)
                    stock_order_number_list2.pop(enu - drop_counter)
                    dropped_ticker_list.append(order_ticker_list1[enu])
                    drop_counter += 1
                    # print(order_ticker_list1[enu], type_iterator, 'append', order_date_list1[enu])
                elif type_iterator == 'long bought' or type_iterator == 'short bought':
                    positions_counter += 1
                    # print(order_ticker_list1[enu], type_iterator,  '+1', order_date_list1[enu])
                elif type_iterator == 'long sold' or type_iterator == 'short sold':
                    positions_counter -= 1
                    # print(order_ticker_list1[enu], type_iterator, '-1', order_date_list1[enu])
            order_nested_list = [order_date_list2, order_ticker_list2, order_type_list2, order_price_list2, order_quality_list2, stock_order_number_list2]
            order_df = pd.DataFrame(list(zip(*order_nested_list)), columns=[date_or_datetime, 'ticker', 'type', 'price', 'quality', 'stock order number'])
            order_df.to_csv('order list.csv', mode='w', index=False)

    order_list_compiler_func()

    def trade_func():
        def long_buy_func(money0, portfolio0, ticker0, price0, estimated_value0, date0, order_quality0, lowest_order_quality0, highest_order_quality0):
            if order_size_based_on_money and order_size_based_on_quality:
                trade_size_before_quality = (money/a)**b
                trade_size_before_quality_list.append(trade_size_before_quality)
                quality_difference = highest_order_quality0-lowest_order_quality0
                trade_size_quality_modifier = (1-(quality_difference/(order_quality0-lowest_order_quality+quality_difference+c)))**d
                trade_size_quality_modifier_list.append(trade_size_quality_modifier)
                trade_size0 = trade_size_before_quality*trade_size_quality_modifier
                trade_size_list.append(((1 - (quality_difference / (order_quality0 - lowest_order_quality + quality_difference + c))) ** d)*(((money/estimated_value0)/a)**b))
                portfolio0[ticker0] = [trade_size0/price0, price0, date0, ((1 - (quality_difference / (order_quality0 - lowest_order_quality + quality_difference + c))) ** d)*(((money/estimated_value0)/a)**b)]
            if order_size_based_on_money and not order_size_based_on_quality:
                trade_size0 = (money/a)**b
                trade_size_list.append(((money/estimated_value0)/a)**b)
                portfolio0[ticker0] = [trade_size0/price0, price0, date0, ((money/estimated_value0)/a)**b]
            if not order_size_based_on_money:
                trade_size0 = estimated_value0 / a
                trade_size_list.append(trade_size0)
                portfolio0[ticker0] = [trade_size0 / price0, price0, date0, trade_size0]
            money0 -= trade_size0
            trade_duration_list.append(None)
            money_list.append(money0)
            return money0, portfolio0

        def long_sell_func(money0, portfolio0, ticker0, price0, date0):
            trade_duration_list.append(get_time_delta.get_time_delta(date0, portfolio0.get(ticker0)[2]))
            money0 += price0*(portfolio0.get(ticker0)[0])
            trade_size0 = (portfolio0.get(ticker0)[3])
            trade_profit0 = (price0/(portfolio0.get(ticker0)[1]))-1
            del portfolio0[ticker0]
            trade_size_list.append(None)
            trade_size_before_quality_list.append(None)
            trade_size_quality_modifier_list.append(None)
            money_list.append(money0)
            return money0, portfolio0, trade_profit0, trade_size0

        def short_buy_func(money0, portfolio0, ticker0, price0, estimated_value0, date0, order_quality0, lowest_order_quality0, highest_order_quality0):
            if order_size_based_on_money and order_size_based_on_quality:
                trade_size_before_quality = (money/a)**b
                trade_size_before_quality_list.append(trade_size_before_quality)
                quality_difference = highest_order_quality0-lowest_order_quality0
                trade_size_quality_modifier = (1-(quality_difference/(order_quality0-lowest_order_quality+quality_difference+c)))**d
                trade_size_quality_modifier_list.append(trade_size_quality_modifier)
                trade_size0 = trade_size_before_quality*trade_size_quality_modifier
                trade_size_list.append(((1 - (quality_difference / (order_quality0 - lowest_order_quality + quality_difference + c))) ** d)*(((money/estimated_value0)/a)**b))
                portfolio0[ticker0] = [trade_size0/price0, price0, date0, ((1 - (quality_difference / (order_quality0 - lowest_order_quality + quality_difference + c))) ** d)*(((money/estimated_value0)/a)**b)]
            if order_size_based_on_money and not order_size_based_on_quality:
                trade_size0 = (money/a)**b
                trade_size_list.append(((money/estimated_value0)/a)**b)
                portfolio0[ticker0] = [trade_size0/price0, price0, date0, ((money/estimated_value0)/a)**b]
            if not order_size_based_on_money:
                trade_size0 = estimated_value0 / a
                trade_size_list.append(trade_size0)
                portfolio0[ticker0] = [trade_size0 / price0, price0, date0, trade_size0]
            money0 -= trade_size0
            trade_duration_list.append(None)
            money_list.append(money0)
            return money0, portfolio0

        def short_sell_func(money0, portfolio0, ticker0, price0, date0):
            trade_duration_list.append(get_time_delta.get_time_delta(date0, portfolio0.get(ticker0)[2]))
            money0 += (portfolio0.get(ticker0)[1]-price0)*(portfolio0.get(ticker0)[0]) + portfolio0.get(ticker0)[1]*portfolio0.get(ticker0)[0]
            trade_size0 = (portfolio0.get(ticker0)[3])
            trade_profit0 = 1-(price0/portfolio0.get(ticker0)[1])
            del portfolio0[ticker0]
            trade_size_list.append(None)
            trade_size_before_quality_list.append(None)
            trade_size_quality_modifier_list.append(None)
            money_list.append(money0)
            return money0, portfolio0, trade_profit0, trade_size0

        order_list_df = pd.read_csv('order list.csv', index_col=False, header=0)
        order_date = order_list_df[date_or_datetime].tolist()
        order_type = order_list_df['type'].tolist()
        order_ticker = order_list_df['ticker'].tolist()
        order_price = order_list_df['price'].tolist()
        order_quality = order_list_df['quality'].tolist()
        order_date_ticker_price_type_nested_list = list(zip(*[order_date, order_ticker, order_price, order_type]))

        # sorted_order_quality_list_without_nones = list(filter(lambda item: not math.isnan(item), order_quality))
        # sorted_order_quality_list_without_nones.sort()
        # lowest_order_quality = sorted_order_quality_list_without_nones[0]
        # highest_order_quality = sorted_order_quality_list_without_nones[-1]

        money = 1
        portfolio = {}  # format = {ticker: [shares, purchase price, date, trade size]}

        trades = 0
        long_trades = 0
        short_trades = 0

        trade_profit_list = []
        trade_profit_list_accounting_trade_size = []
        long_trade_profit_list = []
        long_trade_profit_list_accounting_trade_size = []
        short_trade_profit_list = []
        short_trade_profit_list_accounting_trade_size = []

        trades_won = 0
        trades_lost = 0
        long_trades_won = 0
        long_trades_lost = 0
        short_trades_won = 0
        short_trades_lost = 0

        estimated_value = 1
        estimated_value_list = []
        portfolio_size_list = []
        trade_size_list = []
        trade_duration_list = []
        trade_size_before_quality_list = []
        trade_size_quality_modifier_list = []
        money_list = []
        portfolio_holdings_list = []
        portfolio_trade_size_list = []
        portfolio_price_list = []
        for count in range(len(order_date)):
            if order_type[count] == 'long bought':
                money, portfolio = long_buy_func(money, portfolio, order_ticker[count], order_price[count], estimated_value, order_date[count], order_quality[count], lowest_order_quality, highest_order_quality)
                trades += 1
                long_trades += 1
                trade_profit_list.append('')
                trade_profit_list_accounting_trade_size.append('')
                long_trade_profit_list.append('')
                long_trade_profit_list_accounting_trade_size.append('')
            if order_type[count] == 'long sold':
                money, portfolio, trade_profit, trade_size = long_sell_func(money, portfolio, order_ticker[count], order_price[count], order_date[count])
                trade_profit_list.append(trade_profit)
                trade_profit_list_accounting_trade_size.append(trade_profit*trade_size)
                long_trade_profit_list.append(trade_profit)
                long_trade_profit_list_accounting_trade_size.append(trade_profit*trade_size)
            if order_type[count] == 'short bought':
                money, portfolio = short_buy_func(money, portfolio, order_ticker[count], order_price[count], estimated_value, order_date[count], order_quality[count], lowest_order_quality, highest_order_quality)
                trades += 1
                short_trades += 1
                trade_profit_list.append('')
                trade_profit_list_accounting_trade_size.append('')
                short_trade_profit_list.append('')
                short_trade_profit_list_accounting_trade_size.append('')
            if order_type[count] == 'short sold':
                money, portfolio, trade_profit, trade_size = short_sell_func(money, portfolio, order_ticker[count], order_price[count], order_date[count])
                trade_profit_list.append(trade_profit)
                trade_profit_list_accounting_trade_size.append(trade_profit*trade_size)
                short_trade_profit_list.append(trade_profit)
                short_trade_profit_list_accounting_trade_size.append(trade_profit*trade_size)

            position_value = 0
            for value in portfolio.values():
                position_value += value[0] * value[1]
            estimated_value = money + position_value
            estimated_value_list.append(estimated_value)

            portfolio_size_list.append(len(portfolio))

            portfolio_holdings_appender = []
            portfolio_trade_size_appender = []
            portfolio_price_appender = []
            for key, value in portfolio.items():
                portfolio_holdings_appender.append(key)
                portfolio_trade_size_appender.append(value[3])
                portfolio_price_appender.append(value[1])
            portfolio_holdings_list.append(portfolio_holdings_appender)
            portfolio_trade_size_list.append(portfolio_trade_size_appender)
            portfolio_price_list.append(portfolio_price_appender)
        df_order = pd.read_csv('order list.csv', index_col=False, header=0)
        order_dates_list = df_order[date_or_datetime].tolist()
        years_str_raw = str(datetime.strptime(order_dates_list[-1][:10], '%Y-%m-%d') - datetime.strptime(order_dates_list[0][:10], '%Y-%m-%d'))[:-7]
        years_str = []
        for letter in years_str_raw:
            try:
                if type(int(letter)) is int:
                    years_str.append(str(letter))
            except ValueError:
                pass
        years = int(''.join(years_str))/365

        if calculate_sharpe_ratio:
            df_order = pd.read_csv('order list.csv', index_col=False, header=0)
            order_dates_list = df_order[date_or_datetime].tolist()
            tickers_held_list = []
            trade_size_list = []
            price_list = []
            order_date_counter = 0

            for compiled_date in dates_list_compiled:
                tickers_held_appender = []
                trade_size_appender = []
                price_appender = []
                try:
                    while compiled_date == order_dates_list[order_date_counter]:
                        tickers_held_appender.append(portfolio_holdings_list[order_date_counter])
                        trade_size_appender.append(portfolio_trade_size_list[order_date_counter])
                        price_appender.append(portfolio_price_list[order_date_counter])
                        order_date_counter += 1
                except IndexError:
                    pass
                if len(tickers_held_appender) != 0:
                    tickers_held_list.append(tickers_held_appender[-1])
                    trade_size_list.append(trade_size_appender[-1])
                    price_list.append(price_appender[-1])
                else:
                    if len(tickers_held_list) == 0:
                        tickers_held_list.append(None)
                        trade_size_list.append(None)
                        price_list.append(None)
                    else:
                        tickers_held_list.append(tickers_held_list[-1])
                        trade_size_list.append(trade_size_list[-1])
                        price_list.append(None)

            portfolio_holdings_nested_list = [dates_list_compiled, tickers_held_list]
            order_df = pd.DataFrame(list(zip(*portfolio_holdings_nested_list)), columns=[date_or_datetime, 'tickers held'])
            order_df.to_csv('portfolio holdings hourly.csv', mode='w', index=False)

            sell_dates = []
            sell_tickers = []
            sell_prices = []
            delete_lists = []
            for value in order_date_ticker_price_type_nested_list:
                if value[3] == 'bought':
                    delete_lists.append(value)
                else:
                    sell_dates.append(value[0])
                    sell_tickers.append(value[1])
                    sell_prices.append(value[2])

            for delete_list in delete_lists:
                order_date_ticker_price_type_nested_list.remove(delete_list)

            portfolio_value = 1
            portfolio_value_list = [1]
            date_daily_list = []
            portfolio_value_daily_list = []
            date_weekly_list = []
            portfolio_value_weekly_list = []
            for x, tickers_held in enumerate(tickers_held_list[0:-1]):
                yesterday = dates_list_compiled[x][0:10]
                today = dates_list_compiled[x + 1][0:10]
                if tickers_held is not None:
                    for y, ticker in enumerate(tickers_held):
                        df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), index_col=False, header=0)
                        date_list = df_raw[date_or_datetime].tolist()
                        close_list = df_raw['close'].tolist()
                        date_close_dict = {date_list[z]: close_list[z] for z in range(len(date_list))}
                        try:
                            if dates_list_compiled[x] in sell_dates and ticker in sell_tickers:
                                if sell_dates.index(dates_list_compiled[x]) == sell_tickers.index(ticker):
                                    sell_date_index = sell_dates.index(dates_list_compiled[x])
                                    portfolio_value += ((sell_prices[sell_date_index]/date_close_dict.get(dates_list_compiled[x]))-1) * trade_size_list[x][y]
                                else:
                                    portfolio_value += ((date_close_dict.get(dates_list_compiled[x + 1]) / date_close_dict.get(dates_list_compiled[x])) - 1) * trade_size_list[x][y]
                            else:
                                portfolio_value += ((date_close_dict.get(dates_list_compiled[x + 1]) / date_close_dict.get(dates_list_compiled[x])) - 1) * trade_size_list[x][y]
                        except TypeError:
                            pass
                portfolio_value_list.append(portfolio_value)
                if today != yesterday:
                    date_daily_list.append(today)
                    portfolio_value_daily_list.append(portfolio_value_list[-2])
                    if datetime.strptime(today, '%Y-%m-%d').day - datetime.strptime(yesterday, '%Y-%m-%d').day > 1:
                        date_weekly_list.append(today)
                        portfolio_value_weekly_list.append(portfolio_value_list[-2])

            weeks_won = 0
            weeks_uninvested = 0
            weeks_lost = 0
            for counter in range(len(portfolio_value_weekly_list[:-1])):
                if portfolio_value_weekly_list[counter+1] > portfolio_value_weekly_list[counter]:
                    weeks_won += 1
                elif portfolio_value_weekly_list[counter + 1] < portfolio_value_weekly_list[counter]:
                    weeks_lost += 1
                else:
                    weeks_uninvested += 1

            order_df = order_df.assign(portfolio_value=portfolio_value_list)
            order_df['change'] = order_df['portfolio_value'].pct_change()

            order_daily_nested_list = [date_daily_list, portfolio_value_daily_list]
            order_daily_df = pd.DataFrame(list(zip(*order_daily_nested_list)), columns=['datetime', 'portfolio value'])
            order_daily_df['change'] = order_daily_df['portfolio value'].pct_change()

            order_weekly_nested_list = [date_weekly_list, portfolio_value_weekly_list]
            order_weekly_df = pd.DataFrame(list(zip(*order_weekly_nested_list)), columns=['datetime', 'portfolio value'])
            order_weekly_df['change'] = order_weekly_df['portfolio value'].pct_change()

            order_df.to_csv('portfolio holdings hourly.csv', mode='w', index=False)
            order_daily_df.to_csv('portfolio holdings daily.csv', mode='w', index=False)
            order_weekly_df.to_csv('portfolio holdings weekly.csv', mode='w', index=False)

            pct_change_df = pd.DataFrame()
            pct_change_df['pct'] = order_df['portfolio_value'].pct_change()
            pct_list = pct_change_df['pct'].iloc[1:]
            pct_list_geometric = pct_list + 1
            hourly_volatility = pct_list.std()
            annual_volatility = hourly_volatility * np.sqrt(2008)

            yearly_return = money ** (1/years) - 1

            sharpe_ratio = (yearly_return - risk_free_rate) / annual_volatility

            if calc_profit_odds:
                var_list = []

                for x in range(int(hours)):
                    var_list.append(np.random.uniform(statistics.geometric_mean(pct_list_geometric) - hourly_volatility, statistics.geometric_mean(pct_list_geometric) + hourly_volatility, sims))

                new_var_list = []
                for x in range(len(var_list[0])):
                    new_var_list_appender = []
                    for y in var_list:
                        new_var_list_appender.append(y[x])
                    new_var_list_appender = np.array(new_var_list_appender)
                    new_var_list.append(new_var_list_appender)

                profit_result = np.prod(new_var_list, axis=1)

                profit_odds_result = ((profit_result > profit_minimum).sum() / sims)

                plt.figure()
                plt.hist(profit_result, density=True, edgecolor='white')
                plt.axvline(profit_minimum, color='r')
                plt.show()

        trade_profit_list_ui = []
        avg_win_list = []
        avg_loss_list = []
        for profit in trade_profit_list:
            if profit != '':
                trade_profit_list_ui.append(str(profit*100) + '%')
            else:
                trade_profit_list_ui.append('')
            if profit == '':
                pass
            elif profit > 0:
                trades_won += 1
                avg_win_list.append(profit+1)
            else:
                trades_lost += 1
                avg_loss_list.append(profit+1)
        trade_profit_list_accounting_trade_size_ui = []
        for profit in trade_profit_list_accounting_trade_size:
            if profit != '':
                trade_profit_list_accounting_trade_size_ui.append(str(profit*100) + '%')
            else:
                trade_profit_list_accounting_trade_size_ui.append('')

        if long:
            long_trade_profit_list_ui = []
            long_avg_win_list = []
            long_avg_loss_list = []
            for profit in long_trade_profit_list:
                if profit != '':
                    long_trade_profit_list_ui.append(str(profit * 100) + '%')
                else:
                    long_trade_profit_list_ui.append('')
                if profit == '':
                    pass
                elif profit > 0:
                    long_trades_won += 1
                    long_avg_win_list.append(profit + 1)
                else:
                    long_trades_lost += 1
                    long_avg_loss_list.append(profit + 1)
            long_trade_profit_list_accounting_trade_size_ui = []
            for profit in long_trade_profit_list_accounting_trade_size:
                if profit != '':
                    long_trade_profit_list_accounting_trade_size_ui.append(str(profit * 100) + '%')
                else:
                    long_trade_profit_list_accounting_trade_size_ui.append('')

        if short:
            short_trade_profit_list_ui = []
            short_avg_win_list = []
            short_avg_loss_list = []
            for profit in short_trade_profit_list:
                if profit != '':
                    short_trade_profit_list_ui.append(str(profit * 100) + '%')
                else:
                    short_trade_profit_list_ui.append('')
                if profit == '':
                    pass
                elif profit > 0:
                    short_trades_won += 1
                    short_avg_win_list.append(profit + 1)
                else:
                    short_trades_lost += 1
                    short_avg_loss_list.append(profit + 1)
            short_trade_profit_list_accounting_trade_size_ui = []
            for profit in short_trade_profit_list_accounting_trade_size:
                if profit != '':
                    short_trade_profit_list_accounting_trade_size_ui.append(str(profit * 100) + '%')
                else:
                    short_trade_profit_list_accounting_trade_size_ui.append('')

        order_list_df = pd.read_csv('order list.csv', index_col=False, header=0)
        order_list_df = order_list_df.assign(profit_of_trade_discounting_trade_size=trade_profit_list_ui)
        # order_list_df = order_list_df.assign(profit_of_trade_accounting_trade_size=trade_profit_list_accounting_trade_size_ui)
        # order_list_df = order_list_df.assign(trade_size=trade_size_list)
        if order_size_based_on_quality:
            order_list_df = order_list_df.assign(trade_size_before_quality=trade_size_before_quality_list)
            order_list_df = order_list_df.assign(trade_size_quality_modifier=trade_size_quality_modifier_list)
        order_list_df = order_list_df.assign(trade_duration=trade_duration_list)
        order_list_df = order_list_df.assign(portfolio_value=estimated_value_list)
        order_list_df = order_list_df.assign(portfolio_size=portfolio_size_list)
        # order_list_df = order_list_df.assign(money=money_list)
        order_list_df.to_csv('order list.csv', mode='w', index=False)

        df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, tickers[0]), index_col=False, header=0)
        raw_dates = df_raw[date_or_datetime].tolist()
        avg_port_list = []
        for count in range(len(order_date)-1):
            day_length = get_time_delta.get_time_delta(order_date[count + 1], order_date[count])
            avg_port_list.append(day_length * (1-(money_list[count]/estimated_value_list[count])))
        avg_port = sum(avg_port_list) / get_time_delta.get_time_delta(raw_dates[-1], raw_dates[0])

        sorted_portfolio_size_list = portfolio_size_list
        sorted_portfolio_size_list.sort(reverse=True)

        trade_duration_list_without_nones = list(filter(lambda item: item is not None, trade_duration_list))

        trade_profit_list_without_nones = list(filter(None, trade_profit_list))
        geometric_trade_profit_list = []
        for trade in trade_profit_list_without_nones:
            geometric_trade_profit_list.append(trade+1)
        average_profit_per_trade = statistics.geometric_mean(geometric_trade_profit_list)
        average_profit_per_trade = average_profit_per_trade-1

        trade_profit_list_ats_without_nones = list(filter(None, trade_profit_list_accounting_trade_size))  # ats = accounting trade size
        geometric_trade_profit_list_ats = []
        for trade in trade_profit_list_ats_without_nones:
            geometric_trade_profit_list_ats.append(trade + 1)
        average_profit_per_trade_ats = statistics.geometric_mean(geometric_trade_profit_list_ats)
        average_profit_per_trade_ats = average_profit_per_trade_ats - 1

        if long:
            long_trade_profit_list_without_nones = list(filter(None, long_trade_profit_list))
            long_geometric_trade_profit_list = []
            for trade in long_trade_profit_list_without_nones:
                long_geometric_trade_profit_list.append(trade + 1)
            long_average_profit_per_trade = statistics.geometric_mean(long_geometric_trade_profit_list)
            long_average_profit_per_trade = long_average_profit_per_trade - 1

            long_trade_profit_list_ats_without_nones = list(filter(None, long_trade_profit_list_accounting_trade_size))  # ats = accounting trade size
            long_geometric_trade_profit_list_ats = []
            for trade in long_trade_profit_list_ats_without_nones:
                long_geometric_trade_profit_list_ats.append(trade + 1)
            long_average_profit_per_trade_ats = statistics.geometric_mean(long_geometric_trade_profit_list_ats)
            long_average_profit_per_trade_ats = long_average_profit_per_trade_ats - 1

        if short:
            short_trade_profit_list_without_nones = list(filter(None, short_trade_profit_list))
            short_geometric_trade_profit_list = []
            for trade in short_trade_profit_list_without_nones:
                short_geometric_trade_profit_list.append(trade + 1)
            short_average_profit_per_trade = statistics.geometric_mean(short_geometric_trade_profit_list)
            short_average_profit_per_trade = short_average_profit_per_trade - 1

            short_trade_profit_list_ats_without_nones = list(filter(None, short_trade_profit_list_accounting_trade_size))  # ats = accounting trade size
            short_geometric_trade_profit_list_ats = []
            for trade in short_trade_profit_list_ats_without_nones:
                short_geometric_trade_profit_list_ats.append(trade + 1)
            short_average_profit_per_trade_ats = statistics.geometric_mean(short_geometric_trade_profit_list_ats)
            short_average_profit_per_trade_ats = short_average_profit_per_trade_ats - 1

        def print_data_func():
            try:
                print('----------------------------------------')
                print('start date: ' + start_date)
                print('end date: ' + end_date)
                print('model: ' + str((money-1)*100) + '%')
                print('average percentage of portfolio in the market: ' + str(avg_port*100) + '%')
                print('trades: ' + str(trades))
                print('trades won: ' + str(trades_won))
                print('trades lost: ' + str(trades_lost))
                if long and short:
                    print('long trades won: ' + str(long_trades_won))
                    print('long trades lost: ' + str(long_trades_lost))
                    print('short trades won: ' + str(short_trades_won))
                    print('short trades lost: ' + str(short_trades_lost))
                print('average profit per trade (discounting trade size): ' + str(average_profit_per_trade*100) + '%')
                print('average profit per trade (accounting trade size): ' + str(average_profit_per_trade_ats*100) + '%')
                if long and short:
                    print('average profit per long trade (discounting trade size): ' + str(long_average_profit_per_trade*100) + '%')
                    print('average profit per long trade (accounting trade size): ' + str(long_average_profit_per_trade_ats*100) + '%')
                    print('average profit per short trade (discounting trade size): ' + str(short_average_profit_per_trade*100) + '%')
                    print('average profit per short trade (accounting trade size): ' + str(short_average_profit_per_trade_ats*100) + '%')
                print('average profit of winning trade: ' + str((statistics.geometric_mean(avg_win_list)-1)*100) + '%')
                print('average profit of losing trade: ' + str((statistics.geometric_mean(avg_loss_list)-1)*100) + '%')
                if long and short:
                    print('average profit of winning long trade: ' + str((statistics.geometric_mean(long_avg_win_list)-1)*100) + '%')
                    print('average profit of losing long trade: ' + str((statistics.geometric_mean(long_avg_loss_list)-1)*100) + '%')
                    print('average profit of winning short trade: ' + str((statistics.geometric_mean(short_avg_win_list)-1)*100) + '%')
                    print('average profit of losing short trade: ' + str((statistics.geometric_mean(short_avg_loss_list)-1)*100) + '%')
                print('average trade duration: ' + str(statistics.mean(trade_duration_list_without_nones)))
                print('most positions at once: ' + str(sorted_portfolio_size_list[0]))
            except (IndexError, statistics.StatisticsError):
                print('lack of positions taken')
            if calculate_sharpe_ratio:
                print('----------------------------------------')
                print('yearly sharpe ratio')
                print('hourly return differential')
                print('sharpe ratio: ' + str(sharpe_ratio))
                print('volatility: ' + str(annual_volatility*100) + '%')
                print('volatility divided by average percentage of portfolio in the market: ' + str(annual_volatility/avg_port))
                print('----------------------------------------')
                print('weeks won: ' + str(weeks_won))
                print('weeks uninvested: ' + str(weeks_uninvested))
                print('weeks lost: ' + str(weeks_lost))
            if calc_profit_odds:
                print('chance of making more than a ' + str((profit_minimum-1) * 100) + '% return within ' + str(hours) + ' hours: ' + str(profit_odds_result*100) + '%')
            print('----------------------------------------')
            # print('benchmark: ' + str((statistics.mean(benchmark_list)-1)*100) + '%')
            # print('benchmark hours in market: ' + len(dates_list_compiled))

        print_data_func()

    trade_func()

    print('back_test_func complete')


def graph_func():
    ticker = 'AAPL'
    df_raw = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), header=0)
    df_rsi = pd.read_csv(stocks_csv_file_path + '/{}/rsi/{}_rsi.csv'.format(candle_length, ticker), header=0)
    df_highs_price = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/2/highs price 2/{}_highs price 2.csv'.format(candle_length, ticker), header=0)
    df_lows_price = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/2/lows price 2/{}_lows price 2.csv'.format(candle_length, ticker), header=0)
    df_highs_rsi = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/2/highs rsi 2/{}_highs rsi 2.csv'.format(candle_length, ticker), header=0)
    df_lows_rsi = pd.read_csv(stocks_csv_file_path + '/{}/highs and lows/2/lows rsi 2/{}_lows rsi 2.csv'.format(candle_length, ticker), header=0)

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

    fig.append_trace(go.Scatter(x=df_rsi.index, y=df_rsi[str(rsi_length)], line_color='purple'), row=2, col=1)
    fig.append_trace(go.Scatter(x=dropna_df_highs_rsi.index, y=dropna_df_highs_rsi['rsis'], type='scatter', mode='lines', line_color='green'), row=2, col=1)
    fig.append_trace(go.Scatter(x=dropna_df_lows_rsi.index, y=dropna_df_lows_rsi['rsis'], type='scatter', mode='lines', line_color='red'), row=2, col=1)
    fig.add_scatter(x=df_raw.index, y=df_highs_rsi['point_position'], mode='markers', marker=dict(size=4, color='Green'), row=2, col=1)
    fig.add_scatter(x=df_raw.index, y=df_lows_rsi['point_position'], mode='markers', marker=dict(size=4, color='Red'), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.show()


def graph_data_func():
    df_order = pd.read_csv('order list.csv', index_col=False, header=0)
    profits_list = df_order['profit_of_trade_discounting_trade_size'].tolist()
    datetime_list = df_order[str(date_or_datetime)].tolist()

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


if __name__ == '__main__':
    raw_scraping = True
    if equity == 'stocks' and raw_scraping:
        if nation == 'usa':
            sheet_id = "1U592Qmdg0GAGItmv96Ou6c_WvsFQ14QCkbKXLNMQOEs"
            sp500 = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
            pre_tickers_dot = list(sp500.loc[:])
            tickers0 = []
            tickers1 = []
            for i in range(len(pre_tickers_dot)):
                tickers0.append(pre_tickers_dot[i].replace(".", "_"))
                space_index = tickers0[i - 1].index(" ")
                tickers1.append(tickers0[i - 1][0:space_index])
            tickers1.remove('Symbol')
            tickers1.remove('Symbol')
            tickers1.remove('BF_B')
            tickers1.remove('BRK_B')
            tickers_df0 = pd.DataFrame(tickers1)
            tickers_df0.to_csv('tickers', mode='w')
        elif nation == 'international':
            url = 'https://stockanalysis.com/list/biggest-companies/'
            headers = {'User-Agent': 'Mozilla/5.0'}

            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            resp_data = resp.read()

            tickers_list = re.findall(r's:"(\w{1,})', str(resp_data))
            tickers_list = tickers_list[:stock_number]
            tickers_list = list(set(tickers_list))
            try:
                tickers_list.remove('BRK')
                tickers_list.append('BRK-B')
            except ValueError:
                pass
            try:
                tickers_list.remove('FER')
            except ValueError:
                pass

            tickers_df0 = pd.DataFrame(tickers_list, columns=['tickers'])
            tickers_df0.to_csv('tickers.csv', mode='w')
    if equity == 'stocks':
        tickers_df1 = pd.read_csv('tickers.csv', index_col=False, header=0)
        tickers = tickers_df1['tickers'].tolist()
        config.tickers = tickers
    dates_list_compiled = []
    if raw_scraping:
        tickers_blacklist_list = []
        raw_scraping_func()
    for tick in tickers:
        df_raw_compiled = pd.read_csv(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, tick), index_col=False, header=0)
        dates_list_for_compiler = df_raw_compiled[date_or_datetime].tolist()
        dates_list_compiled = dates_list_compiled + dates_list_for_compiler
    dates_list_compiled = list(set(dates_list_compiled))
    dates_list_compiled.sort()
    config.dates_list_compiled = dates_list_compiled

    # rsi_func()
    # ema_func()
    # stochastic_func()
    # volatility_func()
    # candle_stick_func()
    #
    # rsi_quality_func()
    # ema_quality_func()
    # stochastic_quality_func()
    #
    # highs_lows_func()
    #
    # support_and_resistance_func()
    #
    # breakout_func()
    #
    # indentify_divergences_func()
    #
    # stochastic_crossover_func()

    # buy_signal_func()

    # sell_signal_func()
    #
    # back_test_func()

    # graph_func()

    # graph_data_func()

    time2 = time.time()
    timer = time2 - time1
    print('Finished in', int(timer//3600), 'hours', int((timer % 3600)//60), 'minutes', timer % 60, 'seconds')
