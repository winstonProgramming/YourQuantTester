import time
from datetime import datetime
import math
import statistics
import urllib.request
import urllib.parse

import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import permutations, product
# import multiprocessing
# from tvDatafeed import TvDatafeed, Interval
# import pylab
# from collections import OrderedDict
# import ast
# import itertools

import config
import calibrate_config
import file_management
import get_tickers
import indicators
import signals
import backtest
import trade_signals
import graph

time1 = time.time()

# --------------------settings--------------------

config.stocks_csv_file_path = 'D:/Winston\'s Data/Downloads/stocks_csv'

config.equity = 'stocks'  # currently supporting stocks
config.nation = 'international'  # usa or international
config.stock_number = 20
config.candle_length = '1h'
config.datetime_str = 'datetime'
config.start_date = '2024-6-1'
config.end_date = '2025-5-15'
config.long = True  # must be True if config.short = False
config.short = True  # must be True if config.long = False

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
                                'candle sticks': 1}

# order_dependant_buy_expirations = False

config.buy_signal_expiration_dict = {'divergences': 6,  # if order_dependant_buy_expirations is True
                                     'breakouts': math.nan,
                                     'candle sticks': math.nan}

config.buy_signal_expiration_list = [10, 5]  # len(buy_signal_expiration_list) = len(buy_signal_order_dict) - 1

# config.pre_earnings_date_omission = 1  # set to -1 to be inactive, omits buy signal if buy signal is before earnings date
# config.post_earnings_date_omission = 2  # set to -1 to be inactive, omits buy signal if buy signal is after earnings date

config.quality_minimum = 0

config.sell_signals_nested_list = [['sell signal indicator 1', 'artificial margin 1'],  # sell signal indicator, support resistance, artificial margin, sell time
                                   ['support resistance 1'],
                                   ['sell time 1']]

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

# --------------------functions--------------------

config.scrape_data_bool = True  # if running for first time or changing raw data of stocks, set to True

file_management.create_stocks_csv_folder()
calibrate_config.calibrate_config()
get_tickers.get_tickers()  # gathers tickers list

indicators.indicators()  # creates indicators
signals.signals()  # creates signals
trade_signals.trade_signals()  # creates trades
backtest.backtest()  # backtests
# graph.graph_profits()  # graphs profits

# graph.graph_stock('AAPL')  # graphs stock price and rsi; pass in ticker symbol of desired stock

time2 = time.time()
timer = time2 - time1
print('Finished in', int(timer//3600), 'hours', int((timer % 3600)//60), 'minutes', timer % 60, 'seconds')
