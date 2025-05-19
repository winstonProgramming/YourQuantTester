import math

import pandas as pd
import talib

import config
import file_management

def rsi():
    for ticker in config.tickers:
        df_raw = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
            index_col=False, header=0)
        close = pd.Series(df_raw.close)
        rsi_vals = talib.RSI(close, timeperiod=config.rsi_length)
        df = pd.DataFrame(rsi_vals.tolist(), columns=[config.rsi_length])
        df.index = df_raw[config.datetime_str]
        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv',
            df, 1)

    print('rsi complete')

def ema():
    for ticker in config.tickers:
        df_raw = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
            index_col=False, header=0)
        close = df_raw['close'].tolist()
        sma_list = []
        for i in range(config.ema_length):
            sma_list.append(close[i])
        starting_sma = sum(sma_list)/config.ema_length
        ema_list_price = [None]*(config.ema_length-1)
        ema_list_price.append(starting_sma)
        for day in range(len(close)-config.ema_length):
            ema_list_price.append(
                close[day+config.ema_length]*(2/(config.ema_length+1)) + ema_list_price[-1]*(1-(2/(config.ema_length+1))))
        ema_list_price[config.ema_length-1] = None
        ema_list = []
        for day in range(len(close)):
            if ema_list_price[day] is None:
                ema_list.append(None)
            else:
                ema_list.append(close[day]/ema_list_price[day])
        df = pd.DataFrame(ema_list, columns=[config.ema_length])
        df.index = df_raw[config.datetime_str]
        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/ema/{ticker}_ema.csv',
            df, 1)

    print('ema complete')

def stochastic():
    for ticker in config.tickers:
        df_raw = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
            index_col=False, header=0)
        high = pd.Series(df_raw.high)
        low = pd.Series(df_raw.low)
        close = pd.Series(df_raw.close)
        stochastic_values = talib.STOCH(high, low, close,
                                 fastk_period=config.fastk_period,
                                 slowk_period=config.slowk_period,
                                 slowd_period=config.slowd_period)
        stochastic_nested_list = [stochastic_values[0], stochastic_values[1]]
        df = pd.DataFrame(list(zip(*stochastic_nested_list)), columns=['k', 'd'])
        df.index = df_raw[config.datetime_str]
        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
            df, 1)

    print('stochastic complete')

def volatility():
    for ticker in config.tickers:
        df_raw = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
            index_col=False, header=0)

        pct_change_df = pd.DataFrame()
        pct_change_df['pct'] = df_raw['close'].pct_change()

        pct_list = pct_change_df['pct'].iloc[1:]
        volatility_series = pct_list.rolling(config.rolling_volatility_length).std(ddof=1)

        volatility_list = volatility_series.tolist()
        volatility_list.insert(0, math.nan)
        volatility_series = pd.Series(volatility_list)

        df = pd.DataFrame({config.datetime_str: df_raw[config.datetime_str], 'volatility': volatility_series.values})

        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/volatility/{ticker}_volatility.csv',
            df, 1)

    print('volatility complete')

def candle_stick():
    for ticker in config.tickers:
        df_raw = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
            index_col=False, header=0)

        df = pd.DataFrame()

        if config.long:
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

        if config.short:
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

        df.index = df_raw[config.datetime_str]

        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/candle stick/{ticker}_candle stick.csv',
            df, 1)

    print('candle_stick complete')

def rsi_quality():
    for ticker in config.tickers:
        df_rsi = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv',
            index_col=False, header=0)
        rsi_list = df_rsi[str(config.rsi_length)].tolist()

        if config.long:
            bullish_rsi_quality_list = []
            for rsi_value in rsi_list:
                bullish_rsi_quality_list.append((100-rsi_value)*config.rsi_quality_multiplier)

            df_rsi = df_rsi.assign(bullish_quality=bullish_rsi_quality_list)

        if config.short:
            bearish_rsi_quality_list = []
            for rsi_value in rsi_list:
                bearish_rsi_quality_list.append(rsi_value*config.rsi_quality_multiplier)

            df_rsi = df_rsi.assign(bearish_quality=bearish_rsi_quality_list)

        df_rsi.to_csv(config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv', mode='w')

    print('rsi_quality_func complete')

def ema_quality():
    for ticker in config.tickers:
        df_ema = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/ema/{ticker}_ema.csv',
            index_col=False, header=0)
        ema_list = df_ema[str(config.ema_length)].tolist()

        if config.long:
            ema_quality_list = []
            for ema_value in ema_list:
                if ema_value > 1:
                    ema_quality_list.append(
                        config.ema_quality_multiplier*(abs((ema_value-1)**config.ema_quality_exponential_multiplier)))
                else:
                    ema_quality_list.append(
                        config.ema_quality_multiplier*(-1*abs((ema_value-1)**config.ema_quality_exponential_multiplier)))

            df_ema = df_ema.assign(bullish_quality=ema_quality_list)

        if config.short:
            ema_quality_list = []
            for ema_value in ema_list:
                if ema_value > 1:
                    ema_quality_list.append(
                        config.ema_quality_multiplier * (-1 * abs((ema_value - 1) ** config.ema_quality_exponential_multiplier)))
                else:
                    ema_quality_list.append(
                        config.ema_quality_multiplier * (abs((ema_value - 1) ** config.ema_quality_exponential_multiplier)))

            df_ema = df_ema.assign(bearish_quality=ema_quality_list)

        df_ema.to_csv(config.stocks_csv_file_path + f'/{config.candle_length}/ema/{ticker}_ema.csv', mode='w')

    print('ema_quality complete')

def stochastic_quality():
    for ticker in config.tickers:
        df_stochastic = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
            index_col=False, header=0)
        k_list = df_stochastic['k'].tolist()

        if config.long:
            stochastic_quality_list = []
            for k in k_list:
                stochastic_quality_list.append((100-k)*config.stochastic_quality_multiplier)

            df_stochastic = df_stochastic.assign(bullish_quality=stochastic_quality_list)

        if config.short:
            stochastic_quality_list = []
            for k in k_list:
                stochastic_quality_list.append(k * config.stochastic_quality_multiplier)

            df_stochastic = df_stochastic.assign(bearish_quality=stochastic_quality_list)

        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
            df_stochastic, 1)

    print('stochastic complete')

def indicators():
    rsi()
    ema()
    stochastic()
    volatility()
    candle_stick()
    rsi_quality()
    ema_quality()
    stochastic_quality()
