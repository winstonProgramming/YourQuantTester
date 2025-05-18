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
        rsi_nested_list = []
        for length in config.rsi_lengths:
            rsi_vals = talib.RSI(close, timeperiod=length)
            rsi_nested_list.append(rsi_vals.tolist())
        df = pd.DataFrame(list(zip(*rsi_nested_list)), columns=config.rsi_lengths)
        df.index = df_raw[config.datetime_str]
        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv',
            df, 1)

    print('rsi complete')

