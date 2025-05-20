import math

import pandas as pd
import numpy as np

import config
import file_management
import get_time_delta

def highs_and_lows():
    def highs_price(before_range, after_range, version):
        for ticker in config.tickers:
            df_raw = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
                index_col=False, header=0)
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
            df.index = df_raw[config.datetime_str]
            file_management.export_csv(config.stocks_csv_file_path +
                                       f'/{config.candle_length}/highs and lows/{version}/highs price {version}/{ticker}_highs price {version}.csv',
                                       df, 2)

    def lows_price(before_range, after_range, version):
        for ticker in config.tickers:
            df_raw = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
                index_col=False, header=0)
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
            df.index = df_raw[config.datetime_str]
            file_management.export_csv(config.stocks_csv_file_path +
                                       f'/{config.candle_length}/highs and lows/{version}/lows price {version}/{ticker}_lows price {version}.csv',
                                       df, 2)

    def highs_rsi(before_range, after_range, version):
        for ticker in config.tickers:
            df_rsi = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv',
                index_col=False, header=0)
            rsi = df_rsi[str(config.rsi_length)].tolist()
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
                        position_list.append(df_rsi[str(config.rsi_length)][day]+1.4)
                        price_list.append(df_rsi[str(config.rsi_length)][day])
                        high_list.append('True')
            high_high_nested_list = [high_list, price_list, position_list]
            df = pd.DataFrame(list(zip(*high_high_nested_list)), columns=['high', 'rsis', 'point_position'])
            df.index = df_rsi[config.datetime_str]
            file_management.export_csv(config.stocks_csv_file_path +
                                       f'/{config.candle_length}/highs and lows/{version}/highs rsi {version}/{ticker}_highs rsi {version}.csv',
                                       df, 2)

    def lows_rsi(before_range, after_range, version):
        for ticker in config.tickers:
            df_rsi = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv',
                index_col=False, header=0)
            rsi = df_rsi[str(config.rsi_length)].tolist()
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
                        position_list.append(df_rsi[str(config.rsi_length)][day]-1.4)
                        price_list.append(df_rsi[str(config.rsi_length)][day])
                        low_list.append('True')
            low_low_nested_list = [low_list, price_list, position_list]
            df = pd.DataFrame(list(zip(*low_low_nested_list)), columns=['low', 'rsis', 'point_position'])
            df.index = df_rsi[config.datetime_str]
            file_management.export_csv(config.stocks_csv_file_path +
                                       f'/{config.candle_length}/highs and lows/{version}/lows rsi {version}/{ticker}_lows rsi {version}.csv',
                                       df, 2)

    # 1, first divergence low
    highs_price(config.high_low_range_b1, config.high_low_range_a1, 1)
    lows_price(config.high_low_range_b1, config.high_low_range_a1, 1)
    highs_rsi(config.high_low_range_b1, config.high_low_range_a1, 1)
    lows_rsi(config.high_low_range_b1, config.high_low_range_a1, 1)
    # 2, second divergence low
    highs_price(config.high_low_range_b2, config.high_low_range_a2, 2)
    lows_price(config.high_low_range_b2, config.high_low_range_a2, 2)
    highs_rsi(config.high_low_range_b2, config.high_low_range_a2, 2)
    lows_rsi(config.high_low_range_b2, config.high_low_range_a2, 2)
    # 3, supports and resistances for sell signals
    highs_price(config.high_low_range_3, config.high_low_range_3, 3)
    lows_price(config.high_low_range_3, config.high_low_range_3, 3)
    # 4, breakouts and reversals
    highs_price(config.high_low_range_4, config.high_low_range_4, 4)
    lows_price(config.high_low_range_4, config.high_low_range_4, 4)

    print('high_and_low complete')

def supports_and_resistances():
    for ticker in config.tickers:
        df_highs_price = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/3/highs price 3/{ticker}_highs price 3.csv',
            index_col=False, header=0)
        df_lows_price = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/3/lows price 3/{ticker}_lows price 3.csv',
            index_col=False, header=0)

        highs_list = df_highs_price['prices'].tolist()
        lows_list = df_lows_price['prices'].tolist()

        supports_and_resistances_list_nested_list = []

        for day in range(len(df_highs_price[config.datetime_str])):
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
        df.index = df_highs_price[config.datetime_str]
        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/supports and resistances/{ticker}_supports and resistances.csv',
            df, 1)

    print('support_and_resistance complete')

def breakouts():
    for ticker in config.tickers:
        df_raw = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
            index_col=False, header=0)
        dates = df_raw[str(config.datetime_str)].tolist()
        close = df_raw['close'].tolist()

        df_volatility = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/volatility/{ticker}_volatility.csv',
            index_col=False, header=0)
        volatility = df_volatility['volatility'].tolist()

        df_highs_price = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/4/highs price 4/{ticker}_highs price 4.csv',
            index_col=False, header=0)
        df_lows_price = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/4/lows price 4/{ticker}_lows price 4.csv',
            index_col=False, header=0)
        highs_list = df_highs_price['prices'].tolist()
        lows_list = df_lows_price['prices'].tolist()

        highs_lows_dict = {}
        key_counter = 0
        for count in range(len(dates)):
            if not math.isnan(highs_list[count]) or not math.isnan(lows_list[count]):
                highs_lows_dict[str(key_counter)] = [dates[count], close[count], volatility[count], highs_list[count], lows_list[count]]
                key_counter += 1

        df = pd.DataFrame()

        if config.long:
            unverified_breakout_data = []  # format: [date, breakout price, type]

            for count in range(len(highs_lows_dict) - 5):  # inverted head and shoulders
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count)][0]) < config.breakout_pattern_formation_expiration * 5:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4] > highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * config.head_and_shoulders_multiplier:  # checking 1 >> 3
                                                    if not math.isnan(highs_lows_dict[str(count + 4)][3]):  # checking 4 == high
                                                        if abs(highs_lows_dict[str(count + 2)][3] - highs_lows_dict[str(count + 4)][3]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * config.zone_multiplier:  # checking 2 == 4
                                                            if not math.isnan(highs_lows_dict[str(count + 5)][4]):  # checking that 5 == low
                                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 4)][3]:
                                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count + 2)][3], 'inverted head and shoulders'])
                                                                else:
                                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count + 4)][3], 'inverted head and shoulders'])

            for count in range(len(highs_lows_dict) - 3):  # double bottom
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 3)][0], highs_lows_dict[str(count)][0]) < config.breakout_pattern_formation_expiration * 3:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 3)][4]:  # checking 2 > 3
                                                    if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * config.zone_multiplier:  # checking 1 == 3
                                                        unverified_breakout_data.append([highs_lows_dict[str(count + 3)][0], highs_lows_dict[str(count + 2)][3], 'double bottom'])

            for count in range(len(highs_lows_dict) - 5):  # triple bottom
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count)][0]) < config.breakout_pattern_formation_expiration * 5:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 3)][4]:  # checking 2 > 3
                                                    if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * config.zone_multiplier:  # checking 1 == 3
                                                        if not math.isnan(highs_lows_dict[str(count + 4)][3]):  # checking 4 == high
                                                            if abs(highs_lows_dict[str(count + 2)][3] - highs_lows_dict[str(count + 4)][3]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * config.zone_multiplier:  # checking 1 == 3
                                                                if not math.isnan(highs_lows_dict[str(count + 5)][4]):  # checking that 5 == low
                                                                    if abs(highs_lows_dict[str(count + 3)][4] - highs_lows_dict[str(count + 5)][4]) < highs_lows_dict[str(count + 5)][1] * highs_lows_dict[str(count + 5)][2] * config.zone_multiplier:  # checking 3 == 5
                                                                        if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 5)][4]) < highs_lows_dict[str(count + 5)][1] * highs_lows_dict[str(count + 5)][2] * config.zone_multiplier:  # checking 1 == 5
                                                                            if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 4)][3]:
                                                                                unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count + 2)][3], 'triple bottom'])
                                                                            else:
                                                                                unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count + 4)][3], 'triple bottom'])

            for count in range(len(highs_lows_dict) - 4):  # bullish rectangle
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 4)][0], highs_lows_dict[str(count)][0]) < config.breakout_pattern_formation_expiration * 4:
                    if not math.isnan(highs_lows_dict[str(count)][4]):  # checking that 0 == low
                        if not math.isnan(highs_lows_dict[str(count + 1)][3]):  # checking that 1 == high
                            if not math.isnan(highs_lows_dict[str(count + 2)][4]):  # checking that 2 == low
                                if not math.isnan(highs_lows_dict[str(count + 3)][3]):  # checking that 3 == high
                                    if abs(highs_lows_dict[str(count + 1)][3] - highs_lows_dict[str(count + 3)][3]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * config.zone_multiplier:  # checking 1 == 3
                                        if not math.isnan(highs_lows_dict[str(count + 4)][4]):  # checking that 4 == low
                                            if abs(highs_lows_dict[str(count + 2)][4] - highs_lows_dict[str(count + 4)][4]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * config.zone_multiplier:  # checking 2 == 4
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
                for x in range(len(dates) - config.breakout_pattern_breakout_expiration):
                    if dates[x - bug_fix_counter] == unverified_breakout_data[breakout_check_list_counter][0]:
                        for date in dates[x - bug_fix_counter:x - bug_fix_counter + config.breakout_pattern_breakout_expiration]:
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
                        print(ticker, 'breakout')

                df = df.assign(bullish_breakout=bullish_breakout_list)
            else:
                empty_list = [None]*len(dates)
                df = df.assign(bullish_breakout=empty_list)

        if config.short:
            highs_lows_dict = {}
            key_counter = 0
            for count in range(len(dates)):
                if not math.isnan(highs_list[count]) or not math.isnan(lows_list[count]):
                    highs_lows_dict[str(key_counter)] = [dates[count], close[count], volatility[count], -lows_list[count], -highs_list[count]]
                    key_counter += 1

            unverified_breakout_data = []  # format: [date, breakout price, type]

            for count in range(len(highs_lows_dict) - 5):  # head and shoulders
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count)][0]) < config.breakout_pattern_formation_expiration * 5:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4] > highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * config.head_and_shoulders_multiplier:  # checking 1 >> 3
                                                    if not math.isnan(highs_lows_dict[str(count + 4)][3]):  # checking 4 == high
                                                        if abs(highs_lows_dict[str(count + 2)][3] - highs_lows_dict[str(count + 4)][3]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * config.zone_multiplier:  # checking 2 == 4
                                                            if not math.isnan(highs_lows_dict[str(count + 5)][4]):  # checking that 5 == low
                                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 4)][3]:
                                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], -highs_lows_dict[str(count + 2)][3], 'head and shoulders'])
                                                                else:
                                                                    unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], -highs_lows_dict[str(count + 4)][3], 'head and shoulders'])

            for count in range(len(highs_lows_dict) - 3):  # double top
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 3)][0], highs_lows_dict[str(count)][0]) < config.breakout_pattern_formation_expiration * 3:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 3)][4]:  # checking 2 > 3
                                                    if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * config.zone_multiplier:  # checking 1 == 3
                                                        unverified_breakout_data.append([highs_lows_dict[str(count + 3)][0], -highs_lows_dict[str(count + 2)][3], 'double top'])

            for count in range(len(highs_lows_dict) - 5):  # triple top
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 5)][0], highs_lows_dict[str(count)][0]) < config.breakout_pattern_formation_expiration * 5:
                    if not math.isnan(highs_lows_dict[str(count)][3]):  # checking that 0 == high
                        if not math.isnan(highs_lows_dict[str(count + 1)][4]):  # checking that 1 == low
                            if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 0 > 1
                                if not math.isnan(highs_lows_dict[str(count + 2)][3]):  # checking 2 == high
                                    if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 1)][4]:  # checking 2 > 1
                                        if highs_lows_dict[str(count)][3] > highs_lows_dict[str(count + 2)][3]:  # checking 0 > 2
                                            if not math.isnan(highs_lows_dict[str(count + 3)][4]):  # checking that 3 == low
                                                if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 3)][4]:  # checking 2 > 3
                                                    if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 3)][4]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * config.zone_multiplier:  # checking 1 == 3
                                                        if not math.isnan(highs_lows_dict[str(count + 4)][3]):  # checking 4 == high
                                                            if abs(highs_lows_dict[str(count + 2)][3] - highs_lows_dict[str(count + 4)][3]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * config.zone_multiplier:  # checking 1 == 3
                                                                if not math.isnan(highs_lows_dict[str(count + 5)][4]):  # checking that 5 == low
                                                                    if abs(highs_lows_dict[str(count + 3)][4] - highs_lows_dict[str(count + 5)][4]) < highs_lows_dict[str(count + 5)][1] * highs_lows_dict[str(count + 5)][2] * config.zone_multiplier:  # checking 3 == 5
                                                                        if abs(highs_lows_dict[str(count + 1)][4] - highs_lows_dict[str(count + 5)][4]) < highs_lows_dict[str(count + 5)][1] * highs_lows_dict[str(count + 5)][2] * config.zone_multiplier:  # checking 1 == 5
                                                                            if highs_lows_dict[str(count + 2)][3] > highs_lows_dict[str(count + 4)][3]:
                                                                                unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], -highs_lows_dict[str(count + 2)][3], 'triple top'])
                                                                            else:
                                                                                unverified_breakout_data.append([highs_lows_dict[str(count + 5)][0], -highs_lows_dict[str(count + 4)][3], 'triple top'])

            for count in range(len(highs_lows_dict) - 4):  # bearish rectangle
                if get_time_delta.get_time_delta(highs_lows_dict[str(count + 4)][0], highs_lows_dict[str(count)][0]) < config.breakout_pattern_formation_expiration * 4:
                    if not math.isnan(highs_lows_dict[str(count)][4]):  # checking that 0 == low
                        if not math.isnan(highs_lows_dict[str(count + 1)][3]):  # checking that 1 == high
                            if not math.isnan(highs_lows_dict[str(count + 2)][4]):  # checking that 2 == low
                                if not math.isnan(highs_lows_dict[str(count + 3)][3]):  # checking that 3 == high
                                    if abs(highs_lows_dict[str(count + 1)][3] - highs_lows_dict[str(count + 3)][3]) < highs_lows_dict[str(count + 3)][1] * highs_lows_dict[str(count + 3)][2] * config.zone_multiplier:  # checking 1 == 3
                                        if not math.isnan(highs_lows_dict[str(count + 4)][4]):  # checking that 4 == low
                                            if abs(highs_lows_dict[str(count + 2)][4] - highs_lows_dict[str(count + 4)][4]) < highs_lows_dict[str(count + 4)][1] * highs_lows_dict[str(count + 4)][2] * config.zone_multiplier:  # checking 2 == 4
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
                for x in range(len(dates) - config.breakout_pattern_breakout_expiration):
                    if dates[x - bug_fix_counter] == unverified_breakout_data[breakout_check_list_counter][0]:
                        for date in dates[x - bug_fix_counter:x - bug_fix_counter + config.breakout_pattern_breakout_expiration]:
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

        df.index = df_raw[config.datetime_str]
        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/breakouts/{ticker}_breakouts.csv',
            df, 1)

    print('breakout complete')

def divergences():
    for ticker in config.tickers:
        df_rsi = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv',
            index_col=False, header=0)
        dates = df_rsi[config.datetime_str].tolist()

        df_volatility = pd.read_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/volatility/{ticker}_volatility.csv',
            index_col=False, header=0)
        volatility_list = df_volatility['volatility'].tolist()

        df = pd.DataFrame()

        if config.long:
            df_lows_price_1 = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/1/lows price 1/{ticker}_lows price 1.csv',
                index_col=False, header=0)
            df_lows_rsi_1 = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/1/lows rsi 1/{ticker}_lows rsi 1.csv',
                index_col=False, header=0)
            df_lows_price_2 = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/2/lows price 2/{ticker}_lows price 2.csv',
                index_col=False, header=0)
            df_lows_rsi_2 = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/2/lows rsi 2/{ticker}_lows rsi 2.csv',
                index_col=False, header=0)

            dropna_df_lows_price_1 = df_lows_price_1.dropna()
            dropna_df_lows_rsi_1 = df_lows_rsi_1.dropna()
            dropna_df_lows_price_2 = df_lows_price_2.dropna()
            dropna_df_lows_rsi_2 = df_lows_rsi_2.dropna()

            low_price_list_1 = dropna_df_lows_price_1['prices'].tolist()
            low_price_dates_list_1 = dropna_df_lows_price_1[config.datetime_str].tolist()
            low_rsi_list_1 = dropna_df_lows_rsi_1['rsis'].tolist()
            low_rsi_dates_list_1 = dropna_df_lows_rsi_1[config.datetime_str].tolist()
            low_price_list_2 = dropna_df_lows_price_2['prices'].tolist()
            low_price_dates_list_2 = dropna_df_lows_price_2[config.datetime_str].tolist()
            low_rsi_list_2 = dropna_df_lows_rsi_2['rsis'].tolist()
            low_rsi_dates_list_2 = dropna_df_lows_rsi_2[config.datetime_str].tolist()

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
                volatility = volatility_list[get_time_delta.get_time_delta(matching_low_dates_2[x + 1], config.dates_list_compiled[0])]
                if (matching_low_prices_2[x] > matching_low_prices_2[x + 1]
                        and matching_low_rsis_2[x] < matching_low_rsis_2[x + 1]
                        and time_difference < config.divergence_expiration
                        and index_2[x] in index_1
                        and not math.isnan(volatility)):  # standard divergence
                    divergences.append(matching_low_dates_2[x + 1])
                    price_difference = ((matching_low_prices_2[x + 1] / matching_low_prices_2[x])-1)/volatility
                    rsi_difference = (matching_low_rsis_2[x + 1] - matching_low_rsis_2[x])
                    price_difference_quality = config.price_difference_quality_multiplier * -price_difference
                    rsi_difference_quality = config.rsi_difference_quality_multiplier * rsi_difference
                    rsi_level_quality = ((100-matching_low_rsis_2[x + 1]) + (100-matching_low_rsis_2[x])) * config.rsi_level_quality_multiplier
                    divergence_quality.append(price_difference_quality + rsi_difference_quality + rsi_level_quality)
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
                    print(ticker, 'divergence')
                else:
                    divergences_list.append(None)
                    divergence_quality_list.append(None)

            df = df.assign(bullish_divergences=divergences_list)
            df = df.assign(bullish_quality=divergence_quality_list)

        if config.short:
            df_highs_price_1 = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/1/highs price 1/{ticker}_highs price 1.csv',
                index_col=False, header=0)
            df_highs_rsi_1 = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/1/highs rsi 1/{ticker}_highs rsi 1.csv',
                index_col=False, header=0)
            df_highs_price_2 = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/2/highs price 2/{ticker}_highs price 2.csv',
                index_col=False, header=0)
            df_highs_rsi_2 = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/highs and lows/2/highs rsi 2/{ticker}_highs rsi 2.csv',
                index_col=False, header=0)

            dropna_df_highs_price_1 = df_highs_price_1.dropna()
            dropna_df_highs_rsi_1 = df_highs_rsi_1.dropna()
            dropna_df_highs_price_2 = df_highs_price_2.dropna()
            dropna_df_highs_rsi_2 = df_highs_rsi_2.dropna()

            high_price_list_1 = dropna_df_highs_price_1['prices'].tolist()
            high_price_dates_list_1 = dropna_df_highs_price_1[config.datetime_str].tolist()
            high_rsi_list_1 = dropna_df_highs_rsi_1['rsis'].tolist()
            high_rsi_dates_list_1 = dropna_df_highs_rsi_1[config.datetime_str].tolist()
            high_price_list_2 = dropna_df_highs_price_2['prices'].tolist()
            high_price_dates_list_2 = dropna_df_highs_price_2[config.datetime_str].tolist()
            high_rsi_list_2 = dropna_df_highs_rsi_2['rsis'].tolist()
            high_rsi_dates_list_2 = dropna_df_highs_rsi_2[config.datetime_str].tolist()

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
                volatility = volatility_list[get_time_delta.get_time_delta(matching_high_dates_2[x + 1], config.dates_list_compiled[0])]
                if (matching_high_prices_2[x] < matching_high_prices_2[x + 1]
                        and matching_high_rsis_2[x] > matching_high_rsis_2[x + 1]
                        and time_difference < config.divergence_expiration
                        and index_2[x] in index_1
                        and not math.isnan(volatility)):  # standard divergence
                    divergences.append(matching_high_dates_2[x + 1])
                    price_difference = ((matching_high_prices_2[x + 1] / matching_high_prices_2[x])-1)/volatility
                    rsi_difference = (matching_high_rsis_2[x + 1] - matching_high_rsis_2[x])
                    price_difference_quality = price_difference * config.price_difference_quality_multiplier
                    rsi_difference_quality = -rsi_difference * config.rsi_difference_quality_multiplier
                    rsi_level_quality = ((matching_high_rsis_2[x + 1]) + (matching_high_rsis_2[x])) * config.rsi_level_quality_multiplier
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

        df.index = df_rsi[config.datetime_str]
        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/divergences/{ticker}_divergences.csv',
            df, 1)

    print('divergences complete')

def stochastic_crossovers():
    for ticker in config.tickers:
        df = pd.DataFrame()

        if config.long:
            df_divergences = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/divergences/{ticker}_divergences.csv',
                index_col=False, header=0)
            dates = df_divergences[config.datetime_str].tolist()
            divergences_list = df_divergences['bullish_divergences'].tolist()
            divergence_quality_list = df_divergences['bullish_quality'].tolist()

            df_stochastic = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
                index_col=False, header=0)
            k_list = df_stochastic['k'].tolist()
            d_list = df_stochastic['d'].tolist()

            crossover_list = []
            quality_list = []
            divergence_day = math.nan
            quality_day = math.nan
            if not config.flexible_stochastic_cross_level:
                for day in range(len(dates)):
                    quality = divergence_quality_list[day]
                    if divergences_list[day] == 'long' and k_list[day] < config.stochastic_maximum:
                        divergence_day = day
                        quality_day = quality
                    if divergence_day <= day:
                        if divergence_day + config.stochastic_cross_expiration > day:
                            if k_list[day] > d_list[day] and k_list[day] > config.stochastic_cross_level:
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
                    if divergences_list[day] == 'long' and k_list[day] < config.stochastic_maximum:
                        divergence_day = day
                        quality_day = quality
                        inital_k = k_list[day]
                    if divergence_day <= day:
                        if divergence_day + config.stochastic_cross_expiration > day:
                            if k_list[day] > d_list[day] and k_list[day] > inital_k + config.stochastic_cross_level:
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

        if config.short:
            df_divergences = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/divergences/{ticker}_divergences.csv',
                index_col=False, header=0)
            dates = df_divergences[config.datetime_str].tolist()
            divergences_list = df_divergences['bearish_divergences'].tolist()
            divergence_quality_list = df_divergences['bearish_quality'].tolist()

            df_stochastic = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
                index_col=False, header=0)
            k_list = df_stochastic['k'].tolist()
            d_list = df_stochastic['d'].tolist()

            crossover_list = []
            quality_list = []
            divergence_day = math.nan
            quality_day = math.nan
            if not config.flexible_stochastic_cross_level:
                for day in range(len(dates)):
                    quality = divergence_quality_list[day]
                    if divergences_list[day] == 'short' and k_list[day] > (100-config.stochastic_maximum):
                        divergence_day = day
                        quality_day = quality
                    if divergence_day <= day:
                        if divergence_day + config.stochastic_cross_expiration > day:
                            if k_list[day] < d_list[day] and k_list[day] < (100-config.stochastic_cross_level):
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
                    if divergences_list[day] == 'short' and k_list[day] > (100-config.stochastic_maximum):
                        divergence_day = day
                        quality_day = quality
                        inital_k = k_list[day]
                    if divergence_day <= day:
                        if divergence_day + config.stochastic_cross_expiration > day:
                            if k_list[day] < d_list[day] and k_list[day] < inital_k - config.stochastic_cross_level:
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

        df.index = df_divergences[config.datetime_str]
        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/stochastic crossover divergence/{ticker}_stochastic crossover divergence.csv',
            df, 1)

    print('stochastic_crossover complete')

def signals():
    highs_and_lows()
    supports_and_resistances()
    breakouts()
    divergences()
    stochastic_crossovers()
