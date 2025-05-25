import math
from itertools import permutations, product

import pandas as pd

import config
import file_management

def buy_signals():
    def flatten(container):
        for i2 in container:
            if isinstance(i2, (list, tuple)):
                for j in flatten(i2):
                    yield j
            else:
                yield i2

    buy_order_permutations = []
    product_list = []
    for unique_order in set(config.buy_signal_order_dict.values()):
        key_list = []
        for key, value in config.buy_signal_order_dict.items():
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

    for ticker in config.tickers:
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

        if config.long:
            df_rsi = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv',
                index_col=False, header=0)
            rsi_quality_list = df_rsi['bullish_quality'].tolist()

            df_ema = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/ema/{ticker}_ema.csv',
                index_col=False, header=0)
            ema_quality_list = df_ema['bullish_quality'].tolist()

            df_stochastic = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
                index_col=False, header=0)
            stochastic_quality_list = df_stochastic['bullish_quality'].tolist()

            if not config.stochastic_crossover:
                df_divergences = pd.read_csv(
                    config.stocks_csv_file_path + f'/{config.candle_length}/divergences/{ticker}_divergences.csv',
                    index_col=False, header=0)
                dates = df_divergences[config.datetime_str].tolist()
                divergences = df_divergences['bullish_divergences'].tolist()
                divergence_quality_list = df_divergences['bullish_quality'].tolist()
            else:
                df_divergences = pd.read_csv(
                    config.stocks_csv_file_path + f'/{config.candle_length}/stochastic crossover divergence/{ticker}_stochastic crossover divergence.csv',
                    index_col=False, header=0)
                dates = df_divergences[config.datetime_str].tolist()
                divergences = df_divergences['bullish_divergences'].tolist()
                divergence_quality_list = df_divergences['bullish_quality'].tolist()

            df_breakouts = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/breakouts/{ticker}_breakouts.csv',
                index_col=False, header=0)
            breakouts = df_breakouts['bullish_breakout'].tolist()

            df_candle_stick = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/candle stick/{ticker}_candle stick.csv',
                index_col=False, header=0)
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

                if bought and quality >= config.quality_minimum and not omission:
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
                            for day2 in range(config.buy_signal_expiration_list[count]+1):
                                buy_signal_expiration_modifier = sum(config.buy_signal_expiration_list[count+1:]) - reserve_days
                                if buy_signal_data[day1 - day2 - buy_signal_expiration_modifier] == 'long':
                                    reserve_days = config.buy_signal_expiration_list[count] - day2
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

        if config.short:
            df_rsi = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv',
                index_col=False, header=0)
            rsi_quality_list = df_rsi['bearish_quality'].tolist()

            df_ema = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/ema/{ticker}_ema.csv',
                index_col=False, header=0)
            ema_quality_list = df_ema['bearish_quality'].tolist()

            df_stochastic = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
                index_col=False, header=0)
            stochastic_quality_list = df_stochastic['bearish_quality'].tolist()

            if not config.stochastic_crossover:
                df_divergences = pd.read_csv(
                    config.stocks_csv_file_path + f'/{config.candle_length}/divergences/{ticker}_divergences.csv',
                    index_col=False, header=0)
                dates = df_divergences[config.datetime_str].tolist()
                divergences = df_divergences['bearish_divergences'].tolist()
                divergence_quality_list = df_divergences['bearish_quality'].tolist()
            else:
                df_divergences = pd.read_csv(
                    config.stocks_csv_file_path + f'/{config.candle_length}/stochastic crossover divergence/{ticker}_stochastic crossover divergence.csv',
                    index_col=False, header=0)
                dates = df_divergences[config.datetime_str].tolist()
                divergences = df_divergences['bearish_divergences'].tolist()
                divergence_quality_list = df_divergences['bearish_quality'].tolist()

            df_breakouts = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/breakouts/{ticker}_breakouts.csv',
                index_col=False, header=0)
            breakouts = df_breakouts['bearish_breakout'].tolist()

            df_candle_stick = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/candle stick/{ticker}_candle stick.csv',
                index_col=False, header=0)
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

                if bought and quality >= config.quality_minimum and not omission:
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
                            for day2 in range(config.buy_signal_expiration_list[count]+1):
                                buy_signal_expiration_modifier = sum(config.buy_signal_expiration_list[count+1:]) - reserve_days
                                if buy_signal_data[day1 - day2 - buy_signal_expiration_modifier] == 'short':
                                    reserve_days = config.buy_signal_expiration_list[count] - day2
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

        df.index = df_divergences[config.datetime_str]
        file_management.export_csv(config.stocks_csv_file_path + f'/{config.candle_length}/buy signals/{ticker}_buy signals.csv', df, 1)

    print('buy_signal complete')

def sell_signals():
    for ticker in config.tickers:
        df = pd.DataFrame()

        if config.long:
            sup_res_df = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/supports and resistances/{ticker}_supports and resistances.csv',
                index_col=False, header=0)
            sup_res_raw = sup_res_df['supports and resistances'].tolist()

            df_raw = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
                index_col=False, header=0)
            dates = df_raw[config.datetime_str].tolist()
            high_list = df_raw['high'].tolist()
            close_list = df_raw['close'].tolist()

            df_buy_signals = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/buy signals/{ticker}_buy signals.csv',
                index_col=False, header=0)
            buy_signals = df_buy_signals['bullish_buy_signal'].tolist()

            if config.sell_signal_indicator_type_1 == 'rsi':
                df_rsi = pd.read_csv(
                    config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv',
                    index_col=False, header=0)
                sell_signal_indicator_list = df_rsi[str(config.rsi_length)].tolist()
            elif config.sell_signal_indicator_type_1 == 'k':
                df_stochastic = pd.read_csv(
                    config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
                    index_col=False, header=0)
                sell_signal_indicator_list = df_stochastic['k'].tolist()
            elif config.sell_signal_indicator_type_1 == 'd':
                df_stochastic = pd.read_csv(
                    config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
                    index_col=False, header=0)
                sell_signal_indicator_list = df_stochastic['d'].tolist()

            sup_res = []
            for x in sup_res_raw:
                sup_res_appender = x.split('_')
                sup_res_appender.remove('')
                sup_res.append(sup_res_appender)

            sell_signals_dict = {}
            for x, sell_signals in enumerate(config.sell_signals_nested_list):
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
                for x, sell_signals in enumerate(config.sell_signals_nested_list):
                    for y, sell_signal in enumerate(sell_signals):
                        sell_signals_dict[x][y] = [False, None, math.nan]  # sell signal, sell price, sell day

                        if sell_signal == 'sell signal indicator 1':
                            sell_signal_active_today_1 = False
                            if not config.sell_signal_indicator_flexible_1:
                                if buy_signals[day] == 'long':
                                    sell_signal_indicator_bought_1 = True
                                if sell_signal_indicator_bought_1 and sell_signal_indicator_list[day] >= config.sell_signal_indicator_value_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif config.sell_signal_simultaneous_fulfillment_1:
                                    sell_signal_active_today_1 = True
                            else:
                                if buy_signals[day] == 'long':
                                    sell_signal_indicator_buy_value_1 = sell_signal_indicator_list[day]
                                if sell_signal_indicator_list[day] - sell_signal_indicator_buy_value_1 >= config.sell_signal_indicator_value_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif config.sell_signal_simultaneous_fulfillment_1:
                                    sell_signal_active_today_1 = True
                            if config.sell_signal_simultaneous_fulfillment_1 and not sell_signal_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'support resistance 1':
                            support_resistance_active_today_1 = False
                            if buy_signals[day] == 'long':
                                for sr in sup_res[day]:
                                    if float(sr) > close_list[day] * config.support_resistance_resistance_minimum_distance_1:
                                        support_resistance_resistance_price_1 = float(sr)
                                        break
                                for sr in reversed(sup_res[day]):
                                    if float(sr) < close_list[day] * config.support_resistance_support_minimum_distance_1:
                                        support_resistance_support_price_1 = float(sr)
                                        break
                            if not config.support_resistance_resistance_high_1:
                                if close_list[day] >= support_resistance_resistance_price_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif config.support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            else:
                                if high_list[day] >= support_resistance_resistance_price_1:
                                    sell_signals_dict[x][y] = [True, str(support_resistance_resistance_price_1) + ' nc', day]
                                elif config.support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            if close_list[day] <= support_resistance_support_price_1:
                                sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                            elif config.support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            if config.support_resistance_simultaneous_fulfillment_1 and not support_resistance_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'artificial margin 1':
                            artificial_margin_active_today_1 = False
                            if buy_signals[day] == 'long':
                                artificial_margin_take_profit_price_1 = close_list[day] * config.artificial_margin_take_profit_1
                                artificial_margin_stop_loss_price_1 = close_list[day] * config.artificial_margin_stop_loss_1
                            if not config.artificial_margin_take_profit_high_1:
                                if close_list[day] >= artificial_margin_take_profit_price_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif config.artificial_margin_simultaneous_fulfillment_1:
                                        artificial_margin_active_today_1 = True
                            else:
                                if high_list[day] >= artificial_margin_take_profit_price_1:
                                    sell_signals_dict[x][y] = [True, str(artificial_margin_take_profit_price_1) + ' nc', day]
                                elif config.artificial_margin_simultaneous_fulfillment_1:
                                        artificial_margin_active_today_1 = True
                            if close_list[day] <= artificial_margin_stop_loss_price_1:
                                sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                            elif config.artificial_margin_simultaneous_fulfillment_1:
                                    artificial_margin_active_today_1 = True
                            if config.artificial_margin_simultaneous_fulfillment_1 and not artificial_margin_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'sell time 1':
                            if buy_signals[day] == 'long':
                                sell_time_buy_date_1 = day
                            if day - sell_time_buy_date_1 >= config.sell_time_value_1:
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

        if config.short:
            sup_res_df = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/supports and resistances/{ticker}_supports and resistances.csv',
                index_col=False, header=0)
            sup_res_raw = sup_res_df['supports and resistances'].tolist()

            df_raw = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
                index_col=False, header=0)
            dates = df_raw[config.datetime_str].tolist()
            low_list = df_raw['low'].tolist()
            close_list = df_raw['close'].tolist()

            df_buy_signals = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/buy signals/{ticker}_buy signals.csv',
                index_col=False, header=0)
            buy_signals = df_buy_signals['bearish_buy_signal'].tolist()

            if config.sell_signal_indicator_type_1 == 'rsi':
                df_rsi = pd.read_csv(
                    config.stocks_csv_file_path + f'/{config.candle_length}/rsi/{ticker}_rsi.csv',
                    index_col=False, header=0)
                sell_signal_indicator_list = df_rsi[str(config.rsi_length)].tolist()
            elif config.sell_signal_indicator_type_1 == 'k':
                df_stochastic = pd.read_csv(
                    config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
                    index_col=False, header=0)
                sell_signal_indicator_list = df_stochastic['k'].tolist()
            elif config.sell_signal_indicator_type_1 == 'd':
                df_stochastic = pd.read_csv(
                    config.stocks_csv_file_path + f'/{config.candle_length}/stochastic/{ticker}_stochastic.csv',
                    index_col=False, header=0)
                sell_signal_indicator_list = df_stochastic['d'].tolist()

            sup_res = []
            for x in sup_res_raw:
                sup_res_appender = x.split('_')
                sup_res_appender.remove('')
                sup_res.append(sup_res_appender)

            sell_signals_dict = {}
            for x, sell_signals in enumerate(config.sell_signals_nested_list):
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
                for x, sell_signals in enumerate(config.sell_signals_nested_list):
                    for y, sell_signal in enumerate(sell_signals):
                        sell_signals_dict[x][y] = [False, None, math.nan]  # sell signal, sell price, sell day

                        if sell_signal == 'sell signal indicator 1':
                            sell_signal_active_today_1 = False
                            if not config.sell_signal_indicator_flexible_1:
                                if buy_signals[day] == 'short':
                                    sell_signal_indicator_bought_1 = True
                                if sell_signal_indicator_bought_1 and sell_signal_indicator_list[day] <= (100 - config.sell_signal_indicator_value_1):
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif config.sell_signal_simultaneous_fulfillment_1:
                                    sell_signal_active_today_1 = True
                            else:
                                if buy_signals[day] == 'short':
                                    sell_signal_indicator_buy_value_1 = sell_signal_indicator_list[day]
                                if sell_signal_indicator_list[day] - sell_signal_indicator_buy_value_1 <= (100 - config.sell_signal_indicator_value_1):
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif config.sell_signal_simultaneous_fulfillment_1:
                                    sell_signal_active_today_1 = True
                            if config.sell_signal_simultaneous_fulfillment_1 and not sell_signal_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'support resistance 1':
                            support_resistance_active_today_1 = False
                            if buy_signals[day] == 'short':
                                for sr in sup_res[day]:
                                    if float(sr) > close_list[day] * (1/config.support_resistance_support_minimum_distance_1):
                                        support_resistance_resistance_price_1 = float(sr)
                                        break
                                for sr in reversed(sup_res[day]):
                                    if float(sr) < close_list[day] * (1/config.support_resistance_resistance_minimum_distance_1):
                                        support_resistance_support_price_1 = float(sr)
                                        break
                            if not config.support_resistance_resistance_high_1:
                                if close_list[day] <= support_resistance_support_price_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif config.support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            else:
                                if low_list[day] <= support_resistance_support_price_1:
                                    sell_signals_dict[x][y] = [True, str(support_resistance_support_price_1) + ' nc', day]
                                elif config.support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            if close_list[day] >= support_resistance_resistance_price_1:
                                sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                            elif config.support_resistance_simultaneous_fulfillment_1:
                                    support_resistance_active_today_1 = True
                            if config.support_resistance_simultaneous_fulfillment_1 and not support_resistance_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'artificial margin 1':
                            artificial_margin_active_today_1 = False
                            if buy_signals[day] == 'short':
                                artificial_margin_take_profit_price_1 = close_list[day] * (1/config.artificial_margin_take_profit_1)
                                artificial_margin_stop_loss_price_1 = close_list[day] * (1/config.artificial_margin_stop_loss_1)

                            if not config.artificial_margin_take_profit_high_1:
                                if close_list[day] <= artificial_margin_take_profit_price_1:
                                    sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                                elif config.artificial_margin_simultaneous_fulfillment_1:
                                        artificial_margin_active_today_1 = True
                            else:
                                if low_list[day] <= artificial_margin_take_profit_price_1:
                                    sell_signals_dict[x][y] = [True, str(artificial_margin_take_profit_price_1) + ' nc', day]
                                elif config.artificial_margin_simultaneous_fulfillment_1:
                                    artificial_margin_active_today_1 = True
                            if close_list[day] >= artificial_margin_stop_loss_price_1:
                                sell_signals_dict[x][y] = [True, str(close_list[day]) + ' cl', day]
                            elif config.artificial_margin_simultaneous_fulfillment_1:
                                artificial_margin_active_today_1 = True
                            if config.artificial_margin_simultaneous_fulfillment_1 and not artificial_margin_active_today_1:
                                sell_signals_dict[x][y] = [False, None, math.nan]

                        elif sell_signal == 'sell time 1':
                            if buy_signals[day] == 'short':
                                sell_time_buy_date_1 = day
                            if day - sell_time_buy_date_1 >= config.sell_time_value_1:
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

        df.index = df_raw[config.datetime_str]
        file_management.export_csv(
            config.stocks_csv_file_path + f'/{config.candle_length}/sell signals/{ticker}_sell signals.csv', df, 1)

    print('sell_signal complete')

def trade_signals():
    buy_signals()
    sell_signals()
