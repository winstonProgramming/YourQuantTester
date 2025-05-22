from datetime import datetime
import statistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config
import file_management
import get_time_delta

def back_test_func():
    benchmark_list = []

    def individual_stock_order_list():
        for ticker in config.tickers:
            # 1d
            df_raw = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/raw data/{ticker}_raw data.csv',
                index_col=False, header=0)
            open_list = df_raw['open'].tolist()
            del open_list[0:config.longest_indicator_length]
            close_list = df_raw['close'].tolist()
            del close_list[0:config.longest_indicator_length]
            date_list = df_raw[config.datetime_str].tolist()
            del date_list[0:config.longest_indicator_length]

            df_buy = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/buy signals/{ticker}_buy signals.csv',
                index_col=False, header=0)
            df_sell = pd.read_csv(
                config.stocks_csv_file_path + f'/{config.candle_length}/sell signals/{ticker}_sell signals.csv',
                index_col=False, header=0)

            if config.long:
                # buy
                bullish_buy_signal = df_buy['bullish_buy_signal'].tolist()
                del bullish_buy_signal[0:config.longest_indicator_length]
                bullish_quality = df_buy['bullish_quality'].tolist()
                del bullish_quality[0:config.longest_indicator_length]
                # sell
                bullish_sell_signal = df_sell['bullish_sell_signal'].tolist()
                del bullish_sell_signal[0:config.longest_indicator_length]
                bullish_sell_price = df_sell['bullish_sell_price'].tolist()
                del bullish_sell_price[0:config.longest_indicator_length]
            if config.short:
                # buy
                bearish_buy_signal = df_buy['bearish_buy_signal'].tolist()
                del bearish_buy_signal[0:config.longest_indicator_length]
                bearish_quality = df_buy['bearish_quality'].tolist()
                del bearish_quality[0:config.longest_indicator_length]
                # sell
                bearish_sell_signal = df_sell['bearish_sell_signal'].tolist()
                del bearish_sell_signal[0:config.longest_indicator_length]
                bearish_sell_price = df_sell['bearish_sell_price'].tolist()
                del bearish_sell_price[0:config.longest_indicator_length]

            order_list = []
            last_order = None
            ignore_sold = None
            for day in range(len(date_list)):
                if day != len(date_list) - 1:
                    if config.long and config.short:
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
                    elif config.long:
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
                    elif config.short:
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

            df = pd.DataFrame(order_list, columns=[config.datetime_str, 'type', 'price', 'quality'])
            file_management.export_csv(config.stocks_csv_file_path + f'/{config.candle_length}/order list/{ticker}_order list.csv', df, 1)

    def order_list_compiler():
        order_date_list = []
        order_ticker_list = []
        order_type_list = []
        order_price_list = []
        order_quality_list = []
        stock_order_number_list = []
        for ticker in config.tickers:
            order_list_df_individual = pd.read_csv(config.stocks_csv_file_path + f'/{config.candle_length}/order list/{ticker}_order list.csv', index_col=False, header=0)
            order_date_list_individual = order_list_df_individual[config.datetime_str].tolist()
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
        order_df = pd.DataFrame(list(zip(*order_nested_list)), columns=[config.datetime_str, 'ticker', 'type', 'price', 'quality', 'stock order number'])
        order_df = order_df.sort_values(by=[config.datetime_str, 'stock order number'], ascending=[True, True])
        order_df.to_csv('order list.csv', mode='w', index=False)

        if not config.order_size_based_on_money:
            order_date_list1 = order_df[config.datetime_str].tolist()
            order_ticker_list1 = order_df['ticker'].tolist()
            order_type_list1 = order_df['type'].tolist()
            order_price_list1 = order_df['price'].tolist()
            order_quality_list1 = order_df['quality'].tolist()
            stock_order_number_list1 = order_df['stock order number'].tolist()
            order_date_list2 = order_df[config.datetime_str].tolist()
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
                elif positions_counter == config.a and type_iterator == 'long bought' or positions_counter == config.a and type_iterator == 'short bought':
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
            order_df = pd.DataFrame(list(zip(*order_nested_list)), columns=[config.datetime_str, 'ticker', 'type', 'price', 'quality', 'stock order number'])
            order_df.to_csv('order list.csv', mode='w', index=False)

    def trade():
        def long_buy_func(money0, portfolio0, ticker0, price0, estimated_value0, date0, order_quality0, lowest_order_quality0, highest_order_quality0):
            if config.order_size_based_on_money and config.order_size_based_on_quality:
                trade_size_before_quality = (money/config.a)**config.b
                trade_size_before_quality_list.append(trade_size_before_quality)
                quality_difference = highest_order_quality0-lowest_order_quality0
                trade_size_quality_modifier = (1-(quality_difference/(order_quality0-config.lowest_order_quality+quality_difference+config.c)))**config.d
                trade_size_quality_modifier_list.append(trade_size_quality_modifier)
                trade_size0 = trade_size_before_quality*trade_size_quality_modifier
                trade_size_list.append(((1 - (quality_difference / (order_quality0 - config.lowest_order_quality + quality_difference + config.c))) ** config.d)*(((money/estimated_value0)/config.a)**config.b))
                portfolio0[ticker0] = [trade_size0/price0, price0, date0, ((1 - (quality_difference / (order_quality0 - config.lowest_order_quality + quality_difference + config.c))) ** config.d)*(((money/estimated_value0)/config.a)**config.b)]
            if config.order_size_based_on_money and not config.order_size_based_on_quality:
                trade_size0 = (money/config.a)**config.b
                trade_size_list.append(((money/estimated_value0)/config.a)**config.b)
                portfolio0[ticker0] = [trade_size0/price0, price0, date0, ((money/estimated_value0)/config.a)**config.b]
            if not config.order_size_based_on_money:
                trade_size0 = estimated_value0 / config.a
                trade_size_list.append(trade_size0)
                portfolio0[ticker0] = [trade_size0 / price0, price0, date0, trade_size0]
            money0 -= trade_size0
            trade_duration_list.append(None)
            money_list.append(money0)
            return money0, portfolio0

        def long_sell_func(money0, portfolio0, ticker0, price0, date0):
            trade_duration_list.append(config.get_time_delta.get_time_delta(date0, portfolio0.get(ticker0)[2]))
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
            if config.order_size_based_on_money and config.order_size_based_on_quality:
                trade_size_before_quality = (money/config.a)**config.b
                trade_size_before_quality_list.append(trade_size_before_quality)
                quality_difference = highest_order_quality0-lowest_order_quality0
                trade_size_quality_modifier = (1-(quality_difference/(order_quality0-config.lowest_order_quality+quality_difference+config.c)))**config.d
                trade_size_quality_modifier_list.append(trade_size_quality_modifier)
                trade_size0 = trade_size_before_quality*trade_size_quality_modifier
                trade_size_list.append(((1 - (quality_difference / (order_quality0 - config.lowest_order_quality + quality_difference + config.c))) ** config.d)*(((money/estimated_value0)/config.a)**config.b))
                portfolio0[ticker0] = [trade_size0/price0, price0, date0, ((1 - (quality_difference / (order_quality0 - config.lowest_order_quality + quality_difference + config.c))) ** config.d)*(((money/estimated_value0)/config.a)**config.b)]
            if config.order_size_based_on_money and not config.order_size_based_on_quality:
                trade_size0 = (money/config.a)**config.b
                trade_size_list.append(((money/estimated_value0)/config.a)**config.b)
                portfolio0[ticker0] = [trade_size0/price0, price0, date0, ((money/estimated_value0)/config.a)**config.b]
            if not config.order_size_based_on_money:
                trade_size0 = estimated_value0 / config.a
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
        order_date = order_list_df[config.datetime_str].tolist()
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
                money, portfolio = long_buy_func(money, portfolio, order_ticker[count], order_price[count], estimated_value, order_date[count], order_quality[count], config.lowest_order_quality, config.highest_order_quality)
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
                money, portfolio = short_buy_func(money, portfolio, order_ticker[count], order_price[count], estimated_value, order_date[count], order_quality[count], config.lowest_order_quality, config.highest_order_quality)
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
        order_dates_list = df_order[config.datetime_str].tolist()
        years_str_raw = str(datetime.strptime(order_dates_list[-1][:10], '%Y-%m-%d') - datetime.strptime(order_dates_list[0][:10], '%Y-%m-%d'))[:-7]
        years_str = []
        for letter in years_str_raw:
            try:
                if type(int(letter)) is int:
                    years_str.append(str(letter))
            except ValueError:
                pass
        years = int(''.join(years_str))/365

        if config.calculate_sharpe_ratio:
            df_order = pd.read_csv('order list.csv', index_col=False, header=0)
            order_dates_list = df_order[config.datetime_str].tolist()
            tickers_held_list = []
            trade_size_list = []
            price_list = []
            order_date_counter = 0

            for compiled_date in config.dates_list_compiled:
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

            portfolio_holdings_nested_list = [config.dates_list_compiled, tickers_held_list]
            order_df = pd.DataFrame(list(zip(*portfolio_holdings_nested_list)), columns=[config.datetime_str, 'tickers held'])
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
                yesterday = config.dates_list_compiled[x][0:10]
                today = config.dates_list_compiled[x + 1][0:10]
                if tickers_held is not None:
                    for y, ticker in enumerate(tickers_held):
                        df_raw = pd.read_csv(config.stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(config.candle_length, ticker), index_col=False, header=0)
                        date_list = df_raw[config.datetime_str].tolist()
                        close_list = df_raw['close'].tolist()
                        date_close_dict = {date_list[z]: close_list[z] for z in range(len(date_list))}
                        try:
                            if config.dates_list_compiled[x] in sell_dates and ticker in sell_tickers:
                                if sell_dates.index(config.dates_list_compiled[x]) == sell_tickers.index(ticker):
                                    sell_date_index = sell_dates.index(config.dates_list_compiled[x])
                                    portfolio_value += ((sell_prices[sell_date_index]/date_close_dict.get(config.dates_list_compiled[x]))-1) * trade_size_list[x][y]
                                else:
                                    portfolio_value += ((date_close_dict.get(config.dates_list_compiled[x + 1]) / date_close_dict.get(config.dates_list_compiled[x])) - 1) * trade_size_list[x][y]
                            else:
                                portfolio_value += ((date_close_dict.get(config.dates_list_compiled[x + 1]) / date_close_dict.get(config.dates_list_compiled[x])) - 1) * trade_size_list[x][y]
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

            sharpe_ratio = (yearly_return - config.risk_free_rate) / annual_volatility

            if config.calc_profit_odds:
                var_list = []

                for x in range(int(config.hours)):
                    var_list.append(np.random.uniform(statistics.geometric_mean(pct_list_geometric) - hourly_volatility, statistics.geometric_mean(pct_list_geometric) + hourly_volatility, config.sims))

                new_var_list = []
                for x in range(len(var_list[0])):
                    new_var_list_appender = []
                    for y in var_list:
                        new_var_list_appender.append(y[x])
                    new_var_list_appender = np.array(new_var_list_appender)
                    new_var_list.append(new_var_list_appender)

                profit_result = np.prod(new_var_list, axis=1)

                profit_odds_result = ((profit_result > config.profit_minimum).sum() / config.sims)

                plt.figure()
                plt.hist(profit_result, density=True, edgecolor='white')
                plt.axvline(config.profit_minimum, color='r')
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

        if config.long:
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

        if config.short:
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
        if config.order_size_based_on_quality:
            order_list_df = order_list_df.assign(trade_size_before_quality=trade_size_before_quality_list)
            order_list_df = order_list_df.assign(trade_size_quality_modifier=trade_size_quality_modifier_list)
        order_list_df = order_list_df.assign(trade_duration=trade_duration_list)
        order_list_df = order_list_df.assign(portfolio_value=estimated_value_list)
        order_list_df = order_list_df.assign(portfolio_size=portfolio_size_list)
        # order_list_df = order_list_df.assign(money=money_list)
        order_list_df.to_csv('order list.csv', mode='w', index=False)

        df_raw = pd.read_csv(config.stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(config.candle_length, config.tickers[0]), index_col=False, header=0)
        raw_dates = df_raw[config.datetime_str].tolist()
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

        if config.long:
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

        if config.short:
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
                print('start date: ' + config.start_date)
                print('end date: ' + config.end_date)
                print('model: ' + str((money-1)*100) + '%')
                print('average percentage of portfolio in the market: ' + str(avg_port*100) + '%')
                print('trades: ' + str(trades))
                print('trades won: ' + str(trades_won))
                print('trades lost: ' + str(trades_lost))
                if config.long and config.short:
                    print('long trades won: ' + str(long_trades_won))
                    print('long trades lost: ' + str(long_trades_lost))
                    print('short trades won: ' + str(short_trades_won))
                    print('short trades lost: ' + str(short_trades_lost))
                print('average profit per trade (discounting trade size): ' + str(average_profit_per_trade*100) + '%')
                print('average profit per trade (accounting trade size): ' + str(average_profit_per_trade_ats*100) + '%')
                if config.long and config.short:
                    print('average profit per long trade (discounting trade size): ' + str(long_average_profit_per_trade*100) + '%')
                    print('average profit per long trade (accounting trade size): ' + str(long_average_profit_per_trade_ats*100) + '%')
                    print('average profit per short trade (discounting trade size): ' + str(short_average_profit_per_trade*100) + '%')
                    print('average profit per short trade (accounting trade size): ' + str(short_average_profit_per_trade_ats*100) + '%')
                print('average profit of winning trade: ' + str((statistics.geometric_mean(avg_win_list)-1)*100) + '%')
                print('average profit of losing trade: ' + str((statistics.geometric_mean(avg_loss_list)-1)*100) + '%')
                if config.long and config.short:
                    print('average profit of winning long trade: ' + str((statistics.geometric_mean(long_avg_win_list)-1)*100) + '%')
                    print('average profit of losing long trade: ' + str((statistics.geometric_mean(long_avg_loss_list)-1)*100) + '%')
                    print('average profit of winning short trade: ' + str((statistics.geometric_mean(short_avg_win_list)-1)*100) + '%')
                    print('average profit of losing short trade: ' + str((statistics.geometric_mean(short_avg_loss_list)-1)*100) + '%')
                print('average trade duration: ' + str(statistics.mean(trade_duration_list_without_nones)))
                print('most positions at once: ' + str(sorted_portfolio_size_list[0]))
            except (IndexError, statistics.StatisticsError):
                print('lack of positions taken')
            if config.calculate_sharpe_ratio:
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
            if config.calc_profit_odds:
                print('chance of making more than a ' + str((config.profit_minimum-1) * 100) + '% return within ' + str(config.hours) + ' hours: ' + str(profit_odds_result*100) + '%')
            print('----------------------------------------')
            # print('benchmark: ' + str((statistics.mean(benchmark_list)-1)*100) + '%')
            # print('benchmark hours in market: ' + len(dates_list_compiled))

        print_data_func()

    individual_stock_order_list()
    order_list_compiler()
    trade()

    print('back_test complete')
