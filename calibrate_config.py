import config

def clean():
    config.rsi_lengths.sort()

def calculate_variables():
    # config.longest_indicator_list = [rsi_length, ema_length, fastk_period, slowk_period, slowd_period, rolling_volatility_length, pre_earnings_date_omission]
    config.longest_indicator_list = [config.rsi_lengths[-1],
                                     config.ema_length,
                                     config.fastk_period,
                                     config.slowk_period,
                                     config.slowd_period,
                                     config.rolling_volatility_length]
    config.longest_indicator_list.sort()
    config.longest_indicator_length = config.longest_indicator_list[-1]

    config.buy_signal_order_dict = dict(sorted(config.buy_signal_order_dict.items(), key=lambda item: item[1]))  # may break if vals are greater than one digit
    # config.buy_signal_expiration_dict = OrderedDict(sorted(config.buy_signal_expiration_dict.items()))
    config.buy_signal_expiration_list.append(0)

def calibrate_config():
    clean()
    calculate_variables()
