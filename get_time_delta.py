import config

def get_time_delta(time_value_1, time_value_2):  # later date, earlier date
    pos_1 = config.dates_list_compiled.index(time_value_1)
    pos_2 = config.dates_list_compiled.index(time_value_2)
    date_counter = pos_1 - pos_2
    return date_counter
