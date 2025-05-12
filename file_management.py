import os

import config

def create_stocks_csv_folder():
    if not os.path.exists(config.stocks_csv_file_path):
        os.makedirs(config.stocks_csv_file_path)

    candle_path = f'{config.stocks_csv_file_path}/{config.candle_length}'
    if not os.path.exists(candle_path):
        os.makedirs(candle_path)

def export_csv(file_location, data_frame, directory_level):
    if directory_level == 1:
        try:
            data_frame.to_csv(file_location, mode='w')
        except OSError:
            try:
                os.mkdir('/'.join(map(str, file_location.split('/')[:-1])))
                data_frame.to_csv(file_location, mode='w')
            except OSError:
                os.mkdir('/'.join(map(str, file_location.split('/')[:-2])))
                os.mkdir('/'.join(map(str, file_location.split('/')[:-1])))
                data_frame.to_csv(file_location, mode='w')
    if directory_level == 2:
        try:
            data_frame.to_csv(file_location, mode='w')
        except OSError:
            try:
                os.mkdir('/'.join(map(str, file_location.split('/')[:-1])))
                data_frame.to_csv(file_location, mode='w')
            except OSError:
                try:
                    os.mkdir('/'.join(map(str, file_location.split('/')[:-2])))
                    os.mkdir('/'.join(map(str, file_location.split('/')[:-1])))
                    data_frame.to_csv(file_location, mode='w')
                except OSError:
                    os.mkdir('/'.join(map(str, file_location.split('/')[:-3])))
                    os.mkdir('/'.join(map(str, file_location.split('/')[:-2])))
                    os.mkdir('/'.join(map(str, file_location.split('/')[:-1])))
                    data_frame.to_csv(file_location, mode='w')
