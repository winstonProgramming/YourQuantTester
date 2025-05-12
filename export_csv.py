import os

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
