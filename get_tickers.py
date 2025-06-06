import urllib.request
import urllib.parse
import re

import pandas as pd

import config
import scrape_data

def get_tickers():
    if config.equity == 'stocks' and config.scrape_data_bool:
        if config.nation == 'usa':
            sheet_id = "1U592Qmdg0GAGItmv96Ou6c_WvsFQ14QCkbKXLNMQOEs"
            sp500 = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
            pre_tickers_dot = list(sp500.loc[:])
            tickers0 = []
            tickers1 = []
            for i in range(len(pre_tickers_dot)):
                tickers0.append(pre_tickers_dot[i].replace(".", "_"))
                space_index = tickers0[i - 1].index(" ")
                tickers1.append(tickers0[i - 1][0:space_index])
            tickers1.remove('Symbol')
            tickers1.remove('Symbol')
            for t in tickers1:
                if t.index('_') != -1:
                    tickers1.remove(t)
            tickers_df0 = pd.DataFrame(tickers1)
            tickers_df0.to_csv('tickers', mode='w')
        elif config.nation == 'international':
            url = 'https://stockanalysis.com/list/biggest-companies/'
            headers = {'User-Agent': 'Mozilla/5.0'}

            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            resp_data = resp.read()

            tickers_list = re.findall(r's:"(\w{1,})', str(resp_data))
            tickers_list = tickers_list[:config.stock_number]
            tickers_list = list(set(tickers_list))

            def replace_ticker(t1, t2):
                try:
                    tickers_list.remove(t1)
                    tickers_list.append(t2)
                except ValueError:
                    pass

            def remove_ticker(t):
                try:
                    tickers_list.remove(t)
                    print('removed')
                except ValueError:
                    pass

            replace_ticker('BRK', 'BRK-B')
            replace_ticker('FWON', 'FWONK')
            remove_ticker('CRWV')

            tickers_df0 = pd.DataFrame(tickers_list, columns=['tickers'])
            tickers_df0.to_csv('tickers.csv', mode='w')

    if config.equity == 'stocks':
        tickers_df1 = pd.read_csv('tickers.csv', index_col=False, header=0)
        tickers = tickers_df1['tickers'].tolist()
        config.tickers = tickers

    if config.scrape_data_bool:
        scrape_data.scrape_data()

    dates_list_compiled = []
    for tick in config.tickers:
        df_raw_compiled = pd.read_csv(
            f'{config.stocks_csv_file_path}/{config.candle_length}/raw data/{tick}_raw data.csv',
            index_col=False, header=0)
        dates_list_for_compiler = df_raw_compiled[config.datetime_str].tolist()
        dates_list_compiled = dates_list_compiled + dates_list_for_compiler
    dates_list_compiled = list(set(dates_list_compiled))
    dates_list_compiled.sort()
    config.dates_list_compiled = dates_list_compiled
