import urllib.request
import urllib.parse
import re

import pandas as pd
import yfinance as yf

import config
import file_management


def get_tickers():
    def scrape_data():
        if config.equity == 'stocks':
            for ticker in config.tickers:
                df = pd.DataFrame(yf.download(str(ticker),
                                              start=config.start_date,
                                              end=config.end_date,
                                              interval=config.candle_length))
                df.index.name = 'datetime'
                df.columns = [col[0].lower() for col in df.columns]
                file_management.export_csv(f'{config.stocks_csv_file_path}/{config.candle_length}/raw data/{ticker}_raw data.csv', df, 1)

        print('raw scraping complete')

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

            try:
                tickers_list.remove('BRK')
                tickers_list.append('BRK-B')
            except ValueError:
                pass

            tickers_df0 = pd.DataFrame(tickers_list, columns=['tickers'])
            tickers_df0.to_csv('tickers.csv', mode='w')
    if config.equity == 'stocks':
        tickers_df1 = pd.read_csv('tickers.csv', index_col=False, header=0)
        tickers = tickers_df1['tickers'].tolist()
        config.tickers = tickers
    dates_list_compiled = []
    if config.scrape_data_bool:
        scrape_data()
    for tick in config.tickers:
        df_raw_compiled = pd.read_csv(f'{config.stocks_csv_file_path}/{config.candle_length}/raw data/{tick}_raw data.csv', index_col=False, header=0)
        dates_list_for_compiler = df_raw_compiled[config.datetime_str].tolist()
        dates_list_compiled = dates_list_compiled + dates_list_for_compiler
    dates_list_compiled = list(set(dates_list_compiled))
    dates_list_compiled.sort()
    config.dates_list_compiled = dates_list_compiled
