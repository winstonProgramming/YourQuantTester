import pandas as pd
import yfinance as yf

import config
import file_management

def scrape_data():
    if config.equity == 'stocks':
        for ticker in config.tickers:
            df = pd.DataFrame(yf.download(str(ticker),
                                          start=config.start_date,
                                          end=config.end_date,
                                          interval=config.candle_length))
            df.index.name = 'datetime'
            df.columns = [col[0].lower() for col in df.columns]
            file_management.export_csv(
                f'{config.stocks_csv_file_path}/{config.candle_length}/raw data/{ticker}_raw data.csv', df, 1)

    print('data scraping complete')
