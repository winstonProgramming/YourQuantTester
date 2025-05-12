def raw_scraping_func():
    if equity == 'stocks':
        for ticker in tickers:
            df = pd.DataFrame(yf.download(str(ticker), start=start_date, end=end_date, interval=candle_length))
            df.index.name = 'datetime'
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
            export_csv_func(stocks_csv_file_path + '/{}/raw data/{}_raw data.csv'.format(candle_length, ticker), df, 1)

    print('raw_func complete')