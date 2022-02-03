import csv
import requests
import pandas as pd

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=15min&slice=year1month1&apikey=demo'

api_key = "2PMR84MYY2GUEOPN"

def request_stock_price_hist(symbol, interval = None, y = 1, m = 1):

    CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=%s&interval=%s&slice=year%dmonth%d&apikey=%s' % (symbol, interval, y, m, api_key)

    with requests.Session() as s:
        download = s.get(CSV_URL)

        #In JSON format
        key = 'Time Series (%s)' % interval
        decoded_content = download.content.decode('utf-8')
        reader = csv.reader(decoded_content.splitlines(), delimiter='\n')
        headers = reader.__next__()
        headers = headers[0].split(',')
        
        df = pd.DataFrame(columns=list(range(6)))
        i = 0
        for row in reader:
            row = row[0].split(",")

            a_series = pd.Series(row, index = df.columns)
            df = df.append(a_series, ignore_index=True) 
        df.columns = headers
        return df
    
        decoded_content = download.content.decode('utf-8')
        reader = csv.reader(decoded_content.splitlines(), delimiter='\n')
        my_list = list(reader)

        
symbols = ['MMM', 'UAL', 'NFLX']
interval = "1min"

output = pd.DataFrame(columns=['1'])
for symbol in symbols:
    print(symbol)
    for year in range(1, 2 + 1):
        for month in range(1, 12 + 1):
            if output.empty:
                output = request_stock_price_hist(symbol, interval, year, month)
            else:
                output = pd.concat([output, request_stock_price_hist(symbol, interval, year, month)])
            print(year, month, flush=True)

    file_name = "%s_%s_2years.csv" % (symbol, interval)
    output.to_csv(file_name, sep=',', encoding='utf-8')

exit()


# for y in range(1, 2 + 1):
#     for m in range (1, 12 + 1):
#         print(

# CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=%s&apikey=%s' % (symbol, interval, api_key)