import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
from download import download_raw_data
from tqdm import tqdm
import pymongo


# Database (internet connection needed)
load_dotenv()
BINANCE_CLIENT = Client(os.getenv('API_KEY'), os.getenv('API_SECRET'))
CLIENT = pymongo.MongoClient(os.getenv('MONGODB'))
DRAWDOWN_DB = CLIENT.bitget.traderDrawdown
TITLE = ["open_timestamp", "close_timestamp", "open", "high", "low", "close", "volume", "amount"]
DATASET_TITLE = ["input_timestamp", "high_1d", "low1d", "high_2d", "low2d", "high_3d", "low3d", "high_4d", "low4d", "high_5d", "low5d"]


# ========================================================================================================================
# Functions
# ========================================================================================================================
def get_time(timestamp: int = None):  # get current time, precise to seconds.
    if not timestamp:
        dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    else:
        dt1 = datetime.fromtimestamp(timestamp)

    dt2 = dt1.astimezone(timezone(timedelta(hours=8)))
    return dt2.strftime("%Y-%m-%d %H:%M:%S")

def round_timestamp(time_string):
    minute = int(int(time_string.split(':')[1]) / 5) * 5
    time_string = f"{time_string.split(':')[0]}:{minute}:00"
    return int(time.mktime(time.strptime(time_string, "%Y-%m-%d %H:%M:%S")))

def load_json(file_name):
    # load data for any usage
    data = []
    file = open(f"./{file_name}", "r")

    for line in file.readlines():
        dic = json.loads(line)
        data.append(dic)

    return data

def merge_kline_data(dict_data, timescale_in_min, output_file_name):
    try:
        if os.path.exists(f'./{output_file_name}.csv'):
            os.remove(f'./{output_file_name}.csv')

        for index in tqdm(range(0, len(dict_data), timescale_in_min)):
            if index+timescale_in_min-1 > len(dict_data)-1:
                new_line = [
                    dict_data[index]["timestamp"],
                    dict_data[-1]["closeTimestamp"],
                    dict_data[index]["open"],
                    max([ dict_data[i]["high"] for i in range(index, len(dict_data)) ]),
                    min([ dict_data[i]["low"] for i in range(index, len(dict_data)) ]),
                    dict_data[-1]["close"],
                    sum([ dict_data[i]["volume"] for i in range(index, len(dict_data)) ]),
                    sum([ dict_data[i]["amount"] for i in range(index, len(dict_data)) ]),
                ]
            else:
                new_line = [
                    dict_data[index]["timestamp"],
                    dict_data[index+timescale_in_min-1]["closeTimestamp"],
                    dict_data[index]["open"],
                    max([ dict_data[index+i]["high"] for i in range(timescale_in_min) ]),
                    min([ dict_data[index+i]["low"] for i in range(timescale_in_min) ]),
                    dict_data[index+timescale_in_min-1]["close"],
                    sum([ dict_data[index+i]["volume"] for i in range(timescale_in_min) ]),
                    sum([ dict_data[index+i]["amount"] for i in range(timescale_in_min) ]),
                ]
            new_line_df = pd.DataFrame(data=[new_line])
            new_line_df.to_csv(f'./{output_file_name}.csv', mode='a', index=False, header=False)

        return True
    except Exception as e:
        print(e)
        return False

def gen_csv():
    # generate kline data in larger time scale
    data_1min = load_json('1min_BTCUSDT.json')

    # 5 min k
    print('5 min')
    result = merge_kline_data(data_1min, 5, '5min_BTCUSDT')
    print('done' if result else 'failed')
    # 15 min k
    print('15 min')
    result = merge_kline_data(data_1min, 15, '15min_BTCUSDT')
    print('done' if result else 'failed')
    # 1 hour k
    print('1 hour')
    result = merge_kline_data(data_1min, 60, '1hour_BTCUSDT')
    print('done' if result else 'failed')
    # 4 hour k
    print('4 hour')
    result = merge_kline_data(data_1min, 240, '4hour_BTCUSDT')
    print('done' if result else 'failed')
    # 1 day k
    print('1 day')
    result = merge_kline_data(data_1min, 1440, '1day_BTCUSDT')
    print('done' if result else 'failed')

    return None

def update_data_from_binance():
    # just as its name, it gets data from binance api.
    symbols = ["BTCUSDT", "ETHUSDT"]

    for symbol in symbols:
        print("getData symbol : ", symbol)
        data = []
        file = open(f"./1min_{symbol}.json", "r")
        outfile = open(f"./1min_{symbol}.json", "a")

        for line in file.readlines():
            dic = json.loads(line)
            data.append(dic)

        startTimestamp = int(data[-1]["timestamp"]) + 60
        endTimestamp = int(time.mktime(time.strptime(get_time(), "%Y-%m-%d %H:%M:%S"))) - 60
        endTimestamp -= endTimestamp % 60

        try:
            kLines = BINANCE_CLIENT.get_historical_klines(
                symbol, Client.KLINE_INTERVAL_1MINUTE, str(startTimestamp), str(endTimestamp))
            # kLines = BINANCE_CLIENT.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, str(
            #     startTimestamp - 28800), str(endTimestamp - 28741))
        except Exception as e:
            print("an exception occured - {}".format(e))
            return False

        prekLine = None
        for k in kLines:
            kLine = {
                "timestamp": str(int(k[0] / 1000)),
                "date": str(datetime.fromtimestamp(int(k[0] / 1000))),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "closeTimestamp": str(int((k[6] + 1) / 1000)),
                "amount": float(k[7])
            }

            ## 補齊缺失的kline ##
            if prekLine == None:
                prekLine = kLine
            else:

                while int(kLine["timestamp"])-int(prekLine["timestamp"]) > 60:
                    print("missing kLine : ", prekLine)
                    prekLine["timestamp"] = str(int(prekLine["timestamp"])+60)
                    data = json.dumps(prekLine)
                    outfile.write(data)
                    outfile.write("\n")
                prekLine = kLine

            data = json.dumps(kLine)
            outfile.write(data)
            outfile.write("\n")

        outfile.close()
        file.close()

    return None

def get_trader_open_timestamp(symbol):
    if os.path.exists(f'./{symbol}_open_timestamp.csv'):
        os.remove(f'./{symbol}_open_timestamp.csv')
    print('query mongoDB')
    all_trader = DRAWDOWN_DB.find({})
    all_trader = list(all_trader)
    print('data loaded')

    trade_list = []
    for each_trader in tqdm(all_trader):
        for each_trade in each_trader["history"]:
            if symbol == each_trade["symbol"]:
                new_line = round_timestamp(each_trade["openDate"])
                if new_line > 1685203200 or new_line < 1648828800: # trades between 2022-04-02 to 2023-05-28
                    pass
                else:
                    trade_list.append(new_line)

    set_list = set(trade_list)
    unique_list = (set_list)
    unique_list_df = pd.DataFrame(data=unique_list, columns=['rounded_open_timestamp'])
    unique_list_df.to_csv(f'./{symbol}_open_timestamp.csv', mode='w', index=False, header=False)
    return None

def gen_dataset(symbol):
    if os.path.exists(f'./{symbol}_dataset_y.csv'):
        os.remove(f'./{symbol}_dataset_y.csv')
    if not os.path.exists(f'./dataset_x'):
        os.mkdir(f'./dataset_x')

    df_5m = pd.read_csv(f'./5min_{symbol}.csv', header=None, names=TITLE)
    df_15m = pd.read_csv(f'./15min_{symbol}.csv', header=None, names=TITLE)
    df_1h = pd.read_csv(f'./1hour_{symbol}.csv', header=None, names=TITLE)
    df_4h = pd.read_csv(f'./4hour_{symbol}.csv', header=None, names=TITLE)
    df_1d = pd.read_csv(f'./1day_{symbol}.csv', header=None, names=TITLE)
    df_open_time = pd.read_csv(f'./{symbol}_open_timestamp.csv', header=None, names=["rounded_open_timestamp"], dtype=int)
    # print(df_5m)
    print('data loaded')

    counter = 0
    for each_timestamp in tqdm(df_open_time["rounded_open_timestamp"]):
        # try to match every data
        match_index_5m = df_5m.index[df_5m["open_timestamp"] == each_timestamp].tolist()
        match_index_15m = df_15m.index[(df_15m["open_timestamp"] <= each_timestamp) & (df_15m["open_timestamp"]+900 > each_timestamp)].tolist()
        match_index_1h = df_1h.index[(df_1h["open_timestamp"] <= each_timestamp) & (df_1h["open_timestamp"]+3600 > each_timestamp)].tolist()
        match_index_4h = df_4h.index[(df_4h["open_timestamp"] <= each_timestamp) & (df_4h["open_timestamp"]+14400 > each_timestamp)].tolist()
        match_index_1d = df_1d.index[(df_1d["open_timestamp"] <= each_timestamp) & (df_1d["open_timestamp"]+86400 > each_timestamp)].tolist()
        
        if len(match_index_5m) == 1 and len(match_index_15m) == 1 and len(match_index_1h) == 1 and len(match_index_4h) == 1 and len(match_index_1d) == 1:
            match_index_5m = match_index_5m[0]
            match_index_15m = match_index_15m[0]
            match_index_1h = match_index_1h[0]
            match_index_4h = match_index_4h[0]
            match_index_1d = match_index_1d[0]

            if match_index_5m < 90 or match_index_15m < 90 or match_index_1h < 90 or match_index_4h < 90 or match_index_1d < 90:
                print([match_index_5m, match_index_15m, match_index_1h, match_index_4h, match_index_1d])
                break

            # syntax: df.iloc[row, column]
            x_data = np.concatenate((
                df_5m.iloc[match_index_5m-90:match_index_5m-1, [2,3,4,5,7]].T.to_numpy(),
                df_15m.iloc[match_index_15m-89:match_index_15m, [2,3,4,5,7]].T.to_numpy(),
                df_1h.iloc[match_index_1h-89:match_index_1h, [2,3,4,5,7]].T.to_numpy(),
                df_4h.iloc[match_index_4h-89:match_index_4h, [2,3,4,5,7]].T.to_numpy(),
                df_1d.iloc[match_index_1d-89:match_index_1d, [2,3,4,5,7]].T.to_numpy(),
            ), axis=0)

            append_last = np.array([[
                x_data[3][-1],
                df_5m['high'][match_index_5m],
                df_5m['low'][match_index_5m],
                df_5m['close'][match_index_5m],
                df_5m['volume'][match_index_5m],
                x_data[8][-1],
                max(df_5m['high'][((match_index_15m - 1) * 3) + 1 : match_index_5m+1]),
                min(df_5m['low'][((match_index_15m - 1) * 3) + 1 : match_index_5m+1]),
                df_5m['close'][match_index_5m],
                sum(df_5m['volume'][((match_index_15m - 1) * 3) + 1 : match_index_5m+1]),
                x_data[13][-1],
                max(df_5m['high'][((match_index_1h - 1) * 12) + 1 : match_index_5m+1]),
                min(df_5m['low'][((match_index_1h - 1) * 12) + 1 : match_index_5m+1]),
                df_5m['close'][match_index_5m],
                sum(df_5m['volume'][((match_index_1h - 1) * 12) + 1 : match_index_5m+1]),
                x_data[18][-1],
                max(df_5m['high'][((match_index_4h - 1) * 48) + 1 : match_index_5m+1]),
                min(df_5m['low'][((match_index_4h - 1) * 48) + 1 : match_index_5m+1]),
                df_5m['close'][match_index_5m],
                sum(df_5m['volume'][((match_index_4h - 1) * 48) + 1 : match_index_5m+1]),
                x_data[23][-1],
                max(df_5m['high'][((match_index_1d - 1) * 288) + 1 : match_index_5m+1]),
                min(df_5m['low'][((match_index_1d - 1) * 288) + 1 : match_index_5m+1]),
                df_5m['close'][match_index_5m],
                sum(df_5m['volume'][((match_index_1d - 1) * 288) + 1 : match_index_5m+1]),
            ]])

            x_data = np.concatenate((x_data, append_last.T), axis=1)

            with open(f"./dataset_x/BTCUSDT_{counter}.txt", "w") as f:
                np.savetxt(f, x_data, delimiter=',')
            counter += 1

            # ---------------------------------
            y_data = [
                max(df_5m['high'][match_index_5m:match_index_5m+288]),
                min(df_5m['low'][match_index_5m:match_index_5m+288]),
                max(df_5m['high'][match_index_5m:match_index_5m+576]),
                min(df_5m['low'][match_index_5m:match_index_5m+576]),
                max(df_5m['high'][match_index_5m:match_index_5m+864]),
                min(df_5m['low'][match_index_5m:match_index_5m+864]),
                max(df_5m['high'][match_index_5m:match_index_5m+1152]),
                min(df_5m['low'][match_index_5m:match_index_5m+1152]),
                max(df_5m['high'][match_index_5m:match_index_5m+1440]),
                min(df_5m['low'][match_index_5m:match_index_5m+1440]),
            ]
            y_data_df = pd.DataFrame(data=[y_data])
            y_data_df.to_csv(f'./{symbol}_dataset_y.csv', mode='a', index=False, header=False)

        else:
            # print(match_index_5m)
            raise Exception('wrong data!')

    return None

# ========================================================================================================================
# Main Function
# ========================================================================================================================
if __name__ == "__main__":
    print(f'\n{get_time()} start generating dataset')
    # run these function if it's your first time using this module
    # download_raw_data()
    # update_data_from_binance()

    # make csv
    # gen_csv()
    # get_trader_open_timestamp('BTCUSDT')

    # label data
    gen_dataset('BTCUSDT')

    print(f'{get_time()} done')