import gdown
import os


def download_raw_data():
    # 2023-04-25
    BTCURL = "https://drive.google.com/u/0/uc?id=1Kn8oUOjch5eoupFqPQ7zt6EWm-0D2Kfi&export=download"
    ETHURL = "https://drive.google.com/u/0/uc?id=1LvH0x2d2si9kmHfC7qr_50KgWDsT593K&export=download"

    gdown.download(BTCURL, output="1min_BTCUSDT.json")
    gdown.download(ETHURL, output="1min_ETHUSDT.json")
    print("Finished.")

    return None