from dataManager import DataManager
from filter import Filter

if __name__ == "__main__":

    folder = "D:/Data/EOD"
    exchanges = ["F", "XETRA"]
    types = ["ETF"]
    timeframes = [1440]
    features = ["close"]
    startdate = "2017-01-01"
    enddate = "2021-06-03"

    dm = DataManager(folder=folder, verbose=True)
    filt = Filter(folder=folder)
    dm.pivot_table(types=types, timeframes=timeframes, features=features, remainder=features[0])
    df = filt.filter_pivot_table(typ=types[0], timeframe=timeframes[0], feature=features[0], startdate=startdate, enddate=enddate)
    df.to_parquet(f"{folder}/Datasets/ETF_{timeframes[0]}_{features[0]}_pivot_filt.parquet", compression="gzip")