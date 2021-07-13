import pandas as pd
from data_manager import DataManager

if __name__ == "__main__":

    folder = "D:/EOD/"
    types = ["ETF"]
    timeframes = [1440]
    return_periods = [1, 3, 5, 7, 14, 21, 30]
    start = pd.to_datetime("2020-05-01")
    end = pd.to_datetime("2021-07-12")

    dm = DataManager(folder=folder, plot=True)
    # dm.create_dataset(types=types, timeframes=timeframes,
    #                  indicators=False,
    #                  normalize=True,
    #                  historic_returns=None,
    #                  forward_returns=None,
    #                  n_rand=None)
    dm.pivot_table(types=types, timeframes=timeframes, start=start, end=end, remainder="close")
    # dm.synchronize_dataset(types=types, timeframes=timeframes, start=start, end=end)
    # dm.combine_pivot_tables(types=["ETF"], timeframe=timeframes[0])
    
    
