import pandas as pd
from data_manager import DataManager

if __name__ == "__main__":

    root = "D:/Tiingo/"
    types = ["ETF"]
    timeframes = [1440]
    return_periods = [1, 3, 5, 7, 14, 21, 30]
    start = pd.to_datetime("2012-06-01")
    end = pd.to_datetime("2021-08-10")

    selection = ["LYPG", "QDVR", "EL4C", "USPY", "T3KE", "IQQT", "ETLH", "LYMD", "LCTR", "DBXH",
                "WTEJ", "SPF1", "QDVE", "EMQQ", "EKUS", "DBPG", "BLUM", "WTI2", "BATE", "XDEQ",
                "IQQH", "XAIX", "XMLH", "2B78", "LYPA", "BNXG", "ROAI", "GENY", "ELCR", "DRUP", 
                "DX2D", "SNAZ", "18MU", "FRC4", "LYQ7", "SC0P", "IQQL", "QDVB"]

    dm = DataManager(root=root, plot=False)
    dm.create_dataset(types=types, timeframes=timeframes, n_rand=None)
    # dm.pivot_table(types=types,
    #                 timeframes=timeframes, 
    #                 start=start, end=end, 
    #                 remainder="close", 
    #                 selection=selection)
    dm.synchronize_dataset(types=types, timeframes=timeframes, start=start, end=end)
    # dm.combine_pivot_tables(types=["ETF"], timeframe=timeframes[0])
    
    
