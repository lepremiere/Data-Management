import os
import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm
from datetime import datetime
from lib.baseClass import BaseClass

class DataManager(BaseClass):

    def __init__(self, folder, verbose=True):
        super().__init__(verbose=verbose)
        self.path = folder
        self.verbose = verbose
    
    def create_dataset(self, types: list, timeframes: list):

        self.create_folder(folder="Datasets")
        at = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()

        for type in types:
            for timeframe in timeframes:
                l = []
                ind = np.logical_and(at.loc[:, "type"] == type, at.loc[:, "timeframe"] == timeframe)
                ticker_list = at.loc[ind]
                self.print(msg=f"Gathering:\t{type}, timeframe: {timeframe}...")

                for i in tqdm(range(len(ticker_list))):
                    symbol, _, _, _, _, path = ticker_list.iloc[i].to_numpy()
                    temp_df = pl.read_csv(path).to_pandas()
                    temp_df.insert(loc=1, column="symbol", value=symbol)
                    temp_df.drop_duplicates(subset=['date'], inplace=True)
                    l.append(temp_df)

                df = pd.concat(l)  
                df.to_parquet(f"{self.path}/DataSets/{type}_{timeframe}.parquet", compression="gzip")
                self.print(msg=f"Gathering:\t{type}, timeframe: {timeframe} finished!")

    def extractFeature(self, types, timeframes, features):
        
        for typ in types:
            for timeframe in timeframes:
                if not os.path.isfile(f"{self.path}/Datasets/{typ}_{timeframe}.parquet"):
                    self.create_dataset(types=[typ], timeframes=[timeframe])

                self.print(msg=f"Extracting:\t{typ}_{timeframe} - {features}")
                df = pl.read_parquet(f"{self.path}/Datasets/{typ}_{timeframe}.parquet").to_pandas()
                df = df.loc[:, np.concatenate([["date", "symbol"], features])]
                
                tag = ""
                for feature in features:
                    tag = tag+"_"+feature
                df.to_parquet(f"{self.path}/Datasets/{typ}_{timeframe}{tag}.parquet", compression="gzip")
    
    def pivot_table(self, types, timeframes, features, remainder="close"):

        tag = ""
        for feature in features:
            tag = tag+"_"+feature
        for typ in types:
            for timeframe in timeframes:

                if not os.path.isfile(f"{self.path}/Datasets/{typ}_{timeframe}{tag}.parquet"):
                    self.extractFeature(types=[typ], timeframes=[timeframe], features=features)

                self.print(msg=f"Pivoting:\t{typ}_{timeframe} - ['{remainder}']")
                df = pl.read_parquet(f"{self.path}/Datasets/{typ}_{timeframe}{tag}.parquet").to_pandas()
                df.set_index('date', inplace=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                ind = np.array([df.columns == "symbol", df.columns == remainder]).T.sum(axis=1) == 0
                df.drop(columns=df.columns[ind], inplace=True)

                table = df.pivot(columns='symbol')
                table.columns = [col[1] for col in table.columns]
                table = table.reset_index()
                table.to_parquet(f"{self.path}/Datasets/{typ}_{timeframe}_{remainder}_pivot.parquet", compression="gzip")


if __name__ == "__main__":

    folder = "D:/Data/EOD"
    exchanges = ["F", "XETRA"]
    types = ["Common Stock"]
    timeframes = [1440]
    features = ["close"]

    dm = DataManager(folder=folder, verbose=True)
    dm.pivot_table(types=types, timeframes=timeframes, features=features, remainder=features[0])
  

