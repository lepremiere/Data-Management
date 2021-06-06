import os
import pandas as pd 
import polars as pl
import numpy as np
from datetime import datetime
from lib.baseClass import BaseClass

class Filter(BaseClass):

    def __init__(self, folder, verbose=True):
        super().__init__(verbose=verbose)
        self.path = folder
        self.verbose = verbose
    
    def print(self, msg):
        if self.verbose:
            print(f"{datetime.now().time().strftime('%H:%M:%S')}:\t{msg}")

    def create_folder(self, folder):
        if not os.path.exists(self.path + f"/{folder}/"):
            self.print(msg=f"Creating new folder {folder}...")
            os.makedirs(self.path + f"/{folder}/")
    
    def isMonotonic(self, A):
       return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
               all(A[i] >= A[i + 1] for i in range(len(A) - 1)))

    def filter_pivot_table(self, typ, timeframe, feature, startdate, enddate, nthresh=10):

        self.print(msg=f"Filtering: {typ}_{timeframe}_{feature} from {startdate} to {enddate} at threshold {nthresh}...")
        startdate = pd.to_datetime(startdate)
        enddate= pd.to_datetime(enddate)
        df = pl.read_parquet(f"{folder}/Datasets/{typ}_{timeframe}_{feature}_pivot.parquet").to_pandas()
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.loc[startdate:enddate, :] 
        df.reset_index(inplace=True)

        self.print(msg=f"Filtering: found {pd.isnull(df).sum().sum()} missing values...")
        first_null = pd.isnull(df.fillna(method="ffill")).sum(axis=0)
        last_legit = pd.isnull(df.fillna(method="bfill")).sum(axis=0)
        ind = np.logical_or(first_null > nthresh, last_legit > nthresh)

        df = df.loc[:, df.columns[ind==False]]
        err = pd.isnull(df).sum(axis=0)/len(df)

        df = df.fillna(method="ffill")       
        df = df.fillna(method="bfill")
        for col in df.columns[1:]:
            d = (df.loc[:, col].iloc[-1] - df.loc[:, col].iloc[0]) / df.loc[:, col].iloc[0] 
            res = (df.loc[:, col] - df.loc[:, col].rolling(window=100).mean()) 
            negvar = np.std(res[res < 0]) / np.mean(df.loc[:, col])
            sudden = np.max(np.abs(df.loc[:, col].pct_change()))
            e = err.loc[col]
            mono = self.isMonotonic(df.loc[:, col].to_numpy())

            if  sudden > 1 or mono:
                print(f"{col}:\tDiff: {round(d,2)},\tNegVar: {round(negvar,2)},\tSChange:{round(sudden,2)}\tError: {round(e,2)},\tMonotony: {mono}")
                df.drop(columns=col, inplace=True)

        self.print(msg=f"Filtering: {pd.isnull(df).sum().sum()} missing values left!")
        pl.from_pandas(df).to_csv("AAAA.csv")
        

if __name__ == "__main__":

    folder = "D:/Data/EOD"
    typ = "ETF"
    timeframe = 1440
    feature = "close"
    startdate = "2020-06-01"
    enddate = "2021-06-03"

    filt = Filter(folder=folder)
    filt.filter_pivot_table(typ=typ, timeframe=timeframe, feature=feature, startdate=startdate, enddate=enddate)



