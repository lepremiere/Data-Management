import random
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from lib.baseClass import BaseClass
from lib.indicator_settings import settings
from lib.indicators import get_indicators, get_returns

class DataManager(BaseClass):
    
    def __init__(self, folder=None, verbose=True, save=True, plot=True, num_cores=1):
        super().__init__(folder=folder, verbose=verbose)
        self.save = save
        self.plot = plot
        self.num_cores = num_cores

    def create_dataset( self,
                        types,
                        timeframes, 
                        indicators=False, 
                        normalize=False, 
                        historic_returns=None, 
                        forward_returns=None, 
                        n_rand=None
                        ):
        """
        This function collects all files that match 'types' and 'timeframes', calculates indicators if choosen, and
        concatenates all files on axis 0.

        Inputs
            types:              List containing strings that specifies the asset class 
            timeframes:         List containing integers with the timeframes of choice. In minutes.
            indicators:         Bool that indicates if indicators should be calculated
            normalize:          Bool that indicates if indicators should be normalized following the logic definded in 'indicator_settings'
            historic_returns:   List of integers that contains periods for which 'historic' returns should be calculated
            forward_returns:    List of integers that contains periods for which 'historic' returns should be calculated and shifted to form 'target' returns

        Outputs
            df:                 Pandas DataFrame that contains all assets concatenated on axis 0.
        """

        self.create_folder(folder="Datasets")
        av_files = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()

        for typ in types:
            for timeframe in timeframes:
                l = []
                ind = np.logical_and(av_files.loc[:, "type"]        == typ,
                                     av_files.loc[:, "timeframe"]   == timeframe)
                ind = np.logical_and(ind , ["ETC" not in name for name in av_files.loc[:, "name"]])

                ticker_list = av_files.loc[ind]
                self.print(msg=f"Gathering:\t{typ}, timeframe: {timeframe}...")

                [l.append((symbol, path, indicators, normalize, historic_returns, forward_returns))
                for symbol, path in ticker_list.loc[:, ["symbol", "path"]].to_numpy()]

                if n_rand: l = random.sample(l, n_rand) 
                with Pool(self.num_cores) as p:
                    res = p.starmap(self._load_file, l)

                df = pd.concat(res) 
                self.print(msg=f"Gathering:\tFiles remaining:\t{len(df.symbol.unique())} / {len(ticker_list)}")
                pl.from_pandas(df).to_parquet(f"{self.path}/DataSets/{typ}_{timeframe}.parquet")
                self.print(msg=f"Gathering:\t{typ}, timeframe: {timeframe} finished!")
                self.print(msg=f"\n{df.info()}\n")

        return df

    def synchronize_dataset(self, types, timeframes, start, end, n_thresh=5):
        """
        This function synchronizes assets for a given time period. 

        Inputs
            types:          List containing strings that specifies the asset class 
            timeframes:     List containing integers with the timeframes of choice. In minutes.
            start:          Pandas Timestamp that defines the start of the time period of interest.
            end:            Pandas Timestamp that defines the end of the time period of interest.
            n_thresh:       Integer that mirrors the tolerance of consecutive missing values befor exclusion.

        Outputs
            new_df:         Pandas DataFrame that contains all assets synchronized from start to end.
        """

        for typ in types:
            for timeframe in timeframes:
                if timeframe < 1440: n_thresh = 10
                self.print(msg=f"Synchronizing {typ}_{timeframe}")
                df = pl.read_parquet(f"{self.path}Datasets/{typ}_{timeframe}.parquet").to_pandas()
                df.date = pd.to_datetime(df.date)
                df.set_index(["date", "symbol"], inplace=True)
                
                t = df.close.reset_index("symbol").pivot(columns="symbol")
                t.columns = [col[1] for col in t.columns]
                t = t.loc[start:end, :]

                consecutive_nulls = self._consec_null(nulls=1*t.isnull(), thresh=n_thresh)
                maxs = consecutive_nulls.max(axis=0)
                
                cols = maxs.loc[maxs.values < n_thresh].index
                dates = t.index.get_level_values(0)
                inds = []; [[inds.append((date, col)) for date in dates] for col in cols]
                inds = pd.MultiIndex.from_tuples(inds, names=df.index.names)
                
                new_df = pd.DataFrame(np.nan, index=inds, columns=df.columns)
                new_df = new_df.combine_first(df.loc[inds.intersection(df.index)])
                self.print(msg=f"Files remaining: {np.sum(maxs.values < n_thresh)} / {len(maxs.values)}")
                new_df = self._fill_nans(df=new_df)
                new_df = new_df.reset_index()
                new_df.date = new_df.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

                pl.from_pandas(new_df).to_parquet(f"{self.path}/Datasets/{typ}_{timeframe}_synchronized.parquet")

        return new_df 
    
    def pivot_table(self, types, timeframes, start, end, remainder="close", n_thresh=5):
        """
        This function pivots a specific column in the asset dataset and synchronizes the pivot table. 

        Inputs
            types:          List containing strings that specifies the asset class 
            timeframes:     List containing integers with the timeframes of choice. In minutes.
            start:          Pandas Timestamp that defines the start of the time period of interest.
            end:            Pandas Timestamp that defines the end of the time period of interest.
            remainder:      String that specifies the column over which should be pivoted.
            n_thresh:       Integer that mirrors the tolerance of consecutive missing values befor exclusion.

        Outputs
            new_df:         Pandas DataFrame that contains all assets synchronized from start to end.
        """

        for typ in types:
            for timeframe in timeframes:
                if timeframe < 1440: n_thresh = 16
                self.print(msg=f"Pivoting:\t{typ}_{timeframe} - Remainder: {remainder}")
                try:
                    df = pd.read_parquet(f"{self.path}/Datasets/{typ}_{timeframe}.parquet", columns=["date", "symbol", "close", "adjclose"])
                    if f"adj{remainder}" in df.columns: remainder = f"adj{remainder}"
                except:
                    df = pd.read_parquet(f"{self.path}/Datasets/{typ}_{timeframe}.parquet", columns=["date", "symbol", "close", "adjusted_close"])
  
                df = df.loc[:, ["date", "symbol", remainder]]
                df.set_index('date', inplace=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                
                # Pivot and select time period
                df = df.pivot(columns='symbol')
                df.columns = [col[1] for col in df.columns]
                df = df.loc[start:end]
                

                consecutive_nulls = self._consec_null(nulls=df.isnull().astype(np.int16), thresh=n_thresh)
                maxs = consecutive_nulls.max(axis=0)
                cols = maxs.loc[maxs.values < n_thresh].index

                df = df.loc[:, cols]
                self.print(msg=f"Symbols remaining: {np.sum(maxs.values < n_thresh)} / {len(maxs.values)}")
                df = self._fill_nans(df=df)
                df = df.reset_index()
                df.date = df.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
                self.print(msg=df.info())
                pl.from_pandas(df).to_parquet(f"{self.path}/Datasets/{typ}_{timeframe}_{remainder}_pivot.parquet")

        return df

    def combine_pivot_tables(self, types, timeframe):
        """
        This function concatenates pivot tables horizontally. 

        Inputs
            types:          List containing strings that specify the asset classes to combine. 
            timeframe:      Integer with the timeframe of choice. In minutes.

        Outputs
            df:             Pandas DataFrame that contains all asset pivot tables concatenated over axis 1.
        """
        l = []
        for typ in types:
            df = pl.read_parquet(f"{self.path}/Datasets/{typ}_{timeframe}_adjclose_pivot.parquet").to_pandas()
            df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index)

            df.columns = df.columns + "_" + typ
            l.append(df)

        df = pd.concat(l, axis=1)
        df.reset_index(inplace=True)
        pl.from_pandas(df).to_parquet(f"{self.path}/Datasets/Ultra_{timeframe}.parquet")

        return df

    def _load_file(self, symbol, path, indicators, normalize, historic_returns, forward_returns):
        df = pl.read_csv(path).to_pandas()
        df.columns = [col.lower() for col in df.columns]
        df.drop_duplicates(subset=['date'], inplace=True)

        df, flag = self._filter_ticker(df, ticker=symbol)
        if flag: return pd.DataFrame()
        df.volume.fillna(0, inplace=True)

        if historic_returns or forward_returns: df = get_returns(df, historic_returns=historic_returns, forward_returns=forward_returns)
        if indicators: df, flag = get_indicators(df=df, settings=settings, normalize=normalize, verbose=False)
        if flag: self.print(msg=f"{symbol} getting indicators failed!"); return pd.DataFrame()
        
        df.dropna(inplace=True)
        df.insert(loc=1, column="symbol", value=symbol)
        df.reset_index(inplace=True)  
        df.drop(columns="index", inplace=True) 
        df.date = df.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        return df

    def _fill_nans(self, df):
        self.print(msg=f"Missing values before 'ffill': {df.isnull().sum().sum(), df.isnull().sum(axis=0).max()}")
        df.fillna(method="ffill", inplace=True)
        self.print(msg=f"Missing values after 'ffill': {df.isnull().sum().sum(), df.isnull().sum().max()}")
        df.fillna(method="bfill", inplace=True)
        self.print(msg=f"Missing values after 'bfill': {df.isnull().sum().sum(), df.isnull().sum().max()}")
        
        return df

    def _filter_ticker(self, df, ticker, max_delta=1, max_mono=100, minlen=100):
        
        close = df.loc[:, ["date", "adjusted_close" if "adjusted_close" in df.columns else "close"]]
        close.columns = ["date", "close"]
        close.set_index("date", inplace=True)
        close.index = pd.to_datetime(close.index)

        # Filter: max percentage change, length, monotony/length, max monotony
        deltas = close.pct_change()
        max_delta = deltas.max().values
        end_return = (close.values[-1]  / close.values[0])[0]
        length = len(close)
        monotony = np.sum(deltas.values == 0) / length
        max_consec_mono = self._consec_null(pd.DataFrame( 1 * (deltas.values == 0))).max().values[0]

        if max_delta > max_delta or end_return < 0.5 or length < minlen or max_consec_mono > max_mono:
            if self.plot:
                fig = plt.figure(figsize=(12,8))
                axs = fig.subplots(2, sharex=True)
                close.plot(ax=axs[0])
                axs[1].plot(deltas)
                plt.title(f"{ticker}: max_delta: {round(max_delta[0], 2)}, length: {length}, monotony: {round(monotony, 2)}, max_mono: {max_consec_mono}")
                plt.savefig(f"Temp/{ticker}.png")
                plt.close()
            self.print(msg=f"{ticker}\texcluded! max_delta: {round(max_delta[0], 2)},\tlength: {length},\tmonotony: {round(monotony, 2)},\tmax_mono: {max_consec_mono}\tend_return: {round(end_return,2)}")
      
            return None, True

        return df, False

    def _consec_null(self, nulls, thresh=1):
        out = nulls
        nulls = nulls.to_numpy()
        l = len(nulls)

        for i in range(l):
            if not i == 0:
                pre = (nulls[i-1,:], 0)
                now = (nulls[i,:], 0)
                if i != l:
                    nulls[i,:] = (pre[0] + now[0]) * np.max(nulls[i:i+thresh+1,:], axis=0)
                else:
                    nulls[i,:] = (pre[0] + now[0]) * now[0]
        out = pd.DataFrame(nulls, columns=out.columns, index=out.index)

        return out

if __name__ == "__main__":
    pass

    