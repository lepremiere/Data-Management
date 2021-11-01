import random
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from lib.baseClass import BaseClass

class DataManager(BaseClass):
    
    def __init__(self, 
                root=None, 
                fmt="parquet",
                verbose=True, 
                save=False, 
                plot=True, 
                num_cores=None):
        super().__init__(root=root, verbose=verbose)
        self.save = save
        self.plot = plot
        self.fmt = fmt
        if not num_cores:
            self.num_cores = cpu_count()
        else:
            self.num_cores = num_cores

    def create_dataset( self,
                        types,
                        timeframes, 
                        n_rand=None,
                        selection=None
                        ):
        """ This function collects all files that match 'types' and 'timeframes', calculates 
        indicators and/or normalizes data if choosen, and concatenates all files on axis 0.

        Parameters
        ----------
        types: list           
            List containing strings that specifies the asset class 
        timeframes: list        
            List containing integers with the timeframes of choice. In minutes.
        n_rand: int            
            Integer that specifies how many random tickers should be drawn

        Returns
        -------
            df: pd.DataFrame           
            Table containing all assets concatenated on axis 0.
        """

        # Get available files
        self.create_folder(folder="Datasets")
        av_files = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()

        for typ in types:
            for timeframe in timeframes:
                if selection:
                    ticker_list = av_files.loc[[symbol in selection 
                                                for symbol in av_files.symbol]] 
                    print(ticker_list)
                else:
                    # Filter for types and timeframes
                    ind = np.logical_and(av_files.loc[:, "type"     ] == typ,
                                         av_files.loc[:, "timeframe"] == timeframe)
                    ticker_list = av_files.loc[ind]
                self._print(msg=f"Gathering:\t{typ}, timeframe: {timeframe}...")

                # Gathering information and putting workload to Pool
                l = []
                [l.append((symbol, path))
                       for symbol, path in ticker_list.loc[:, ["symbol", "path"]].to_numpy()]

                # n random samples
                if n_rand: 
                    l = random.sample(l, n_rand) 

                with Pool(self.num_cores) as p:
                    res = p.starmap(self._load_file, l)

                df = pd.concat(res) 
                self._print(msg=f"Gathering:\tFiles remaining:\t{len(df.symbol.unique())}/{len(l)}")
                if self.fmt == "csv":
                    pl.from_pandas(df).to_csv(f"{self.path}/DataSets/{typ}_{timeframe}.csv")
                elif self.fmt == "parquet":
                    pl.from_pandas(df).to_parquet(f"{self.path}/DataSets/{typ}_{timeframe}.parquet")
                self._print(msg=f"Gathering:\t{typ}, timeframe: {timeframe} finished!")
                self._print(msg=f"\n{df.info()}\n")

        return df

    def synchronize_dataset(self, types, timeframes, start, end, n_thresh=5):
        """ This function synchronizes assets for a given time period. 

        Parameters
        ----------
        types: list         
            List containing strings that specifies the asset class 
        timeframes: list     
            List containing integers with the timeframes of choice. In minutes.
        start: pandas.Timestamp         
            Timestamp that defines the start of the time period of interest.
        end: pandas.Timestamp           
            Pandas Timestamp that defines the end of the time period of interest.
        n_thresh: int      
            Integer that mirrors the tolerance of consecutive missing values befor exclusion.

        Returns
        -------
        new_df: pandas.DataFrame        
            Table that contains all assets synchronized from start to end.
        """

        for typ in types:
            for timeframe in timeframes:

                # Loading dataset
                self._print(msg=f"Synchronizing {typ}_{timeframe}")
                df = pl.read_parquet(f"{self.path}Datasets/{typ}_{timeframe}.parquet").to_pandas()
                df.date = pd.to_datetime(df.date)
                df.set_index(["date", "symbol"], inplace=True)
                
                # Pivoting around close
                pivot_table = df.close.reset_index("symbol").pivot(columns="symbol")
                pivot_table.columns = [col[1] for col in pivot_table.columns]
                pivot_table = pivot_table.loc[start:end, :]

                # Determining max consecutive missing values 
                if timeframe < 1440: n_thresh = 16
                consecutive_nulls = self._consec_null(nulls=1*pivot_table.isnull(), n_thresh=n_thresh)
                maxs = consecutive_nulls.max(axis=0)
                
                # Determining the columns to include
                cols = maxs.loc[maxs.values < n_thresh].index
                dates = pivot_table.index.get_level_values(0)

                # Creating MultiIndex for output
                inds = []; [[inds.append((date, col)) for date in dates] for col in cols]
                inds = pd.MultiIndex.from_tuples(inds, names=df.index.names)
                
                # Filling remaining symbols in nan DF via intersection
                new_df = pd.DataFrame(np.nan, index=inds, columns=df.columns)
                new_df = new_df.combine_first(df.loc[inds.intersection(df.index)])
                new_df = self._fill_nans(df=new_df).reset_index()
                new_df.date = new_df.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
                self._print(msg=f"Files remaining: {np.sum(maxs.values < n_thresh)}/{len(maxs.values)}")

                pl.from_pandas(new_df).to_parquet(
                    f"{self.path}/Datasets/{typ}_{timeframe}_synchronized.parquet"
                    )

        return new_df 
    
    def pivot_table(self, 
                    types, 
                    timeframes, 
                    start, end, 
                    remainder="close", 
                    n_thresh=6, 
                    selection=None):
        """ This function pivots a specific column in the asset dataset and synchronizes 
            the pivot table. 

        Parameters
        ----------
        types: list          
            List containing strings that specifies the asset class 
        timeframes: list    
            List containing integers with the timeframes of choice. In minutes.
        start: pandas.Timestamp          
            Timestamp that defines the start of the time period of interest.
        end: pandas.Timestamp           
            Timestamp that defines the end of the time period of interest.
        remainder: str     
            String that specifies the column over which should be pivoted.
        n_thresh: int      
            Integer that mirrors the tolerance of consecutive missing values befor exclusion.

        Returns
        -------
        new_df: pandas.DataFrame        
            Table that contains the remainder of all selected assets synchronized from start to end.
        """

        for typ in types:
            for timeframe in timeframes:
                
                self._print(msg=f"Pivoting:\t{typ}_{timeframe} - Remainder: {remainder}")

                # Special case if adjusted price data is available
                try:
                    df = pd.read_parquet(f"{self.path}/Datasets/{typ}_{timeframe}.parquet",
                                         columns=["date", "symbol", "adjclose"])
                    remainder = "adjclose"
                except:
                    df = pd.read_parquet(f"{self.path}/Datasets/{typ}_{timeframe}.parquet",
                                         columns=["date", "symbol", "adjusted_close"])
                    remainder = "adjusted_close"
  
                df.set_index('date', inplace=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                
                # Pivot and select time period
                df = df.pivot(columns='symbol')
                df.columns = [col[1] for col in df.columns]
                if selection:
                    df = df.loc[start:end, selection]
                else:
                    df = df.loc[start:end, :]
                    
                # Determining max consecutive nulls of collumns
                if timeframe < 1440: n_thresh = 16
                consecutive_nulls = self._consec_null(df.isnull().astype(np.int32), n_thresh)
                maxs = consecutive_nulls.max(axis=0)

                # Determining columns to include, final assembly
                df = df.loc[:, maxs.loc[maxs.values < n_thresh].index]
                df = self._fill_nans(df=df).reset_index()
                df.date = df.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

                self._print(
                    f"Symbols remaining: {np.sum(maxs.values < n_thresh)}/{len(maxs.values)}"
                    )
                pl.from_pandas(df).to_parquet(
                    f"{self.path}/Datasets/{typ}_{timeframe}_pivot.parquet"
                    )

        return df

    def combine_pivot_tables(self, types, timeframe):
        """ This function concatenates pivot tables horizontally. 

        Parameters
        ----------
        types: list         
            List containing strings that specify the asset classes to combine. 
        timeframe: int     
            Integer with the timeframe of choice. In minutes.

        Returns
        -------
        df: pandas.DataFrame            
            Tablethat contains all asset pivot tables concatenated over axis 1.
        """
        l = []
        for typ in types:
            df = pl.read_parquet(f"{self.path}/Datasets/{typ}_{timeframe}_pivot.parquet").to_pandas()
            df.columns[1:] = df.columns[1:] + "_" + typ
            l.append(df)

        df = pd.concat(l, axis=1)
        pl.from_pandas(df).to_parquet(f"{self.path}/Datasets/Ultra_{timeframe}_pivot.parquet")

        return df

    def _load_file(self, symbol, path):
        """ Helper function that loads and processes a single file """

        # Load the file and fillna volume
        df = pd.read_csv(path)
        df.columns = [col.lower() for col in df.columns]
        df.volume.fillna(0, inplace=True)

        # Check price data 
        flag = self._filter_ticker(df.copy(), ticker=symbol)
        if flag: return pd.DataFrame()
        
        df.insert(loc=1, column="symbol", value=symbol)
        df = df.reset_index().drop(columns="index")
        df.date = pd.to_datetime(df.date).dt.strftime("%Y-%m-%d %H:%M:%S")

        return df

    def _fill_nans(self, df):
        """ Helper function that forward and backwards fills missing values """

        self._print(msg=f"Missing values before 'ffill': {df.isnull().sum().sum(), df.isnull().sum(axis=0).max()}")
        df.fillna(method="ffill", inplace=True)
        self._print(msg=f"Missing values after  'ffill': {df.isnull().sum().sum(), df.isnull().sum().max()}")
        df.fillna(method="bfill", inplace=True)
        self._print(msg=f"Missing values after  'bfill': {df.isnull().sum().sum(), df.isnull().sum().max()}")
        
        return df

    def _filter_ticker(self, df, ticker, max_delta=0.15, max_mono=20, minlen=100):
        
        close = df.loc[:, ["date", "adjusted_close" if "adjusted_close" in df.columns else "close"]]
        close.columns = ["date", "close"]
        close.set_index("date", inplace=True)
        close.index = pd.to_datetime(close.index)

        # Filter: max percentage change, length, monotony/length, max monotony
        deltas = close.pct_change()
        max_delta = deltas.abs().max().values
        end_return = (close.values[-1]  / close.values[0])[0]
        length = len(close)
        monotony = np.sum(deltas.values == 0) / length
        max_consec_mono = self._consec_null(pd.DataFrame( 1 * (deltas.values == 0))).max().values[0]

        if end_return < 0.1 or max_delta > max_delta \
            or length < minlen or max_consec_mono > max_mono \
                or ticker in ["PR1U", "GOAI", "XY1D", "XIEE", "XHY1", "XCS2"]:
            
            self._print((f"{ticker}\texcluded! max_delta: {round(max_delta[0], 2)},\t"
                         f"length: {length},\tmonotony: {round(monotony, 2)},\t"
                         f"max_mono: {max_consec_mono}\tend_return: {round(end_return,2)}"))
            if self.plot:
                fig = plt.figure(figsize=(12,8))
                axs = fig.subplots(2, sharex=True)
                close.plot(ax=axs[0])
                axs[1].plot(deltas)
                plt.title((f"{ticker}: max_delta: {round(max_delta[0], 2)}, "
                           f"length: {length}, monotony: {round(monotony, 2)}, "
                           f"max_mono: {max_consec_mono}"))
                plt.savefig(f"Temp/{ticker}.png")
                plt.close()
      
            return True
        return False

    def _consec_null(self, nulls, n_thresh=1):
        out = nulls
        nulls = nulls.to_numpy()
        l = len(nulls)

        for i in range(l):
            if not i == 0:
                pre = (nulls[i-1,:], 0)
                now = (nulls[i,:], 0)
                if i != l:
                    nulls[i,:] = (pre[0] + now[0]) * np.max(nulls[i:i+n_thresh+1,:], axis=0)
                else:
                    nulls[i,:] = (pre[0] + now[0]) * now[0]
        out = pd.DataFrame(nulls, columns=out.columns, index=out.index)
        # out.to_csv("abc.csv")
        return out

if __name__ == "__main__":
    pass

    