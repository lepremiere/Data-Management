import time
import numpy as np 
import pandas as pd 
import polars as pl
import matplotlib.pyplot as plt

from tqdm import tqdm
from talib import abstract
from collections import OrderedDict
from tqdm import tqdm

def get_indicators(df, settings, normalize=False, verbose=True):
    flag = False
    df.columns = [col.lower() for col in df.columns]
    try:
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)
        idx = df.index
    except:
        idx = pd.to_datetime(df.index)
    
    tag = ""
    if "adjclose" in df.columns:
        tag = "adj"
        inputs = {
            'open': df[f"{tag}open"].to_numpy().astype(float)+1,
            'high': df[f"{tag}high"].to_numpy().astype(float)+1,
            'low': df[f"{tag}low"].to_numpy().astype(float)+1,
            'close': df[f"{tag}close"].to_numpy().astype(float)+1,
            'volume': df[f"{tag}volume"].to_numpy().astype(float)+1
        }

    elif "adjusted_close" in df.columns:

        inputs = {
            'open': df[f"adjusted_close"].to_numpy().astype(float)+1,
            'high': df[f"high"].to_numpy().astype(float)+1,
            'low': df[f"low"].to_numpy().astype(float)+1,
            'close': df[f"close"].to_numpy().astype(float)+1,
            'volume': df[f"volume"].to_numpy().astype(float)+1
        }

    # Indicators
    l = []
    for name, setting in settings.items():
        try:
            parameters = setting["parameters"]
            normalization = setting["normalization"]
            fun = abstract.Function(name)
        
            for i in range(len(list(parameters.values())[0])):
                output_names = setting["output_names"]
                params = parameters

                if list(parameters.values())[0][0]:
                    params = OrderedDict([(p, parameters[p][i]) for p in parameters.keys()])
                    fun.parameters = params
                out = np.array(fun(inputs))

                if normalize and normalization:
                    out = eval(normalization)
                    output_names = setting["output_names_normalized"]

                tag = f"_{params[list(params.keys())[0]]}" if len(list(parameters.values())[0]) > 1 else ""
                cols = [f"{name}{tag}"] if "real" in output_names else [f"{name}_{l}{tag}" for l in output_names] 
                out = pd.DataFrame(out.T, columns=cols, index=idx) 
                l.append(out)
                
        except Exception as e:
            print(name, fun.parameters, normalization, output_names, e)
            flag = True

    l = pd.concat(l, axis=1)

    # Final assembly
    df = pd.concat([df, l], axis=1).reset_index()
    df.dropna(inplace=True)

    return df, flag

def get_returns(df, historic_returns, forward_returns):
    tag = ""
    if "adjclose" in df.columns: tag = "adj" 
    close = df.loc[:, f"{tag}close"]

    if historic_returns:
        for lag in historic_returns:
            returns = close.pct_change(lag)
            df[f"historic_return_{lag}d"] = returns 
    if forward_returns:
        for lag in forward_returns:
            returns = close.pct_change(lag)
            df[f"target_return_{lag}d"] = returns.shift(-lag) 

    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    pass
