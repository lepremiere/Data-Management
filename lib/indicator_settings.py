settings = {
    ######################################################################################
    ############################# Cyclic Indicators #####################################
    ######################################################################################
    # Hilbert Transform - Dominant Cycle Period
    "HT_DCPERIOD": {
        "parameters": {
            "None":                 [None],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 50"
    },
    # Hilbert Transform - Dominant Cycle Phase
    "HT_DCPHASE": {
        "parameters": {
            "None":                 [None],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 320"
    },
    # Hilbert Transform - Phasor
    "HT_PHASOR": {
        "parameters": {
            "None":                 [None],
        },
        "output_names":             ['inphase', 'quadrature'],
        "output_names_normalized":  ['inphase', 'quadrature'],
        "normalization":            "out  / (inputs['close'] + 1e-9) "
    },
    # Hilbert Transform - SineWave
    "HT_SINE": {
        "parameters": {
            "None":                 [None],
        },
        "output_names":             ['sine', 'leadsine'],
        "output_names_normalized":  ['sine', 'leadsine'],
        "normalization":            None
    },
    # Hilbert Transform - Trend vs Cycle Mode
    "HT_TRENDMODE": {
        "parameters": {
            "None":                 [None],
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            None
    },

    ######################################################################################
    ############################# Math Transformation ####################################
    ######################################################################################
    # Vector Arithmetic Add
    "ADD": {
        "parameters": {
            "None":                 [None],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "np.concatenate([[np.nan], (out[1:] - out[0:-1]) / (out[0:-1] + 1e-9)])"
    },
    # Vector Arithmetic Div
    "DIV": {
        "parameters": {
            "None":                 [None],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "(out - 1)"
    },
    # Vector Arithmetic Mult
    "MULT": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "np.sqrt(out + 1e-9) / (inputs['close'] + 1e-9) - 1"
    },
    # Vector Arithmetic Substraction
    "SUB": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9)"
    },
    # Vector Arithmetic ATan
    "ATAN": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "np.concatenate([[np.nan], (out[1:] - out[0:-1]) / (out[0:-1] + 1e-9)])"
    },
    # Vector Ceil
    "CEIL": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "(out - inputs['close'])"
    },
    # Vector Trigonometric Cos
    "COS": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            None
    },
    # Vector Floor
    "FLOOR": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "(out - inputs['close'])"
    },
    # Vector Log Natural
    "LN": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['returns'],
        "normalization":            "np.concatenate([[np.nan], (out[1:] - out[0:-1])])"
    },
    # Vector Log 10
    "LOG10": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['returns'],
        "normalization":            "np.concatenate([[np.nan], (out[1:] - out[0:-1])])"
    },
    # Vector Trigonometric Sin
    "SIN": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            None
    },
    # Vector Square Root
    "SQRT": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['returns'],
        "normalization":            "np.concatenate([[np.nan], (out[1:] - out[0:-1]) / (out[0:-1] + 1e-9)])"
    },
    # Vector Trigonometric Tan
    # "TAN": {
    #     "parameters": {
    #         "None":                 [None]
    #     },
    #     "output_names":             ['real'],
    #     "output_names_normalized":  ['real'],
    #     "normalization":            "np.sqrt(np.sqrt(out**2)) * np.sign(out)  / (inputs['close'] + 1e-9)"
    # },
    # Highest value over a specified period
    "MAX": {
        "parameters": {
            "timeperiod":           [30, 60]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Index of highest value over a specified period
    "MAXINDEX": {
        "parameters": {
            "timeperiod":           [30, 60]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "(out - range(len(inputs['close'])))"
    },
    # Lowest value over a specified period
    "MIN": {
        "parameters": {
            "timeperiod":           [30, 60]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Index of lowest value over a specified period
    "MININDEX": {
        "parameters": {
            "timeperiod":           [30, 60]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "(out - range(len(inputs['close'])))"
    },
    # Lowest and highest values over a specified period
    "MINMAX": {
        "parameters": {
            "timeperiod":           [30, 60]
        },
        "output_names":             ['min', 'max'],
        "output_names_normalized":  ['real'],
        "normalization":            "out[0]  / (out[1] + 1e-9) - 1 "
    },
    # Indexes of lowest and highest values over a specified period
    "MINMAXINDEX": {
        "parameters": {
            "timeperiod":           [30, 60]
        },
        "output_names":             ['minidx', 'maxidx'],
        "output_names_normalized":  ['integer'],
        "normalization":            "(out[1] - range(len(inputs['close']))) - (out[0] - range(len(inputs['close'])))"
    },

    ######################################################################################
    ########################## Momentum Indicators #######################################
    ######################################################################################
    # Average Directional Movement Index
    "ADX": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100"
    },
    # Average Directional Movement Index Rating
    "ADXR": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100"
    },
    # Absolute Price Oscillator
    "APO": {
        "parameters": {
            "fastperiod":           [12],
            "slowperiod":           [26]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9)"
    },
    # Aroon
    "AROON": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['aroondown', 'aroonup'],
        "output_names_normalized":  ['aroondown', 'aroonup'],
        "normalization":            "out / 100"
    },
    # Aroon Oscillator
    "AROONOSC": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100"
    },
    # Balance Of Power
    "BOP": {
        "parameters": {
            "None":           [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            None
    },
    # Commodity Channel Index
    "CCI": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 500"
    },
    # Chande Momentum Oscillator
    "CMO": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100"
    },
    # Directional Movement Index
    "DX": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100"
    },
    # Momentum Indicators,Moving Average Convergence/Divergence
    "MACD": {
        "parameters": {
            "fastperiod":           [12, 24],
            "slowperiod":           [26, 52],
            "signalperiod":         [9, 12]
        },
        "output_names":             ['macd', 'macdsignal', 'macdhist'],
        "output_names_normalized":  ['macd', 'macdsignal', 'macdhist'],
        "normalization":            "out / (inputs['close'] + 1e-9)"
    },
    # Momentum Indicators,MACD with controllable MA type
    "MACDEXT": {
        "parameters": {
            "fastperiod":           [12, 24],
            "fastmatype":           [5, 5],
            "slowperiod":           [26, 52],
            "slowmatype":           [5, 5],
            "signalperiod":         [9, 12],
            "signalmatype":         [5, 5],
        },
        "output_names":             ['macd', 'macdsignal', 'macdhist'],
        "output_names_normalized":  ['macd', 'macdsignal', 'macdhist'],
        "normalization":            "out / (inputs['close'] + 1e-9)"
    },
    # Momentum Indicators,Money Flow Index
    "MFI": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100"
    },
    # Momentum Indicators,Minus Directional Indicator
    "MINUS_DI": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100"
    },
    # Momentum Indicators,Minus Directional Movement
    "MINUS_DM": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9)"
    },
    # Momentum
    "MOM": {
        "parameters": {
            "timeperiod":           [10, 21]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9)"
    },
    # Plus Directional Indicator
    "PLUS_DI": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 50"
    },
    # Plus Directional Movement
    "PLUS_DM": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9)"
    },
    # Percentage Price Oscillator
    "PPO": {
        "parameters": {
            "fastperiod":           [12, 24],
            "slowperiod":           [26, 52],
            "matype":               [0, 0]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 20"
    },
    # Rate of change Percentage
    "ROCP": {
        "parameters": {
            "timeperiod":           [1, 5, 10, 21]
        },
        "output_names":             ['returns'],
        "output_names_normalized":  ['returns'],
        "normalization":            None
    },
    # Relative Strength Index
    "RSI": {
        "parameters": {
            "timeperiod":           [5, 14, 21, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100"
    },
    # Stochastic
    "STOCH": {
        "parameters": {
            "fastk_period":         [5, 10, 15, 30],
            "slowk_period":         [3, 6, 9, 18],
            "slowk_matype":         [0, 0, 0, 0],
            "slowd_period":         [3, 6, 9, 18],
            "slowd_matype":         [0, 0, 0, 0]
        },
        "output_names":             ['slowk', 'slowd'],
        "output_names_normalized":  ['slowk', 'slowd'],
        "normalization":            "out / 100"
    },
    # Stochastic Fast
    "STOCHF": {
        "parameters": {
            "fastk_period":         [5, 10, 15, 30],
            "fastd_period":         [3, 6, 9, 18],
            "fastd_matype":         [0, 0, 0, 0],
        },
        "output_names":             ['fastk', 'fastd'],
        "output_names_normalized":  ['fastk', 'fastd'],
        "normalization":            "out / 100"
    },
    # Stochastic Relative Strength Index
    "STOCHRSI": {
        "parameters": {
            "timeperiod":           [14, 21, 30],
            "fastk_period":         [5, 10, 15],
            "fastd_period":         [3, 6, 9],
            "fastd_matype":         [0, 0, 0]
        },
        "output_names":             ['fastk', 'fastd'],
        "output_names_normalized":  ['fastk', 'fastd'],
        "normalization":            "out / 100"
    },
    # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    "TRIX": {
        "parameters": {
            "timeperiod":           [14, 30],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            None
    },
    # Ultimate Oscillator
    "ULTOSC": {
        "parameters": {
            "timeperiod1":           [7, 14],
            "timeperiod2":           [14, 28],
            "timeperiod3":           [28, 56],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100"
    },
    # Williams' %R
    "WILLR": {
        "parameters": {
            "timeperiod":           [14, 30],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100 * -1"
    },

    ######################################################################################
    ############################### Overlap Studies ######################################
    ######################################################################################
    # Bollinger Bands
    "BBANDS": {
        "parameters": {
            "timeperiod":           [5, 14],
            "nbdevup":              [1.96, 2.57],
            "nbdevdn":              [1.96, 2.57],
            "matype":               [0, 0]
        },
        "output_names":             ['upperband', 'middleband', 'lowerband'],
        "output_names_normalized":  ['upper_close', 'lower_close', 'upper_lower'],
        "normalization":            "np.array([out[0] / (inputs['close'] + 1e-9), out[2] / (inputs['close'] + 1e-9), out[0] / (out[2] + 1e-9)]) - 1"
    },
    # Double Exponential Moving Average
    "DEMA": {
        "parameters": {
            "timeperiod":           [14, 30],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Exponential Moving Average
    "EMA": {
        "parameters": {
            "timeperiod":           [14, 30],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Hilbert Transform - Instantaneous Trendline
    "HT_TRENDLINE": {
        "parameters": {
            "None":                 [None],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Kaufman Adaptive Moving Average
    "KAMA": {
        "parameters": {
            "timeperiod":           [14, 30],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Moving average
    "MA": {
        "parameters": {
            "timeperiod":           [9, 14, 21, 30],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # MESA Adaptive Moving Average
    "MAMA": {
        "parameters": {
            "fastlimit":            [0.5, 0.7],
            "slowlimit":            [0.05, 0.1]
        },
        "output_names":             ['mama', 'fama'],
        "output_names_normalized":  ['mama', 'fama'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # MidPoint over period
    "MIDPOINT": {
        "parameters": {
            "timeperiod":           [14, 30],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Midpoint Price over period
    "MIDPRICE": {
        "parameters": {
            "timeperiod":           [14, 30],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Parabolic SAR
    "SAR": {
        "parameters": {
            "acceleration":         [0.02, 0.04, 0.1],
            "maximum":              [0.2, 0.4, 0.8],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Parabolic SAR - Extended
    "SAREXT": {
        "parameters": {
            "startvalue":           [0.],
            "offsetonreverse":      [0.],
            "accelerationinitlong": [0.02],
            "accelerationlong":     [0.02],
            "accelerationmaxlong":  [0.2],
            "accelerationinitshort":[0.02],
            "accelerationshort":    [0.02],
            "accelerationmaxshort": [0.2],
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "np.sqrt(out**2 + 1e-9) / (inputs['close'] + 1e-9) - 1"
    },
    # Simple Moving Average
    "SMA": {
        "parameters": {
            "timeperiod":           [6, 9, 14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Triple Exponential Moving Average (T3)
    "T3": {
        "parameters": {
            "timeperiod":           [5, 10],
            "vfactor":              [0.7, 0.3]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Triple Exponential Moving Average
    "TEMA": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Triangular Moving Average
    "TRIMA": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Weighted Moving Average
    "WMA": {
        "parameters": {
            "timeperiod":           [14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },

    ######################################################################################
    ################################# Price Transform ####################################
    ######################################################################################
    # Average Price
    "AVGPRICE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Median Price
    "MEDPRICE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Typical Price
    "TYPPRICE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Weighted Close Price
    "WCLPRICE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    ######################################################################################
    ############################### Statistic Functions ##################################
    ######################################################################################    
    # Beta
    "BETA": {
        "parameters": {
            "timeperiod":           [5, 10]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 100"
    },
    # Pearson's Correlation Coefficient (r)
    "CORREL": {
        "parameters": {
            "timeperiod":           [9, 14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            None
    },
    # Linear Regression
    "LINEARREG": {
        "parameters": {
            "timeperiod":           [9, 14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Linear Regression Angle
    "LINEARREG_ANGLE": {
        "parameters": {
            "timeperiod":           [9, 14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "(out / 90)"
    },
    # Linear Regression Intercept
    "LINEARREG_INTERCEPT": {
        "parameters": {
            "timeperiod":           [9, 14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Standard Deviation
    "STDDEV": {
        "parameters": {
            "timeperiod":           [9, 14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9)"
    },
    # Time Series Forecast
    "TSF": {
        "parameters": {
            "timeperiod":           [9, 14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9) - 1"
    },
    # Variance
    "VAR": {
        "parameters": {
            "timeperiod":           [9, 14, 30],
            "nbdev":                [1., 1., 1.]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "(pd.DataFrame(out).rolling(5).mean().to_numpy().T - out) / (pd.DataFrame(out).rolling(5).mean().to_numpy().T + 1e-9)"
    },

    ######################################################################################
    ############################## Volatility Indicators #################################
    ######################################################################################
    # Average True Range
    "ATR": {
        "parameters": {
            "timeperiod":           [9, 14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9)"
    },
    # Normalized Average True Range
    "NATR": {
        "parameters": {
            "timeperiod":           [9, 14, 30]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / 10"
    },
    # True Range
    "TRANGE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "out / (inputs['close'] + 1e-9)"
    },

    ######################################################################################
    ############################### Volume Indicators ####################################
    ######################################################################################

    # A/D Line
    "AD": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "np.concatenate([[np.nan], np.sign(out[1:] - out[0:-1]) * np.log(np.abs(out[1:] - out[0:-1] + 1e-9))]) / 10"
    },
    # Chaikin A/D Oscillator
    "ADOSC": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "np.concatenate([[np.nan], np.sign(out[1:] - out[0:-1]) * np.log(np.abs(out[1:] - out[0:-1] + 1e-9))]) / 10"
    },
    # On Balance Volume
    "OBV": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['real'],
        "output_names_normalized":  ['real'],
        "normalization":            "np.concatenate([[np.nan], np.sign(out[1:] - out[0:-1]) * np.log(np.abs(out[1:] - out[0:-1] + 1e-9))]) / 10"
    },

    ######################################################################################
    ############################### Pattern recognition ##################################
    ######################################################################################
    # Two Crows
    "CDL2CROWS": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Three Black Crows
    "CDL3BLACKCROWS": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Three Inside Up/Down
    "CDL3INSIDE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Three-Line Strike 
    "CDL3LINESTRIKE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Three Outside Up/Down
    "CDL3OUTSIDE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Three Stars In The South
    "CDL3STARSINSOUTH": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Three Advancing White Soldiers
    "CDL3WHITESOLDIERS": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Abandoned Baby
    "CDLABANDONEDBABY": {
        "parameters": {
            "penetration":          [0.3]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Advance Block
    "CDLADVANCEBLOCK": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Belt-hold
    "CDLBELTHOLD": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Breakaway
    "CDLBREAKAWAY": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Closing Marubozu
    "CDLCLOSINGMARUBOZU": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Concealing Baby Swallow
    "CDLCONCEALBABYSWALL": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Counterattack
    "CDLCOUNTERATTACK": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Dark Cloud Cover
    "CDLDARKCLOUDCOVER": {
        "parameters": {
            "penetration":          [0.5]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Doji
    "CDLDOJI": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Doji Star
    "CDLDOJISTAR": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Dragonfly Doji
    "CDLDRAGONFLYDOJI": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Engulfing Pattern
    "CDLENGULFING": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Evening Doji Star
    "CDLEVENINGDOJISTAR": {
        "parameters": {
            "penetration":          [0.3]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Evening Star
    "CDLEVENINGSTAR": {
        "parameters": {
            "penetration":          [0.3]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Up/Down-gap side-by-side white lines
    "CDLGAPSIDESIDEWHITE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Gravestone Doji
    "CDLGRAVESTONEDOJI": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Hammer
    "CDLHAMMER": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Hanging Man
    "CDLHANGINGMAN": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Harami Pattern
    "CDLHARAMI": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Harami Cross Pattern
    "CDLHARAMICROSS": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # High-Wave Candle
    "CDLHIGHWAVE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Hikkake Pattern
    "CDLHIKKAKE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 200"
    },
    # Modified Hikkake Pattern
    "CDLHIKKAKEMOD": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Homing Pigeon
    "CDLHOMINGPIGEON": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Identical Three Crows
    "CDLHOMINGPIGEON": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # In-Neck Pattern
    "CDLINNECK": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Inverted Hammer
    "CDLINVERTEDHAMMER": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Kicking
    "CDLKICKING": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Kicking - bull/bear determined by the longer marubozu
    "CDLKICKINGBYLENGTH": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Ladder Bottom
    "CDLLADDERBOTTOM": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Long Legged Doji
    "CDLLONGLEGGEDDOJI": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Long Line Candle
    "CDLLONGLINE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Marubozu
    "CDLMARUBOZU": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Matching Low
    "CDLMATCHINGLOW": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Mat Hold
    "CDLMATHOLD": {
        "parameters": {
            "penetration":          [0.5]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Morning Doji Star
    "CDLMORNINGDOJISTAR": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Morning Star
    "CDLMORNINGSTAR": {
        "parameters": {
            "penetration":          [0.3]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # On-Neck Pattern
    "CDLONNECK": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Piercing Pattern
    "CDLPIERCING": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Rickshaw Man
    "CDLRICKSHAWMAN": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Rising/Falling Three Methods
    "CDLRISEFALL3METHODS": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Separating Lines
    "CDLSEPARATINGLINES": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
     # Shooting Star
    "CDLSHOOTINGSTAR": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Short Line Candle
    "CDLSHORTLINE": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Spinning Top
    "CDLSPINNINGTOP": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Stalled Pattern
    "CDLSTALLEDPATTERN": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Stick Sandwich
    "CDLSTICKSANDWICH": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Takuri (Dragonfly Doji with very long lower shadow)
    "CDLTAKURI": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Tasuki Gap
    "CDLTASUKIGAP": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Thrusting Pattern
    "CDLTHRUSTING": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Tristar Pattern
    "CDLTRISTAR": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Unique 3 River
    "CDLUNIQUE3RIVER": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Upside Gap Two Crows
    "CDLUPSIDEGAP2CROWS": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    },
    # Upside/Downside Gap Three Methods
    "CDLXSIDEGAP3METHODS": {
        "parameters": {
            "None":                 [None]
        },
        "output_names":             ['integer'],
        "output_names_normalized":  ['integer'],
        "normalization":            "out / 100"
    }
}