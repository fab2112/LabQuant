# LabQuant with Tecnhical Analysis - Demo2 - Monte Carlo analysis
import numpy as np
import pandas as pd
from labquant import LabQuant


"""This strategy should only be used for LabQuant demonstration and testing purposes, so do not use it
for decision making in the real market."""


# Load DataFrame (Bitmex OHLCV)
df_ohlcv = pd.read_csv("../../dataset/OHLCV_XBTUSD_1d__25-09-2015__02-08-2024_BITMEX.csv")


# Strategy
def strategy(args):
    
    df = args[0].copy()
    amount = args[1]
    
    # Strategy logic
    df["SMA5"] = df["c"].rolling(5).mean()
    df["SMA20"] = df["c"].rolling(20).mean()
    df["SMA100"] = df["c"].rolling(100).mean()
    df["positions"] = np.where((df.SMA20 > df.SMA100) & (df.SMA5 > df.SMA20), amount,
                               np.where((df.SMA20 < df.SMA100) & (df.SMA5 < df.SMA20), -amount, 0))
    
    # Set features plots
    df["SMA5_PLT2_BLUE"] = df.SMA5.values
    df["SMA20_PLT2_RED"] = df.SMA20.values
    df["SMA100_PLT2_YELLOW"] = df.SMA100.values

    return df

# Amount reference
amount = 1000

# Data preparation for LabQuant
str_params = [df_ohlcv, amount]
data = strategy(str_params)

# Lab init
lab = LabQuant(data, seed=10)
lab.start(
    strategy=strategy,
    str_params=str_params,
    mc_line_plots=True,
    mc_nsim=300,
    mc_paths_colors=True,
)

