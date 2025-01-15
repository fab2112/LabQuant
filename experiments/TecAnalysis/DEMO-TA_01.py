# LabQuant with Tecnhical Analysis - Demo1 - Simple Strategy
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
    df["positions"] = np.where(df.SMA5 > df.SMA20, amount, -amount)

    # Set features plot
    df["SMA5_PLT2_BLUE"] = df.SMA5.values
    df["SMA20_PLT2_RED"] = df.SMA20.values
    df["POSITIONS_PLT3_YELLOW"] = df.positions.values 

    return df


# Amount reference
amount = 1000

# Data preparation for LabQuant
str_params = [df_ohlcv, amount]
data = strategy(str_params)

# Lab start
lab = LabQuant(data)
lab.start(show_candles=True)
