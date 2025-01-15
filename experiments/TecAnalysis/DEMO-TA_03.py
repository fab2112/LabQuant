# LabQuant with Tecnhical Analysis - Demo3 - Monte Carlo | Bayesian optimization
import talib as ta
import numpy as np
import pandas as pd
from labquant import LabQuant
from skopt.space import Integer


"""This strategy should only be used for LabQuant demonstration and testing purposes, so do not use it
for decision making in the real market."""


# Load Data
df_ohlcv = pd.read_csv("../../dataset/OHLCV_XBTUSD_1d__25-09-2015__02-08-2024_BITMEX.csv")


# Strategy
def strategy(args):
    
    df = args[0].copy()
    amount = args[1]

    # Strategy logic
    df["macd_sig"] = ta.MACD(df.c, fastperiod=args[2], slowperiod=args[3], signalperiod=args[4])[1]
    df["macd"] = ta.MACD(df.c, fastperiod=args[5], slowperiod=args[6], signalperiod=args[7])[0]
    df["positions"] = np.where(df.macd_sig > df.macd, amount, 0)

    # Set features plot
    df["macd_PLT1_BLUE"] = df.macd.values
    df["macdSig_PLT1_RED"] = df.macd_sig.values
    df["positions_PLT3_YELLOW"] = df.positions.values

    return df


# Amount reference
amount = 1000

# Data preparation for LabQuant
hyper_params = [2, 31, 36, 19, 22, 15]
str_params = [df_ohlcv, amount] + hyper_params
data = strategy(str_params)

# Bayesian optimization
# spaces
space = [
    Integer(2, 50, name="params-1"),
    Integer(2, 50, name="params-2"),
    Integer(2, 50, name="params-3"),
    Integer(2, 50, name="params-4"),
    Integer(2, 50, name="params-5"),
    Integer(2, 50, name="params-6"),
]
# kwargs
bayesopt_kwargs = {
    "random_state": 10,
    "acq_func": "gp_hedge",
    "acq_optimizer": "auto",
    "verbose": True,
    "n_jobs": 10,
}

# Lab start
lab = LabQuant(data, seed=10)
lab.start(
    strategy=strategy,
    str_params=str_params,
    mc_paths_colors=True,
    mc_line_plots=True,
    mc_nsim=200,
    sim_method="bayesian-opt",
    sim_bayesopt_ncalls=20,
    sim_bayesopt_spaces=space,
    sim_bayesopt_kwargs=bayesopt_kwargs,
)

