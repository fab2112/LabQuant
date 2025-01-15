# LabQuant with Machine Learning - Demo1 - RandomForestClassifier
import os
import pickle
import numpy as np
import pandas as pd
from labquant import LabQuant
from sklearn.ensemble import RandomForestClassifier


"""This strategy should only be used for LabQuant demonstration and testing purposes, so do not use it
for decision making in the real market."""


# Load DataFrame (Bitmex OHLCV)
df_ohlcv = pd.read_csv("../../dataset/OHLCV_XBTUSD_1d__25-09-2015__02-08-2024_BITMEX.csv")

# Set features
df_ohlcv["feature_1"] = df_ohlcv.c / df_ohlcv.c.rolling(7).mean()
df_ohlcv["feature_2"] = df_ohlcv.c / df_ohlcv.c.rolling(17).mean()
df_ohlcv["feature_3"] = df_ohlcv.c / df_ohlcv.c.rolling(27).mean()
df_ohlcv["feature_4"] = df_ohlcv.c / df_ohlcv.c.rolling(117).mean()
df_ohlcv["feature_5"] = df_ohlcv.c / df_ohlcv.c.rolling(177).mean()

# Set target
time_ref = 12
df_ohlcv["target"] = df_ohlcv.c.pct_change(time_ref).shift(-time_ref)

# Discretize target
pct_ref = 5 / 100  # percentage threshold
df_ohlcv.target = np.where(df_ohlcv.target >= pct_ref, 1, np.where(df_ohlcv.target < -pct_ref, 0, np.nan))
df_ohlcv.target = df_ohlcv.target.ffill()
df_ohlcv = df_ohlcv.dropna()
df_ohlcv.reset_index(drop=True, inplace=True)

# Set features & target
features = df_ohlcv[["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]]
target = df_ohlcv.target

# Train test split
splitmark = int(0.7 * target.shape[0])
X_train = features[:splitmark].values
y_train = target[:splitmark].values
X_test = features[splitmark:].values
y_test = target[splitmark:].values

# Build model
def build_model():
    SEED = 11
    model = RandomForestClassifier(
        max_depth=100,
        max_features="log2",
        max_leaf_nodes=51,
        min_samples_leaf=9,
        min_samples_split=182,
        n_estimators=115,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    
    # Save model
    if not os.path.exists("Saved_models"):
        os.makedirs("Saved_models")
    pickle.dump(model, open("./Saved_models/rf_model_classifier_01.sav", "wb"))
    
# Build model
build_model()    

# Load model
model = pickle.load(open("./Saved_models/rf_model_classifier_01.sav", "rb"))

# Score metrics
print(f"\nScore >> Train: {model.score(X_train, y_train)} | Test: {model.score(X_test, y_test)}")


# Strategy
def strategy(args):
    
    df = args[0].copy()
    amount = args[1]
    model = args[2]

    features = df[["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]].values

    y_true = df.target
    y_pred = model.predict(features)

    df["true"] = np.where(y_true == 1, amount, 0)
    df["pred"] = np.where(y_pred == 1, amount, 0)

    return df


# Amount reference
amount = 1000
# Data preparation for LabQuant
df_ohlcv_test = df_ohlcv[splitmark:]
str_params = [df_ohlcv_test, amount, model]
data = strategy(str_params)

# Lab start
lab = LabQuant(data, seed=10)
lab.start()
