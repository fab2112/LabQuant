import os
import pytest
import numpy as np
import pandas as pd
from labquant import LabQuant

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "ohlcv_data.csv")

df = pd.read_csv(csv_path)

# Main Strategy
def strategy(args):   
    df = args[0].copy()
    amount = args[1]
    df["SMA1"] = df.c.rolling(args[2]).mean()
    df["SMA2"] = df.c.rolling(args[3]).mean()
    df["pred"] = np.where(df.SMA1 > df.SMA2, amount, -amount)
    df["true"] = np.where(df.SMA1 > df.SMA2, amount, -amount)
    df["SMA1_PLT1_BLUE"] = df.SMA1.values
    df["PRED_PLT2_GREEN"] = df.SMA1.values
    df["SMA2_PLT3_RED"] = df.SMA2.values
    return df

amount = 1
str_params = [df, amount, 44, 120]
data = strategy(str_params)

@pytest.fixture(scope="class")
def lab_inst():
    # Instance creation
    lab_instance = LabQuant(data, seed=1, show_roi=False)
    lab_instance.test_ = True
    lab_instance.start(
        opers_fee=1,
        stop_rate=2,
        gain_rate=2
    )
    # Returns the instance to the test
    yield lab_instance
    # Del instance
    del lab_instance


class TestLab1:
    
    def test_price(self, lab_inst):
        assert sum(lab_inst.df_1.c) == 110643379.0
        
    def test_strategy_returns_pred(self, lab_inst):
        assert round(sum(lab_inst.strategy_returns_pred), 4) == -0.3209
    
    def test_strategy_returns_true(self, lab_inst):
        assert round(sum(lab_inst.strategy_returns_true), 4) == 0.0091

    def test_positions(self, lab_inst):
        assert sum(lab_inst.df_1.positions_pred[lab_inst.df_1.positions_pred > 0]) == 118.0
        assert sum(lab_inst.df_1.positions_pred[lab_inst.df_1.positions_pred < 0]) == -121.0

    def test_signals(self, lab_inst):
        assert sum(abs(lab_inst.df_1.signals_pred)) == 34

    def test_signals_size_pred(self, lab_inst):
        assert sum(lab_inst.df_1.signals_size_pred[lab_inst.df_1.signals_size_pred > 0]) == 17
        assert sum(lab_inst.df_1.signals_size_pred[lab_inst.df_1.signals_size_pred < 0]) == -17
        
    def test_signals_size_true(self, lab_inst):
        assert sum(lab_inst.df_1.signals_size_true[lab_inst.df_1.signals_size_true > 0]) == 17
        assert sum(lab_inst.df_1.signals_size_true[lab_inst.df_1.signals_size_true < 0]) == -17
        
    def test_hitrate_textitem(self, lab_inst):
        assert lab_inst.hit_trads_textitem.toPlainText() == "HIT-RATE: 24.2%     n-HITS: 8     n-LOSSES: 25    n-TRADS: 33 "
        
    def test_scores_textitem(self, lab_inst):
        assert lab_inst.scores_textitem.toPlainText() == "SCORE: 1.00     F1-SCORE: 1.00     "

    def test_show_distribution(self, lab_inst):
        lab_inst._show_pricedistribution()
        assert lab_inst.dist_textitem.toPlainText() == "MEAN: 55321.7     MEDIAN: 56199.2     ASYMMETRY: -0.5    STD: 4191.0 "
    
    def test_drawdown_textitem(self, lab_inst):
        lab_inst.showplt1 = 2
        lab_inst._show_plot()
        assert lab_inst.drawdown_textitem.toPlainText() == "DD-MAX: 32.30%     DD-DURATION: 1992"
        
    def test_drawdowns(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.drawdown[0]), 4) == -369.6303
    
    def test_drawdowns_duration(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.drawdown[1]), 4) == 1985049.0
        
    def test_pnl_textitem(self, lab_inst):
        lab_inst._show_plot()
        assert lab_inst.pnl_textitem.toPlainText() == "PNL: -32.09%"
        
    def test_market_returns_cum(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.market_returns_cum), 4) == 52.5716
        
    def test_strategy_returns_cum(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.strategy_returns_cum), 4) == -365.3629
    
    def test_equity_curve_true(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.equity_curve_true), 4) == 1966.4971
    
    def test_equity_curve_pred(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.equity_curve_pred), 4) == 1634.6371
    
    def test_risk_metrics_textitem(self, lab_inst):
        lab_inst._show_plot()
        assert lab_inst.risk_metrics_textitem.toPlainText() == "SHARPE: -2.68     SORTINO: -1.25     CALMAR: -0.18 "
        
    def test_show_returns_1(self, lab_inst):
        lab_inst.showplt11 = 1
        lab_inst._show_returns()
        assert lab_inst.value_var_hist_axis.value == 0
   
    def test_show_returns_2(self, lab_inst):
        lab_inst.showplt11 = 2
        lab_inst._show_returns()
        assert lab_inst.value_var_hist_axis.value == 10
        assert round(sum(lab_inst.strategy_returns_pred), 4) == -0.3209
        assert round(sum(lab_inst.strategy_returns_true), 4) == 0.0091
        assert lab_inst.returns_textitem.toPlainText() == "STRATEGY  (MEAN: -0.02%  STD: 0.31%)     "
        
    def test_show_returns_3(self, lab_inst):
        lab_inst.showplt11 = 3
        lab_inst._show_returns()
        assert lab_inst.value_var_hist_axis.value == 10
        assert round(sum(lab_inst.trads_pct_changes), 4) == 0.0091
        assert lab_inst.returns_textitem.toPlainText() == "n-TRADS: 33  (MEAN: 0.05%  STD: 2.78%)     "
        
    def test_show_returns_4(self, lab_inst):
        lab_inst.showplt11 = 4
        lab_inst._show_returns()
        assert lab_inst.value_var_hist_axis.value == 10
        
    def test_show_returns_5(self, lab_inst):
        lab_inst.showplt11 = 5
        lab_inst._show_returns()
        assert lab_inst.value_var_hist_axis.value == 10
        assert lab_inst.returns_textitem.toPlainText() == "P: (MEAN: 2.89%  STD: 1.10%)  L: (MEAN: 2.47%  STD: 0.28%)  "
        
    def test_show_returns_6(self, lab_inst):
        lab_inst.showplt11 = 6
        lab_inst._show_returns()
        assert round(sum(lab_inst.strategy_returns_cum), 4) == -365.3629
        assert round(sum(lab_inst.market_returns_cum), 4) == 52.5716
        assert lab_inst.value_var_hist_axis.value == 0
        
    def test_show_cumulative_gains_1(self, lab_inst):
        lab_inst.showplt4 = 1
        lab_inst._show_cumulative_gains()
        assert round(sum(lab_inst.df_1.cumul_gains_str.fillna(0)), 4) == 1634.6371
        assert round(sum(lab_inst.df_1.cumul_gains_hold.fillna(0)), 4) == 2051.5614
        
    def test_show_cumulative_gains_2(self, lab_inst):
        lab_inst.showplt4 = 2
        lab_inst._show_cumulative_gains()
        assert round(sum(lab_inst.df_1.cumul_gains_reappl_str.fillna(0)), 4) == 1663.3355
        assert round(sum(lab_inst.df_1.cumul_gains_reappl_hold.fillna(0)), 4) == 1968.8835
    
    def test_show_features(self, lab_inst):
        lab_inst.showplt3 = 1
        lab_inst._show_features()
        
        data_plt_1 = []
        for item in lab_inst.plt_1.listDataItems():
            y_data = item.yData
            data_plt_1.append(y_data)
        data_plt_1 = np.nan_to_num(data_plt_1, nan=0.0)
 
        data_plt_3 = []
        for item in lab_inst.plt_3.listDataItems():
            y_data = item.yData
            data_plt_3.append(y_data)
        data_plt_3 = np.nan_to_num(data_plt_3, nan=0.0)
        
        x_long_pred, _ = lab_inst.scatter_long_pred.getData()
        x_short_pred, _ = lab_inst.scatter_short_pred.getData()

        assert np.round(sum(data_plt_1[0]), 1) == 108346643.8
        assert np.round(sum(x_long_pred), 1) == 7707.0
        assert np.round(sum(x_short_pred), 1) == 8717.0
        assert np.round(sum(data_plt_3[0]), 1) == 104306792.0
        
        
    
