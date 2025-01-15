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
    df["SMA1_PLT2_BLUE"] = df.SMA1.values
    df["SMA2_PLT2_RED"] = df.SMA2.values
    return df

amount = 1
str_params = [df, amount, 44, 120]
data = strategy(str_params)

@pytest.fixture(scope="class")
def lab_inst():
    # Instance creation
    lab_instance = LabQuant(data, seed=1, show_roi=True)
    lab_instance.test_ = True
    lab_instance.start(
        opers_fee=1,
        stop_rate=2,
        gain_rate=2,
        str_params=str_params,
    )
    # Returns the instance to the test
    yield lab_instance
    # Del instance
    del lab_instance


class TestLab2:
    
    def test_price(self, lab_inst):
        assert sum(lab_inst.df_1.c) == 10082426.5  # Atualize com o valor correto

    def test_strategy_returns_pred(self, lab_inst):
        assert round(sum(lab_inst.strategy_returns_pred), 4) == -0.0368  # Atualize com o valor correto
    
    def test_strategy_returns_true(self, lab_inst):
        assert round(sum(lab_inst.strategy_returns_true), 4) == -0.0268  # Atualize com o valor correto

    def test_positions(self, lab_inst):
        assert sum(lab_inst.df_1.positions_pred[lab_inst.df_1.positions_pred > 0]) == 0  # Atualize com o valor correto
        assert sum(lab_inst.df_1.positions_pred[lab_inst.df_1.positions_pred < 0]) == -19  # Atualize com o valor correto

    def test_signals(self, lab_inst):
        assert sum(abs(lab_inst.df_1.signals_pred)) == 2  # Atualize com o valor correto

    def test_signals_size_pred(self, lab_inst):
        assert sum(lab_inst.df_1.signals_size_pred[lab_inst.df_1.signals_size_pred > 0]) == 1.0  # Atualize com o valor correto
        assert sum(lab_inst.df_1.signals_size_pred[lab_inst.df_1.signals_size_pred < 0]) == -1.0  # Atualize com o valor correto
        
    def test_signals_size_true(self, lab_inst):
        assert sum(lab_inst.df_1.signals_size_true[lab_inst.df_1.signals_size_true > 0]) == 1.0  # Atualize com o valor correto
        assert sum(lab_inst.df_1.signals_size_true[lab_inst.df_1.signals_size_true < 0]) == -1.0  # Atualize com o valor correto
        
    def test_hitrate_textitem(self, lab_inst):
        assert lab_inst.hit_trads_textitem.toPlainText() == "HIT-RATE: 0.0%     n-HITS: 0     n-LOSSES: 1    n-TRADS: 1 "  
        
    def test_show_distribution(self, lab_inst):
        lab_inst._show_pricedistribution()
        assert lab_inst.dist_textitem.toPlainText() == "MEAN: 50161.3     MEDIAN: 49460.0     ASYMMETRY: 0.5    STD: 4203.5 "  
    
    def test_drawdown_textitem(self, lab_inst):
        lab_inst.showplt1 = 2
        lab_inst._show_plot()
        assert lab_inst.drawdown_textitem.toPlainText() == "DD-MAX: 3.90%     DD-DURATION: 193"  
        
    def test_drawdowns(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.drawdown[0]), 4) == -7.3622  
    
    def test_drawdowns_duration(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.drawdown[1]), 4) == 18742.0  
        
    def test_pnl_textitem(self, lab_inst):
        lab_inst._show_plot()
        assert lab_inst.pnl_textitem.toPlainText() == "PNL: -3.68%"  
        
    def test_market_returns_cum(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.market_returns_cum), 4) == -21.2825  
        
    def test_strategy_returns_cum(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.strategy_returns_cum), 4) == -7.01  
    
    def test_equity_curve_true(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.equity_curve_true), 4) == 196.0  
    
    def test_equity_curve_pred(self, lab_inst):
        lab_inst._show_plot()
        assert round(sum(lab_inst.equity_curve_pred), 4) == 193.99  
    
    def test_risk_metrics_textitem(self, lab_inst):
        lab_inst._show_plot()
        assert lab_inst.risk_metrics_textitem.toPlainText() == "SHARPE: -3.64     SORTINO: -2.17     CALMAR: -1.71 "  
        
    def test_show_returns_1(self, lab_inst):
        lab_inst.showplt11 = 1
        lab_inst._show_returns()
        assert lab_inst.value_var_hist_axis.value == 0
   
    def test_show_returns_2(self, lab_inst):
        lab_inst.showplt11 = 2
        lab_inst._show_returns()
        assert lab_inst.value_var_hist_axis.value == 10
        assert round(sum(lab_inst.strategy_returns_pred), 4) == -0.0368  
        assert round(sum(lab_inst.strategy_returns_true), 4) == -0.0268  
        assert lab_inst.returns_textitem.toPlainText() == "STRATEGY  (MEAN: -0.02%  STD: 0.24%)     "
        
    def test_show_returns_3(self, lab_inst):
        lab_inst.showplt11 = 3
        lab_inst._show_returns()
        assert lab_inst.value_var_hist_axis.value == 10
        assert round(sum(lab_inst.trads_pct_changes), 4) == -0.0266  
        assert lab_inst.returns_textitem.toPlainText() == "n-TRADS: 1  (MEAN: -2.66%  STD: 0.00%)     "
        
    def test_show_returns_4(self, lab_inst):
        lab_inst.showplt11 = 4
        lab_inst._show_returns()
        assert lab_inst.value_var_hist_axis.value == 10
        
    def test_show_returns_5(self, lab_inst):
        lab_inst.showplt11 = 5
        lab_inst._show_returns()
        assert lab_inst.value_var_hist_axis.value == 10
        assert lab_inst.returns_textitem.toPlainText() == "P: (MEAN: 0.00%  STD: 0.00%)  L: (MEAN: 2.66%  STD: 0.00%)  "  
        
    def test_show_returns_6(self, lab_inst):
        lab_inst.showplt11 = 6
        lab_inst._show_returns()
        assert round(sum(lab_inst.strategy_returns_cum), 4) == -7.01  
        assert round(sum(lab_inst.market_returns_cum), 4) == -21.2825  
        assert lab_inst.value_var_hist_axis.value == 0
        
    def test_show_cumulative_gains_1(self, lab_inst):
        lab_inst.showplt4 = 1
        lab_inst._show_cumulative_gains()
        assert round(sum(lab_inst.df_1.cumul_gains_str.fillna(0)), 4) == 193.99  
        assert round(sum(lab_inst.df_1.cumul_gains_hold.fillna(0)), 4) == 178.7073 
        
    def test_show_cumulative_gains_2(self, lab_inst):
        lab_inst.showplt4 = 2
        lab_inst._show_cumulative_gains()
        assert round(sum(lab_inst.df_1.cumul_gains_reappl_str.fillna(0)), 4) == 194.0076  
        assert round(sum(lab_inst.df_1.cumul_gains_reappl_hold.fillna(0)), 4) == 179.4154  

        
        
    
