import os
import pytest
import numpy as np
import pandas as pd
from skopt.space import Integer
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
        strategy=strategy,
        str_params=str_params,
        mc_nsim=10,
        mc_nsteps=len(df),
        sim_taskmode='process',
    )
    # Returns the instance to the test
    yield lab_instance
    # Del instance
    del lab_instance


class TestLab3:
        
    def test_show_monte_carlo_simulation_1(self, lab_inst):
        lab_inst.showplt6 = 1
        lab_inst.mc_mode = "random_prices_price_base"
        lab_inst._show_monte_carlo_simulation()
        while True:
            if lab_inst.value_var_mc.value == 1:
                lab_inst.value_var_mc.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_1[:][:, -1])), 4) == 8.2157
        assert round(sum(np.array(lab_inst.np_mem_2[:][:, -1])), 4) == -4.253
        
    def test_show_monte_carlo_simulation_2(self, lab_inst):
        lab_inst.showplt6 = 1
        lab_inst.mc_mode = "random_prices_black_scholes"
        lab_inst._show_monte_carlo_simulation()
        while True:
            if lab_inst.value_var_mc.value == 1:
                lab_inst.value_var_mc.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_1[:][:, -1])), 4) == 25.0994
        assert round(sum(np.array(lab_inst.np_mem_2[:][:, -1])), 4) == -2.4047
        
    def test_show_monte_carlo_simulation_3(self, lab_inst):
        lab_inst.showplt6 = 1
        lab_inst.mc_mode = "random_prices_merton_jump_diffusion"
        lab_inst._show_monte_carlo_simulation()
        while True:
            if lab_inst.value_var_mc.value == 1:
                lab_inst.value_var_mc.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_1[:][:, -1])), 4) == 24.3394
        assert round(sum(np.array(lab_inst.np_mem_2[:][:, -1])), 4) == -2.6189
        
    def test_show_monte_carlo_simulation_4(self, lab_inst):
        lab_inst.showplt6 = 1
        lab_inst.mc_mode = "random_positions"
        lab_inst._show_monte_carlo_simulation()
        while True:
            if lab_inst.value_var_mc.value == 1:
                lab_inst.value_var_mc.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_1[:][:, -1])), 4) == 9.8833
        assert round(sum(np.array(lab_inst.np_mem_2[:][:, -1])), 4) == -2.4916
           
    def test_show_monte_carlo_simulation_5(self, lab_inst):
        lab_inst.showplt6 = 1
        lab_inst.mc_mode = "random_returns"
        lab_inst._show_monte_carlo_simulation()
        while True:
            if lab_inst.value_var_mc.value == 1:
                lab_inst.value_var_mc.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_1[:][:, -1])), 4) == 12.9769
        assert round(sum(np.array(lab_inst.np_mem_2[:][:, -1])), 4) == -1.7261
        
    def test_show_monte_carlo_simulation_6(self, lab_inst):
        lab_inst.showplt6 = 1
        lab_inst.mc_mode = "random_returns_with_replacement"
        lab_inst._show_monte_carlo_simulation()
        while True:
            if lab_inst.value_var_mc.value == 1:
                lab_inst.value_var_mc.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_1[:][:, -1])), 4) == 10.299
        assert round(sum(np.array(lab_inst.np_mem_2[:][:, -1])), 4) == -2.7573
        
    def test_show_monte_carlo_simulation_7(self, lab_inst):
        lab_inst.showplt6 = 1
        lab_inst.mc_mode = "random_endings_positions"
        lab_inst._show_monte_carlo_simulation()
        while True:
            if lab_inst.value_var_mc.value == 1:
                lab_inst.value_var_mc.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_1[:][:, -1])), 4) == 13.2798
        assert round(sum(np.array(lab_inst.np_mem_2[:][:, -1])), 4) == -0.5622
        
    def test_show_monte_carlo_simulation_8(self, lab_inst):
        lab_inst.showplt6 = 1
        lab_inst.mc_mode = "random_startings_positions"
        lab_inst._show_monte_carlo_simulation()
        while True:
            if lab_inst.value_var_mc.value == 1:
                lab_inst.value_var_mc.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_1[:][:, -1])), 4) == 13.1867
        assert round(sum(np.array(lab_inst.np_mem_2[:][:, -1])), 4) == -0.7773
    
    def test_show_params_simulation_1_process(self, lab_inst):
        lab_inst.showplt12 = 1
        lab_inst.sim_method = "grid"
        lab_inst.sim_taskmode="process"
        lab_inst.sim_params={"param1": [10, 20, 30], "param2": [60, 65, 70]}
        lab_inst._show_hypparams_simulation()
        while True:
            if lab_inst.value_var_sim.value == 1:
                lab_inst.value_var_sim.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_3[:][:, -1])), 4) == -145.0846
        assert round(sum(np.array(lab_inst.np_mem_4[:][:, -1])), 4) == 10.7497
        assert round(sum(lab_inst.sim_params_queue.get()[:, -1]), 4) == 585.0
        
    def test_show_params_simulation_2_process(self, lab_inst):
        lab_inst.showplt12 = 1
        lab_inst.sim_method = "random"
        lab_inst.sim_taskmode="process"
        lab_inst.sim_nrandsims=5
        lab_inst.sim_params={"param1": [5, 10, 15, 20, 25, 30],
                        "param2": [40, 45, 50, 55, 60, 65, 70]}
        lab_inst._show_hypparams_simulation()
        while True:
            if lab_inst.value_var_sim.value == 1:
                lab_inst.value_var_sim.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_3[:][:, -1])), 4) == -127.0669
        assert round(sum(np.array(lab_inst.np_mem_4[:][:, -1])), 4) == 5.379
        assert round(sum(lab_inst.sim_params_queue.get()[:, -1]), 4) == 270.0
        
    def test_show_params_simulation_3_process(self, lab_inst):
        lab_inst.showplt12 = 1
        lab_inst.sim_method = "bayesian-opt"
        lab_inst.sim_taskmode="process"
        lab_inst.sim_bayesopt_ncalls = 10
        lab_inst.sim_bayesopt_kwargs =  {
            "acq_func": "gp_hedge",
            "acq_optimizer": "auto",
            "verbose": "True",
        }
        lab_inst.sim_bayesopt_spaces = [
            Integer(2, 50, name="sma_1_in"),
            Integer(51, 150, name="sma_2_in"),
        ]
        lab_inst._show_hypparams_simulation()
        while True:
            if lab_inst.value_var_sim.value == 1:
                lab_inst.value_var_sim.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_3[:][:, -1])), 4) == -118.1221
        assert round(sum(np.array(lab_inst.np_mem_4[:][:, -1])), 4) == 12.4702
        assert round(sum(lab_inst.sim_params_queue.get()[:, -1]), 4) == 1175.0
        
    def test_show_params_simulation_1_thread(self, lab_inst):
        lab_inst.showplt12 = 1
        lab_inst.sim_method = "grid"
        lab_inst.sim_taskmode="thread"
        lab_inst.sim_params={"param1": [10, 20, 30], "param2": [60, 65, 70]}
        lab_inst._show_hypparams_simulation()
        while True:
            if lab_inst.value_var_sim.value == 1:
                lab_inst.value_var_sim.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_3[:][:, -1])), 4) == -145.0846
        assert round(sum(np.array(lab_inst.np_mem_4[:][:, -1])), 4) == 10.7497
        assert round(sum(lab_inst.sim_params_queue.get()[:, -1]), 4) == 585.0
        
    def test_show_params_simulation_2_thread(self, lab_inst):
        lab_inst.showplt12 = 1
        lab_inst.sim_method = "random"
        lab_inst.sim_taskmode="thread"
        lab_inst.sim_nrandsims=5
        lab_inst.sim_params={"param1": [5, 10, 15, 20, 25, 30],
                        "param2": [40, 45, 50, 55, 60, 65, 70]}
        lab_inst._show_hypparams_simulation()
        while True:
            if lab_inst.value_var_sim.value == 1:
                lab_inst.value_var_sim.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_3[:][:, -1])), 4) == -127.0669
        assert round(sum(np.array(lab_inst.np_mem_4[:][:, -1])), 4) == 5.379
        assert round(sum(lab_inst.sim_params_queue.get()[:, -1]), 4) == 270.0
        
    def test_show_params_simulation_3_thread(self, lab_inst):
        lab_inst.showplt12 = 1
        lab_inst.sim_method = "bayesian-opt"
        lab_inst.sim_taskmode="thread"
        lab_inst.sim_bayesopt_ncalls = 10
        lab_inst.sim_bayesopt_kwargs =  {
            "acq_func": "gp_hedge",
            "acq_optimizer": "auto",
            "verbose": "True",
        }
        lab_inst.sim_bayesopt_spaces = [
            Integer(2, 50, name="sma_1_in"),
            Integer(51, 150, name="sma_2_in"),
        ]
        lab_inst._show_hypparams_simulation()
        while True:
            if lab_inst.value_var_sim.value == 1:
                lab_inst.value_var_sim.value = 0
                break
        assert round(sum(np.array(lab_inst.np_mem_3[:][:, -1])), 4) == -163.2657
        assert round(sum(np.array(lab_inst.np_mem_4[:][:, -1])), 4) == 11.509
        assert round(sum(lab_inst.sim_params_queue.get()[:, -1]), 4) == 914
