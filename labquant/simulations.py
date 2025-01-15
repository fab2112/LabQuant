# LabQuant - A visual tool to support the development of algo-strategies in Quantitative Finance - by fab2112
import os
import sys
import traceback
import numpy as np
import pandas as pd

from skopt import dump
from colorama import Fore
from typing import Callable
from threading import Thread
from skopt import gp_minimize
from multiprocessing import Process, sharedctypes, queues


class ProcessMonteCarlo(Process):
    """
    This class is a engine that process the Monte Carlo test in parallel mode.
    """

    def __init__(
        self,
        seed: int,
        mc_mode: str,
        strategy: Callable,
        str_params: list,
        mc_strategy: Callable,
        mc_rndnpositions: int,
        mc_paths: np.ndarray,
        equity_curves: Callable,
        value_var_mc: sharedctypes.Synchronized,
        np_mem_1: np.memmap,
        np_mem_2: np.memmap,
        drawdowns: Callable,
        df_1: pd.DataFrame,
        initial_position: int,
        df_diff_factor: int,
    ):
        """
        Initialization method.

        Args:
            seed (int): Strategy reproductibility.
            mc_mode (string): Monte Carlo test mode.
            strategy (callable): A user strategy function.
            str_params (list): Strategy parameters.
            mc_strategy (callable): The Monte Carlo strategy function
            mc_rndnpositions (int): Set the Window to randomize in "starting" or "ending" position modes
            mc_paths (numpay array): The array of price sequences.
            equity_curves (callable): This function generate equity curves for Monte Carlo tests.
            value_var_mc (sharedctypes.Synchronized): A multiprocessing shared synchronized value.
            np_mem_1 (numpy memory map): Numpy space for share equity curves results.
            np_mem_2 (numpy memory map): Numpy space for share drawdowns results.
            drawdowns (callable): This function generate drawdown for Monte Carlo tests.
            df_1 (pandas dataframe): The main dataframe.
            initial_position (int): First position entry based on strategy.
            df_diff_factor (int): The length difference factor between the strategy input data and the strategy output data.
        """
        Process.__init__(self)

        self.seed = seed
        self.mc_mode = mc_mode
        self.strategy = strategy
        self.str_params = str_params
        self.mc_strategy = mc_strategy
        self.mc_rndnpositions = (mc_rndnpositions,)
        self.mc_paths = mc_paths
        self.equity_curves = equity_curves
        self.value_var_mc = value_var_mc
        self.np_mem_1 = np_mem_1
        self.np_mem_2 = np_mem_2
        self.drawdowns = drawdowns
        self.df_1 = df_1
        self.initial_position = initial_position
        self.df_diff_factor = df_diff_factor

        self.n_sim = mc_paths.shape[0]
        self.n_steps = mc_paths.shape[1]

        # MC random
        np.random.seed(self.seed)

    def run(self):
        self.mc_calc()

    def np_pctchange(self, sequence):
        """
        Calculate a numpy percentage change.

        Args:
            sequence (numpy array): Prices sequence.

        Returns:
            np.array: Array of percentage changes.
        """
        pct_change = np.insert(np.diff(sequence) / sequence[:-1], 0, 0)
        return pct_change

    def random_endings_sequences(self, arr):
        """
        Generate a new positions with random endings sequences.

        Args:
            arr (numpy array): Numpy array sequence of strategy positions.

        Returns:
            np.array: Numpy array sequence of strategy positions with randomly changed ending.
        """
        max_change = self.mc_rndnpositions[0]
        new_arr = arr.copy()
        # Find indices where sequences change
        change_indices = np.where(arr[:-1] != arr[1:])[0] + 1

        # Iterate over the sequences to modify the end of each one
        for idx in change_indices:
            current_value = arr[idx - 1]
            next_value = arr[idx]

            if current_value != 0:
                # Randomly decide to increase or decrease the sequence
                change = np.random.randint(-max_change, max_change + 1)

                # Calculate new sequence size
                new_length = idx + change
                # Avoid negative values
                new_length = max(new_length, 0)
                # Avoid exceeding array length
                new_length = min(new_length, len(arr))

                # Apply change
                if current_value == 1:
                    new_arr[idx:new_length] = 1
                elif current_value == -1:
                    new_arr[idx:new_length] = -1
                # Restore next sequence
                new_arr[new_length:idx] = next_value

        return new_arr

    def random_startings_sequences(self, arr):
        """
        Generate a new positions with random startings sequences.

        Args:
            arr (numpy array): Numpy array sequence of strategy positions.

        Returns:
            np.array: Numpy array sequence of strategy positions with randomly changed starting.
        """
        max_change = self.mc_rndnpositions[0]
        new_arr = arr.copy()
        # Find indices where sequences change
        change_indices = np.where(arr[:-1] != arr[1:])[0] + 1

        # Add index zero at the beginning of the first sequence
        change_indices = np.insert(change_indices, 0, 0)

        # Iterate over the sequences to modify the start of each one
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]

            current_value = arr[start_idx]
            if current_value != 0:
                # Randomly decide to increase or decrease the sequence
                change = np.random.randint(-max_change, max_change + 1)

                # Calculate the new start of the sequence
                new_start_idx = start_idx + change
                # Avoid negative values
                new_start_idx = max(new_start_idx, 0)
                # Avoid exceeding the array length
                new_start_idx = min(new_start_idx, len(arr))

                # Apply change
                if new_start_idx < end_idx:
                    # Reset the old position
                    new_arr[start_idx:new_start_idx] = 0
                    if current_value == 1:
                        new_arr[new_start_idx:end_idx] = 1
                    elif current_value == -1:
                        new_arr[new_start_idx:end_idx] = -1

        return new_arr

    def mc_calc(self):
        """
        This function process the Monte Carlo tests.
        """
        positions = None
        s = None

        try:
            # Price generating model
            if (
                self.mc_mode == "random_prices_price_base"
                or self.mc_mode == "random_prices_black_scholes"
                or self.mc_mode == "random_prices_merton_jump_diffusion"
            ):
                # Set matrix price paths
                s = self.mc_paths

                dfs = [
                    self.mc_strategy(self.str_params, self.strategy, prices)
                    for prices in s
                ]
                positions = np.array(
                    [
                        df.positions.values
                        if "positions" in df.columns
                        else df.pred.values
                        for df in dfs
                    ]
                )
                # Returns
                returns = np.apply_along_axis(self.np_pctchange, 1, s)[
                    :, self.df_diff_factor :
                ]

                # Generate Equity curves
                equity_curves_ = np.array(
                    list(map(self.equity_curves, returns, positions))
                )

                equity_curves = (equity_curves_ + 1) * self.initial_position

            # Random positions
            elif self.mc_mode == "random_positions":
                # Set matrix price paths
                s = self.mc_paths

                # Random positions based in original positions distribution
                position_counts = list(self.df_1.positions_pred.value_counts())
                positions_unique = list(self.df_1.positions_pred.value_counts().index)
                choice_signals_prob = [
                    x / len(self.df_1.positions_pred.values) for x in position_counts
                ]

                positions = np.random.choice(
                    positions_unique,
                    self.n_sim * (self.n_steps - self.df_diff_factor),
                    p=choice_signals_prob,
                ).reshape(self.n_sim, (self.n_steps - self.df_diff_factor))

                # Returns
                returns = np.apply_along_axis(self.np_pctchange, 1, s)[
                    :, self.df_diff_factor :
                ]

                # Generate Equity curves
                equity_curves_ = np.array(
                    list(map(self.equity_curves, returns, positions))
                )
                equity_curves = (equity_curves_ + 1) * self.initial_position

            # Random returns
            elif self.mc_mode == "random_returns":
                # Set matrix price paths
                s = self.mc_paths

                dfs = [
                    self.mc_strategy(self.str_params, self.strategy, prices)
                    for prices in s
                ]
                positions = np.array(
                    [
                        df.positions.values
                        if "positions" in df.columns
                        else df.pred.values
                        for df in dfs
                    ]
                )

                # Set original returns matrix
                returns = np.round(np.apply_along_axis(self.np_pctchange, 1, s), 8)[
                    :, self.df_diff_factor :
                ]

                # Positions Shift(1)
                positions_s = np.roll(positions, 1, axis=1).astype(float)
                positions_s[:, 0] = np.nan

                # Set equity returns shuffled
                equity_returns = np.nan_to_num(positions_s * returns)
                equity_returns_shuffle = np.apply_along_axis(
                    np.random.permutation, 1, equity_returns
                )
                # Set raw equity-curves
                equity_curves_ = np.cumsum(equity_returns_shuffle, axis=1)

                # Generate Equity curves
                equity_curves = (equity_curves_ + 1) * self.initial_position

            # Random returns with replacement
            elif self.mc_mode == "random_returns_with_replacement":
                # Set matrix price paths
                s = self.mc_paths

                dfs = [
                    self.mc_strategy(self.str_params, self.strategy, prices)
                    for prices in s
                ]
                positions = np.array(
                    [
                        df.positions.values
                        if "positions" in df.columns
                        else df.pred.values
                        for df in dfs
                    ]
                )

                # Set original returns matrix
                returns = np.apply_along_axis(self.np_pctchange, 1, s)[
                    :, self.df_diff_factor :
                ]

                # Positions Shift(1)
                positions_s = np.roll(positions, 1, axis=1).astype(float)
                positions_s[:, 0] = np.nan

                # Set equity returns shuffled with replacement
                equity_returns = np.nan_to_num(positions_s * returns)
                equity_returns_shuffle = np.apply_along_axis(
                    lambda x: np.random.choice(x, size=x.size, replace=True),
                    1,
                    equity_returns,
                )

                # Set raw equity-curves
                equity_curves_ = np.cumsum(equity_returns_shuffle, axis=1)

                # Generate Equity curves
                equity_curves = (equity_curves_ + 1) * self.initial_position

            # Random returns with replacement
            elif (
                self.mc_mode == "random_endings_positions"
                or self.mc_mode == "random_startings_positions"
            ):
                # Set matrix price paths
                s = self.mc_paths

                dfs = [
                    self.mc_strategy(self.str_params, self.strategy, prices)
                    for prices in s
                ]
                positions = np.array(
                    [
                        df.positions.values
                        if "positions" in df.columns
                        else df.pred.values
                        for df in dfs
                    ]
                )

                if self.mc_mode == "random_endings_positions":
                    positions = np.apply_along_axis(
                        self.random_endings_sequences, 1, positions
                    )
                else:
                    positions = np.apply_along_axis(
                        self.random_startings_sequences, 1, positions
                    )

                # Returns
                returns = np.apply_along_axis(self.np_pctchange, 1, s)[:, self.df_diff_factor:]

                # Generate Equity curves
                equity_curves_ = np.array(
                    list(map(self.equity_curves, returns, positions))
                )
                equity_curves = (equity_curves_ + 1) * self.initial_position

            # Filter Drawdown - DDDuration not used
            drawdowns = (
                np.array(list(map(self.drawdowns, equity_curves_)))[:, :1]
            ).reshape(self.n_sim, (self.n_steps - self.df_diff_factor))

            # Load all data in temporary memory
            self.np_mem_1[:] = np.zeros(
                [self.n_sim, (self.n_steps - self.df_diff_factor)]
            )
            self.np_mem_1[:] = equity_curves[:]
            self.np_mem_2[:] = np.zeros(
                [self.n_sim, (self.n_steps - self.df_diff_factor)]
            )
            self.np_mem_2[:] = drawdowns[:]
            self.value_var_mc.value = 1

            del self.np_mem_1, self.np_mem_2

        except Exception as e:
            exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
            exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
            track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
            print(
                f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}{Fore.RESET}"
            )
            sys.exit()
            # raise sys.exc_info()[0]


class ProcessHypSimulations(Process):
    """
    This class is a engine that process the Hyperparameter search simulations in parallel mode.
    """

    def __init__(
        self,
        seed: int,
        df_1: pd.DataFrame,
        sim_method: str,
        sim_params: dict,
        sim_nbest: int,
        sim_nrandsims: int,
        strategy: Callable,
        initial_pos: int,
        returns: pd.Series,
        opers_fee: float | int,
        apply_tax: Callable,
        drawdowns: Callable,
        np_mem_3: np.memmap,
        np_mem_4: np.memmap,
        value_var_sim: sharedctypes.Synchronized,
        params_combinations: list,
        sim_bayesopt_ncalls: int,
        sim_bayesopt_spaces: list,
        sim_bayesopt_kwargs: dict,
        sim_params_queue: queues.Queue,
    ):
        """
        Initialization method.

        Args:
            seed (int): Strategy reproductibility.
            df_1 (pandas dataframe): The main dataframe.
            sim_method (string): Hyperparameter simulations - "grid", "random" or "bayesian-opt".
            sim_params (dict): Hyperparameter simulations - Strategy params for "grid" or "random".
            sim_nbest (int): Hyperparameter simulations - Number of best curves to show.
            sim_nrandsims (int): Hyperparameter simulations - Number of "random" simulations.
            strategy (callable): A user strategy function.
            initial_pos (int): First position entry based on strategy.
            returns (pandas series): Strategy market returns.
            opers_fee (float | int): Tax of operation fee (%).
            apply_tax (callable): Emulates and applies fees on market entry and exit positions.
            drawdowns (callable): This function generate drawdown for Monte Carlo tests.
            np_mem_3 (numpy memory map): Numpy space for share drawdowns results.
            np_mem_4 (numpy memory map): Numpy space for share equity curves results.
            value_var_sim (sharedctypes.Synchronized): A multiprocessing shared synchronized value.
            params_combinations (list): List of tuples with parameter combinations.
            sim_bayesopt_ncalls (int): Hyperparameter simulations - Bayesian-opt number of calls (scikit-optimize).
            sim_bayesopt_spaces (list): Hyperparameter simulations - Bayesian-opt spaces (scikit-optimize).
            sim_bayesopt_kwargs (dict): Hyperparameter simulations - Bayesian-opt kwargs (scikit-optimize).
            sim_params_queue (queues.Queue): A multiprocessing queue object to send parameters generated by simulations.
        """
        Process.__init__(self)
        self.seed = seed
        self.df_1 = df_1
        self.sim_method = sim_method
        self.sim_params = sim_params
        self.sim_nbest = sim_nbest
        self.sim_nrandsims = sim_nrandsims
        self.strategy = strategy
        self.initial_pos = initial_pos
        self.returns = returns
        self.opers_fee = opers_fee
        self.apply_tax = apply_tax
        self.drawdowns = drawdowns
        self.np_mem_3 = np_mem_3
        self.np_mem_4 = np_mem_4
        self.value_var_sim = value_var_sim
        self.params_combinations = params_combinations
        self.sim_bayesopt_ncalls = sim_bayesopt_ncalls
        self.sim_bayesopt_spaces = sim_bayesopt_spaces
        self.sim_bayesopt_kwargs = sim_bayesopt_kwargs
        self.sim_params_queue = sim_params_queue
        self.data_ = []
        self.sim_df = pd.DataFrame()

        # MC random
        np.random.seed(self.seed)

    def run(self):
        """
        This function runs the simulation or optimization process based on the defined method.
        """
        if self.sim_method == "bayesian-opt":
            self.process_bayesian_opt()
        elif self.sim_method == "grid" or self.sim_method == "random":
            self.process_simulations()

    def process_simulations(self):
        """
        This function process the hyperparameter simulations in modes "grid" or "random" and send the results to a numpy memory map.
        The simulated parameters "params" are collected and sent by the queue process.
        """
        for params in self.params_combinations:
            results = self.process_stretegy(params)
            equity_curves = results[0]
            drawdowns = results[1]

            # Append data
            self.data_.append((drawdowns, equity_curves, params))

        drawdowns = np.array([t[0] for t in self.data_])
        equity_curves = np.array([t[1] for t in self.data_])
        params = np.array([t[2] for t in self.data_], dtype=np.object_)

        # Load all data in temporary memory
        self.np_mem_3[:] = np.zeros([drawdowns.shape[0], drawdowns.shape[1]])
        self.np_mem_3[:] = drawdowns[:]
        #
        self.np_mem_4[:] = np.zeros([equity_curves.shape[0], equity_curves.shape[1]])
        self.np_mem_4[:] = equity_curves[:]
        #
        self.sim_params_queue.put(params)

        self.value_var_sim.value = 1

        del self.np_mem_3, self.np_mem_4

    def process_bayesian_opt(self):
        """
        This function process the hyperparameter search simulations in mode "bayesian-opt".
        """

        def get_bayesian_curve(results):
            """
            This function collect the results of bayesian-opt simulations and send to numpy memory map.
            The simulated parameters "params" are collected and sent by the queue process.
            """
            drawdowns = np.array([t[0] for t in self.data_])
            equity_curves = np.array([t[1] for t in self.data_])
            params = np.array([t[2] for t in self.data_], dtype=np.object_)

            # Load all data in temporary memory
            self.np_mem_3[:] = np.zeros([drawdowns.shape[0], drawdowns.shape[1]])
            self.np_mem_3[:] = drawdowns[:]
            #
            self.np_mem_4[:] = np.zeros(
                [equity_curves.shape[0], equity_curves.shape[1]]
            )
            self.np_mem_4[:] = equity_curves[:]
            #
            self.sim_params_queue.put(params)

            self.value_var_sim.value = 1

            del self.np_mem_3, self.np_mem_4

        def cost_function(params: list) -> float:
            """
            This function collects the parameters generated by the Gaussian process "gp_minimize" and executes them in the user-defined strategy.
            It appends the results [equity curves and drawdowns] to the "self.data_" attribute for each iteration (call).
            It calculates the cost metric and returns it to the "gp_minimize" engine.
            The cost can be customized by the user, creating a "cost_opt" attribute directly in the strategy dataframe ohlcv.
            
            Args:
                params (list): List of parameters generated by the Gaussian process "gp_minimize".
            
            Returns:
                float: Returns the cost for each iteration (call).
            """
            results = self.process_stretegy(params)

            equity_curve = results[0]
            drawdowns = results[1]

            equity_curve_final = equity_curve[-1]

            cost = -equity_curve_final

            # Append data
            self.data_.append((drawdowns, equity_curve, params))

            # External cost function
            if "cost_opt" in self.sim_df:
                cost = self.sim_df.cost_opt.values[-1]
                
            return cost

        try:
            # Run optimization
            results = gp_minimize(
                func=cost_function,
                dimensions=self.sim_bayesopt_spaces,
                n_calls=self.sim_bayesopt_ncalls,
                **self.sim_bayesopt_kwargs,
            )

            get_bayesian_curve(results)

            # Save model
            if not os.path.exists("./Saved_models"):
                os.makedirs("./Saved_models")
            dump(results, "./Saved_models/simopt_model.pkl", store_objective=False)

            print(f"\nBEST PARAMS: {results.x}")
            print(f"BEST COST: {round(results.fun * -1, 4)}")
            print(f"SEED: {self.seed}")

            # Save logs
            if not os.path.exists("./Sim_logs"):
                os.makedirs("./Sim_logs")
            with open("./Sim_logs/sim_opt_logs.txt", "w") as f:
                f.write(f"BEST PARAMS: {results.x}\n")
                f.write(f"BEST COST: {round(results.fun * -1, 4)}\n")
                f.write(f"SEED: {self.seed}")

        except Exception as e:
            exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
            exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
            track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
            print(
                f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}{Fore.RESET}"
            )
            # pass
            raise sys.exc_info()[0]

    def process_stretegy(self, params: list) -> list:
        """
        This function process and runs the parameters in the strategy to generates new equity and drawdown curves.
        
        Args:
            params (list): List of parameters for simulation on strategy.
            
        Returns:
            list: List of equity curve and drawdown.
        """
        self.sim_df = self.df_1.copy()
        self.sim_df = self.sim_df.iloc[:, : self.sim_df.columns.get_loc("dateindex")]

        # Select params for strategy function
        sim_str_params = []
        sim_str_params.append(self.sim_df)
        # Set amount
        sim_str_params.append(self.initial_pos)

        if self.sim_bayesopt_spaces is not None:
            for j in range(len(self.sim_bayesopt_spaces)):
                sim_str_params.append(params[j])
        else:
            for j in range(len(self.sim_params)):
                sim_str_params.append(params[j])

        # Exec strategy function
        self.sim_df = self.strategy(sim_str_params)

        # Sets the "positions" column if it does not exist
        if "positions" not in self.sim_df.columns:
            self.sim_df["positions"] = self.sim_df.pred

        # Set positions
        self.sim_df.positions = self.sim_df.positions / self.initial_pos
        # Set strategy returns
        strategy_returns_pred = (self.returns * self.sim_df.positions.shift(1)).fillna(
            0
        )

        # Apply Maker Fees
        if self.opers_fee is not None:
            positions_sig = self.sim_df.positions * self.initial_pos
            self.sim_df["signals_size"] = (
                self.sim_df.positions.diff() * self.initial_pos
            )
            self.sim_df["signals_size"] = self.sim_df["signals_size"].fillna(
                positions_sig[0]
            )

            str_returns_pred = self.apply_tax(
                self.opers_fee, strategy_returns_pred, positions_sig
            )
            strategy_returns_pred = pd.Series(str_returns_pred)

        # Calc equity curves cumulative gains
        equity_curve = (strategy_returns_pred.cumsum() + 1) * self.initial_pos

        # Calc drawndowns
        equity_curve_ = (strategy_returns_pred.cumsum()) + 1
        drawdowns = self.drawdowns(pd.Series(equity_curve_))
        drawdowns = drawdowns[0] * 100

        return [equity_curve.values, drawdowns]


class ThreadHypSimulations(Thread):
    """
    This class is a engine that process the Hyperparameter search simulations in concurrent mode.
    """

    def __init__(
        self,
        seed: int,
        df_1: pd.DataFrame,
        sim_method: str,
        sim_params: dict,
        sim_nbest: int,
        sim_nrandsims: int,
        strategy: Callable,
        initial_pos: int,
        returns: pd.Series,
        opers_fee: float | int,
        apply_tax: Callable,
        drawdowns: Callable,
        np_mem_3: np.memmap,
        np_mem_4: np.memmap,
        value_var_sim: sharedctypes.Synchronized,
        params_combinations: list,
        sim_bayesopt_ncalls: int,
        sim_bayesopt_spaces: list,
        sim_bayesopt_kwargs: dict,
        sim_params_queue: queues.Queue,
        value_stopthread_sig: sharedctypes.Synchronized,
    ):
        """
        Initialization method.

        Args:
            seed (int): Strategy reproductibility.
            df_1 (pandas dataframe): The main dataframe.
            sim_method (string): Hyperparameter simulations - "grid", "random" or "bayesian-opt".
            sim_params (dict): Hyperparameter simulations - Strategy params for "grid" or "random".
            sim_nbest (int): Hyperparameter simulations - Number of best curves to show.
            sim_nrandsims (int): Hyperparameter simulations - Number of "random" simulations.
            strategy (callable): A user strategy function.
            initial_pos (int): First position entry based on strategy.
            returns (pandas series): Strategy market returns.
            opers_fee (float | int): Tax of operation fee (%).
            apply_tax (callable): Emulates and applies fees on market entry and exit positions.
            drawdowns (callable): This function generate drawdown for Monte Carlo tests.
            np_mem_3 (numpy memory map): Numpy space for share drawdowns results.
            np_mem_4 (numpy memory map): Numpy space for share equity curves results.
            value_var_sim (sharedctypes.Synchronized): A multiprocessing shared synchronized value.
            params_combinations (list): List of tuples with parameter combinations.
            sim_bayesopt_ncalls (int): Hyperparameter simulations - Bayesian-opt number of calls (scikit-optimize).
            sim_bayesopt_spaces (list): Hyperparameter simulations - Bayesian-opt spaces (scikit-optimize).
            sim_bayesopt_kwargs (dict): Hyperparameter simulations - Bayesian-opt kwargs (scikit-optimize).
            sim_params_queue (queues.Queue): A multiprocessing queue object to send parameters generated by simulations.
            value_stopthread_sig (sharedctypes.Synchronized): A multiprocessing shared synchronized value.
        """
        Thread.__init__(self)
        self.seed = seed
        self.df_1 = df_1
        self.sim_method = sim_method
        self.sim_params = sim_params
        self.sim_nbest = sim_nbest
        self.sim_nrandsims = sim_nrandsims
        self.strategy = strategy
        self.initial_pos = initial_pos
        self.returns = returns
        self.opers_fee = opers_fee
        self.apply_tax = apply_tax
        self.drawdowns = drawdowns
        self.np_mem_3 = np_mem_3
        self.np_mem_4 = np_mem_4
        self.value_var_sim = value_var_sim
        self.params_combinations = params_combinations
        self.sim_bayesopt_ncalls = sim_bayesopt_ncalls
        self.sim_bayesopt_spaces = sim_bayesopt_spaces
        self.sim_bayesopt_kwargs = sim_bayesopt_kwargs
        self.sim_params_queue = sim_params_queue
        self.value_stopthread_sig = value_stopthread_sig
        self.data_ = []
        self.sim_df = pd.DataFrame()

        # MC random
        np.random.seed(self.seed)

    def run(self):
        """
        This function runs the simulation or optimization process based on the defined method.
        """
        if self.sim_method == "bayesian-opt":
            self.process_bayesian_opt()
        elif self.sim_method == "grid" or self.sim_method == "random":
            self.process_simulations()

    def process_simulations(self):
        """
        This function process the hyperparameter simulations in modes "grid" or "random" and send the results to a numpy memory map.
        The simulated parameters "params" are collected and sent by the queue process.
        """
        for params in self.params_combinations:
            results = self.process_stretegy(params)
            equity_curves = results[0]
            drawdowns = results[1]

            # Append data
            self.data_.append((drawdowns, equity_curves, params))

        drawdowns = np.array([t[0] for t in self.data_])
        equity_curves = np.array([t[1] for t in self.data_])
        params = np.array([t[2] for t in self.data_], dtype=np.object_)

        # Load all data in temporary memory
        self.np_mem_3[:] = np.zeros([drawdowns.shape[0], drawdowns.shape[1]])
        self.np_mem_3[:] = drawdowns[:]
        #
        self.np_mem_4[:] = np.zeros([equity_curves.shape[0], equity_curves.shape[1]])
        self.np_mem_4[:] = equity_curves[:]
        #
        self.sim_params_queue.put(params)

        self.value_var_sim.value = 1

        del self.np_mem_3, self.np_mem_4

    def process_bayesian_opt(self):
        """
        This function process the hyperparameter search simulations in mode "bayesian-opt".
        """
        
        def get_bayesian_curve(results):
            """
            This function collect the results of bayesian-opt simulations and send to numpy memory map.
            The simulated parameters "params" are collected and sent by the queue process.
            """
            drawdowns = np.array([t[0] for t in self.data_])
            equity_curves = np.array([t[1] for t in self.data_])
            params = np.array([t[2] for t in self.data_], dtype=np.object_)

            # Load all data in temporary memory
            self.np_mem_3[:] = np.zeros([drawdowns.shape[0], drawdowns.shape[1]])
            self.np_mem_3[:] = drawdowns[:]
            #
            self.np_mem_4[:] = np.zeros(
                [equity_curves.shape[0], equity_curves.shape[1]]
            )
            self.np_mem_4[:] = equity_curves[:]
            #
            self.sim_params_queue.put(params)

            self.value_var_sim.value = 1

            del self.np_mem_3, self.np_mem_4

        def cost_function(params):
            """
            This function collects the parameters generated by the Gaussian process "gp_minimize" and executes them in the user-defined strategy.
            It appends the results [equity curves and drawdowns] to the "self.data_" attribute for each iteration (call).
            It calculates the cost metric and returns it to the "gp_minimize" engine.
            The cost can be customized by the user, creating a "cost_opt" attribute directly in the strategy dataframe ohlcv.
            
            Args:
                params (list): List of parameters generated by the Gaussian process "gp_minimize".
            
            Returns:
                float: Returns the cost for each iteration (call).
            """
            results = self.process_stretegy(params)

            equity_curve = results[0]
            drawdowns = results[1]

            equity_curve_final = equity_curve[-1]

            cost = -equity_curve_final

            # Append data
            self.data_.append((drawdowns, equity_curve, params))

            if self.value_stopthread_sig.value == 1:
                self.value_stopthread_sig.value = 0
                raise InterruptedError()

            # External cost function
            if "cost_opt" in self.sim_df:
                cost = self.sim_df.cost_opt.values[-1]

            return cost

        try:
            # Run optimization
            results = gp_minimize(
                func=cost_function,
                dimensions=self.sim_bayesopt_spaces,
                n_calls=self.sim_bayesopt_ncalls,
                **self.sim_bayesopt_kwargs,
            )

            get_bayesian_curve(results)

            # Save model
            if not os.path.exists("./Saved_models"):
                os.makedirs("./Saved_models")
            dump(results, "./Saved_models/simopt_model.pkl", store_objective=False)

            print(f"\nBEST PARAMS: {results.x}")
            print(f"BEST COST: {round(results.fun * -1, 4)}")
            print(f"SEED: {self.seed}")

            # Save logs
            if not os.path.exists("./Sim_logs"):
                os.makedirs("./Sim_logs")
            with open("./Sim_logs/sim_opt_logs.txt", "w") as f:
                f.write(f"BEST PARAMS: {results.x}\n")
                f.write(f"BEST COST: {round(results.fun * -1, 4)}\n")
                f.write(f"SEED: {self.seed}")

        except InterruptedError:
            print(f"{Fore.LIGHTYELLOW_EX}STOP OPTIMIZATION!{Fore.RESET}")

        except Exception as e:
            exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
            exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
            track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
            print(
                f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}{Fore.RESET}"
            )
            pass
            # raise sys.exc_info()[0]

    def process_stretegy(self, params):
        """
        This function process and runs the parameters in the strategy to generates new equity and drawdown curves.
        
        Args:
            params (list): List of parameters for simulation on strategy.
            
        Returns:
            list: List of equity curve and drawdown.
        """
        self.sim_df = self.df_1.copy()
        self.sim_df = self.sim_df.iloc[:, : self.sim_df.columns.get_loc("dateindex")]

        # Select params for strategy function
        sim_str_params = []
        sim_str_params.append(self.sim_df)
        # Set amount
        sim_str_params.append(self.initial_pos)
        if self.sim_bayesopt_spaces is not None:
            for j in range(len(self.sim_bayesopt_spaces)):
                sim_str_params.append(params[j])
        else:
            for j in range(len(self.sim_params)):
                sim_str_params.append(params[j])

        # Exec strategy function
        self.sim_df = self.strategy(sim_str_params)

        # Sets the "positions" column if it does not exist
        if "positions" not in self.sim_df.columns:
            self.sim_df["positions"] = self.sim_df.pred

        # Set positions
        self.sim_df.positions = self.sim_df.positions / self.initial_pos
        # Set strategy returns
        strategy_returns_pred = (self.returns * self.sim_df.positions.shift(1)).fillna(
            0
        )

        # Apply Maker Fees
        if self.opers_fee is not None:
            positions_sig = self.sim_df.positions * self.initial_pos
            self.sim_df["signals_size"] = (
                self.sim_df.positions.diff() * self.initial_pos
            )
            self.sim_df["signals_size"] = self.sim_df["signals_size"].fillna(
                positions_sig[0]
            )

            str_returns_pred = self.apply_tax(
                self.opers_fee, strategy_returns_pred, positions_sig
            )
            strategy_returns_pred = pd.Series(str_returns_pred)

        # Calc equity curves cumulative gains
        equity_curve = (strategy_returns_pred.cumsum() + 1) * self.initial_pos

        # Calc drawndowns
        equity_curve_ = (strategy_returns_pred.cumsum()) + 1
        drawdowns = self.drawdowns(pd.Series(equity_curve_))
        drawdowns = drawdowns[0] * 100

        return [equity_curve.values, drawdowns]
