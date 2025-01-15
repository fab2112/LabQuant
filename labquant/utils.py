# LabQuant - A visual tool to support the development of algo-strategies in Quantitative Finance - by fab2112
import traceback
import numpy as np
import pandas as pd

from colorama import Fore
from decimal import Decimal
from typing import Callable


def apply_tax(
    tax: float | int, returns: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """
    Emulates and applies fees on market entry and exit positions.

    Args:
        tax (float | int): Tax of operation fee (%)
        returns (numpy array): Array of strategy returns based on strategy use defined
        positions (numpy array): Array of strategy positions based on strategy user defined

    Returns:
        np.ndaaray: The new strategy returns with applied fees.
    """
    try:
        fees_size = np.zeros(len(positions), dtype=int)
        position_diff = np.diff(positions)

        # Identify position changes
        fees_size[1:] = np.where(position_diff != 0, 1, 0)

        # Set first position
        fees_size[0] = 1 if positions[0] != 0 else 0

        # Set for two fees in signals change
        for i in range(len(position_diff)):
            if positions[i] * positions[i + 1] < 0:
                fees_size[i + 1] += 1

        oper_tax = float(tax) / 100
        returns -= fees_size * oper_tax
    except Exception as e:
        exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
        exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
        track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
        print(f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}")
        print("FAIL TO APPLY FEES ON RETURNS")
        # raise sys.exc_info()[0]

    return returns


def process_df(
    df_1: pd.DataFrame,
    pos_true: None | np.ndarray,
    pos_pred: np.ndarray,
    stop_rate: float | int | None,
    gain_rate: float | int | None,
    initial_pos: np.int64,
) -> pd.DataFrame:
    """
    Processes the dataframe df_1, applies stop_rate, gain_rate and sets positions, signals and signals_size.

    Args:
        df_1 (DataFrame): The main dataframe.
        pos_true (None | numpy array): Array of positions "true" based on strategy user defined.
        pos_pred (numpy array): Array of positions "pred" based on strategy user defined.
        stop_rate (float | int | None): Stop loss threshold (%).
        gain_rate (float | int | None): Take profit threshold (%).
        initial_pos (np.int64): First position entry based on strategy.

    Returns:
        pd.DataFrame: The dataframe df_1 processed.
    """
    df_1.reset_index(drop=True, inplace=True)

    pos_var_1 = 0
    pos_var_2 = 0
    pos_var_3 = 0
    pos_var_4 = 0
    c_ref_true = 0
    c_ref_pred = 0

    df_1["positions_pred_ref"] = np.nan
    df_1["positions_true_ref"] = np.nan
    df_1["signals_true"] = np.nan
    df_1["signals_pred"] = np.nan
    df_1["signals_true_scatter"] = 0
    df_1["signals_pred_scatter"] = 0
    df_1["positions_pred"] = np.nan
    df_1["positions_true"] = np.nan
    df_1["cumul_gains_str"] = 0.0
    df_1.loc[0, "cumul_gains_str"] = initial_pos
    df_1["cumul_gains_hold"] = 0.0
    df_1.loc[0, "cumul_gains_hold"] = initial_pos
    df_1["cumul_gains_reappl_str"] = 0.0
    df_1.loc[0, "cumul_gains_reappl_str"] = initial_pos
    df_1["cumul_gains_reappl_hold"] = 0.0
    df_1.loc[0, "cumul_gains_reappl_hold"] = initial_pos

    df_1.positions_true_ref = pos_true
    df_1.positions_pred_ref = pos_pred

    df_1["signals_size_pred"] = df_1.positions_pred_ref.diff()
    df_1.loc[df_1["signals_size_pred"].isna(), "signals_size_pred"] = df_1[
        "positions_pred_ref"
    ][0]
    df_1["signals_size_true"] = df_1.positions_true_ref.diff()
    df_1.loc[df_1["signals_size_true"].isna(), "signals_size_true"] = df_1[
        "positions_true_ref"
    ][0]
    df_1["signals_pred"] = np.where(
        df_1.signals_size_pred > 0, 1, np.where(df_1.signals_size_pred < 0, -1, 0)
    )
    df_1["signals_true"] = np.where(
        df_1.signals_size_true > 0, 1, np.where(df_1.signals_size_true < 0, -1, 0)
    )

    df_1["positions_true"] = df_1.positions_true_ref / initial_pos
    df_1["positions_pred"] = df_1.positions_pred_ref / initial_pos

    # Convertendo colunas do DataFrame para numpy arrays
    positions_true = df_1["positions_true"].values
    positions_pred = df_1["positions_pred"].values
    signals_true = df_1["signals_true"].values
    signals_pred = df_1["signals_pred"].values
    signals_true_scatter = df_1["signals_true_scatter"].values
    signals_pred_scatter = df_1["signals_pred_scatter"].values
    c_values = df_1["c"].values

    for index in range(len(df_1)):
        if pos_true is not None:
            # True
            if (
                pos_var_3 == 20
                and signals_true[index] != 0
                and positions_true[index] != 0
            ):
                pos_var_3 = 0
                pos_var_1 = 0
            if positions_true[index] > 0 and signals_true[index] != 0:
                pos_var_1 = 1
                signals_true_scatter[index] = 1
                if 0 < positions_true[index - 1] < positions_true[index]:
                    signals_true_scatter[index] = 8
                elif 0 < positions_true[index - 1] > positions_true[index]:
                    signals_true_scatter[index] = 6
                c_ref_true = c_values[index]
            elif positions_true[index] < 0 and signals_true[index] != 0:
                pos_var_1 = -1
                signals_true_scatter[index] = -1
                if 0 > positions_true[index - 1] > positions_true[index]:
                    signals_true_scatter[index] = -3
                elif 0 > positions_true[index - 1] < positions_true[index]:
                    signals_true_scatter[index] = -5
                c_ref_true = c_values[index]
            elif (
                positions_true[index] == 0
                and signals_true[index] == -1
                and pos_var_1 == 1
            ):
                pos_var_1 = 0
                signals_true_scatter[index] = -2
            elif (
                positions_true[index] == 0
                and signals_true[index] == 1
                and pos_var_1 == -1
            ):
                pos_var_1 = 0
                signals_true_scatter[index] = 2
            if pos_var_1 == -10 or pos_var_1 == 10:
                pos_var_3 = 20
                positions_true[index] = 0
                signals_true[index] = 0

        # Pred
        if pos_var_4 == 20 and signals_pred[index] != 0 and positions_pred[index] != 0:
            pos_var_4 = 0
            pos_var_2 = 0
        if positions_pred[index] > 0 and signals_pred[index] != 0:
            pos_var_2 = 1
            signals_pred_scatter[index] = 1
            if 0 < positions_pred[index - 1] < positions_pred[index]:
                signals_pred_scatter[index] = 8
            elif 0 < positions_pred[index - 1] > positions_pred[index]:
                signals_pred_scatter[index] = 6
            c_ref_pred = c_values[index]
        elif positions_pred[index] < 0 and signals_pred[index] != 0:
            pos_var_2 = -1
            signals_pred_scatter[index] = -1
            if 0 > positions_pred[index - 1] > positions_pred[index]:
                signals_pred_scatter[index] = -3
            elif 0 > positions_pred[index - 1] < positions_pred[index]:
                signals_pred_scatter[index] = -5
            c_ref_pred = c_values[index]
        elif (
            positions_pred[index] == 0 and signals_pred[index] == -1 and pos_var_2 == 1
        ):
            pos_var_2 = 0
            signals_pred_scatter[index] = -2
        elif (
            positions_pred[index] == 0 and signals_pred[index] == 1 and pos_var_2 == -1
        ):
            pos_var_2 = 0
            signals_pred_scatter[index] = 2
        if pos_var_2 == -10 or pos_var_2 == 10:
            pos_var_4 = 20
            positions_pred[index] = 0
            signals_pred[index] = 0

        if stop_rate is not None:
            if pos_true is not None:
                if pos_var_1 == 1 and c_values[index] < c_ref_true - (
                    (stop_rate / 100) * c_ref_true
                ):
                    signals_true_scatter[index] = -4
                    positions_true[index] = 0
                    c_ref_true = 0
                    pos_var_1 = 10
                if pos_var_1 == -1 and c_values[index] > c_ref_true + (
                    (stop_rate / 100) * c_ref_true
                ):
                    signals_true_scatter[index] = -4
                    positions_true[index] = 0
                    c_ref_true = 0
                    pos_var_1 = -10
            if pos_var_2 == 1 and c_values[index] < c_ref_pred - (
                (stop_rate / 100) * c_ref_pred
            ):
                signals_pred_scatter[index] = -4
                positions_pred[index] = 0
                c_ref_pred = 0
                pos_var_2 = 10
            if pos_var_2 == -1 and c_values[index] > c_ref_pred + (
                (stop_rate / 100) * c_ref_pred
            ):
                signals_pred_scatter[index] = -4
                positions_pred[index] = 0
                c_ref_pred = 0
                pos_var_2 = -10

        if gain_rate is not None:
            if pos_true is not None:
                if pos_var_1 == 1 and c_values[index] > c_ref_true + (
                    (gain_rate / 100) * c_ref_true
                ):
                    signals_true_scatter[index] = 4
                    positions_true[index] = 0
                    c_ref_true = 0
                    pos_var_1 = 10
                if pos_var_1 == -1 and c_values[index] < c_ref_true - (
                    (gain_rate / 100) * c_ref_true
                ):
                    signals_true_scatter[index] = 4
                    positions_true[index] = 0
                    c_ref_true = 0
                    pos_var_1 = -10
            if pos_var_2 == 1 and c_values[index] > c_ref_pred + (
                (gain_rate / 100) * c_ref_pred
            ):
                signals_pred_scatter[index] = 4
                positions_pred[index] = 0
                c_ref_pred = 0
                pos_var_2 = 10
            if pos_var_2 == -1 and c_values[index] < c_ref_pred - (
                (gain_rate / 100) * c_ref_pred
            ):
                signals_pred_scatter[index] = 4
                positions_pred[index] = 0
                c_ref_pred = 0
                pos_var_2 = -10

    # Converting numpy arrays back to DataFrame columns
    if pos_true is not None:
        df_1["positions_true"] = positions_true
        df_1["signals_true"] = signals_true
        df_1["signals_true_scatter"] = signals_true_scatter
    df_1["positions_pred"] = positions_pred
    df_1["signals_pred"] = signals_pred
    df_1["signals_pred_scatter"] = signals_pred_scatter

    if pos_true is not None:
        df_1["signals_size_true"] = (
            df_1.positions_true.diff().fillna(df_1.positions_true[0]) * initial_pos
        )
    df_1["signals_size_pred"] = (
        df_1.positions_pred.diff().fillna(df_1.positions_pred[0]) * initial_pos
    )

    if pos_true is not None:
        df_1["signals_true"] = np.where(
            df_1.signals_size_true > 0, 1, np.where(df_1.signals_size_true < 0, -1, 0)
        )
    df_1["signals_pred"] = np.where(
        df_1.signals_size_pred > 0, 1, np.where(df_1.signals_size_pred < 0, -1, 0)
    )

    return df_1


def process_mc_strategy(
    strategy_params: list, strategy_function: Callable, prices_array: np.ndarray
) -> pd.DataFrame:
    """
    Process the Monte Carlo prices sequences into user strategy.

    Args:
        strategy_params (list): Strategy parameters.
        strategy_function (Callable): A function that takes strategy parameters and returns a DataFrame.
        prices_array (np.ndarray): Array of prices sequences - Monte Carlo test.

    Returns:
        pd.DataFrame: The DataFrame returned by the strategy function.
    """
    # Set new close price
    strategy_params[0].loc[:, "c"] = prices_array
    # Set amount
    strategy_params[1] = 1
    return strategy_function(strategy_params)


def decimal_round(value: float, places: int) -> str:
    """
    Round float numbers based on the number of places and return in string format.

    Args:
        value (float): Float number
        places (int): Places number

    Returns:
        string: Returns the value rounded in string format.
    """
    try:
        rounded_val = str(Decimal(value).quantize(Decimal(f"1.{'0' * places}")))
    except Exception as e:
        exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
        exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
        track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
        print(f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}")
        print("FAILED TO APPLY DECIMAL ROUND")
        rounded_val = 0
        # raise sys.exc_info()[0]
    return rounded_val


def get_drawdowns(eqcurve: pd.Series) -> tuple:
    """
    Generate a drawdown and drawdown duration.

    Args:
        eqcurve (Series): Equity curve

    Returns:
        tuple: Returns a tuple of arrays with drawdown and drawdown duration.
    """
    s = np.array(eqcurve)
    highwatermark = np.maximum.accumulate(s)
    drawdown = -(highwatermark - s)
    drawdowndur = np.zeros(len(s))

    nonzero_drawdown_indices = np.where(drawdown != 0)[0]
    if len(nonzero_drawdown_indices) > 0:
        for start_idx in np.split(
            nonzero_drawdown_indices,
            np.where(np.diff(nonzero_drawdown_indices) != 1)[0] + 1,
        ):
            drawdowndur[start_idx] = np.arange(1, len(start_idx) + 1)
    return (drawdown, drawdowndur)


def get_equitycurve(mkt_returns: np.ndarray, str_positions: np.ndarray) -> np.ndarray:
    """
    Generate equity curve for Monte Carlo test.

    Args:
        mkt_returns (numpy array): Array of strategy returns.
        str_positions (numpy array): Array of stategy positions

    Returns:
        np.naarray: Returns a array with equity curve.
    """
    # Positions - Shift(1)
    positions = np.roll(str_positions, 1).astype(float)
    positions[0] = np.nan
    # Calculates equity returns by multiplying returns by positions
    equity_returns = mkt_returns * positions
    # Replaces NaN values ​​with 0
    equity_returns = np.nan_to_num(equity_returns)
    # Calculates the cumulative sum of equity returns
    equity_curves = np.cumsum(equity_returns)
    return equity_curves


def get_hitrate(signals: pd.Series, str_returns: pd.Series) -> tuple:
    """
    Generate the hitrate values.

    Args:
        signals (pandas series): Signals of market operations.
        str_returns (pandas series): Series of strategy returns.

    Returns:
        tuple: Returns a tuple of hitrate values.
    """
    try:
        if (signals != 0).sum() >= 2:
            trads = ((str_returns.cumsum())[signals != 0]).diff().fillna(0)
            trads = np.sign(trads).value_counts()
            trads.sort_index(inplace=True)
            n_hits = trads.get(1, 0)
            n_losses = trads.get(-1, 0)
            if n_hits == 0 and n_losses == 0:
                hit_rate = 0
            else:
                hit_rate = (n_hits / (n_losses + n_hits)) * 100
            n_trads = int(n_hits) + int(n_losses)
        else:
            n_hits = 0
            n_losses = 0
            hit_rate = 0
            n_trads = 0

        return (n_hits, n_losses, hit_rate, n_trads)

    except Exception as e:
        exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
        exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
        track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
        print(
            f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}{Fore.RESET}"
        )
        pass
        # raise sys.exc_info()[0]


def get_riskmetrics(
    period_metric: float | int,
    riskfree_metric: float | int,
    str_returns: pd.Series,
    drawdown: np.ndarray,
) -> tuple:
    """
    Generate the values of risk metrics (Sharpe-rate, Sortino-rate and Calmar-rate).

    Args:
        period_metric (float or int): Period parameter of Sharpe-ratio, Sortino-ratio and Calmar-ratio.
        riskfree_metric (float or int): Risk free parameter (%) of Sharpe-Ratio and Sortino-Ratio.
        str_returns (pandas series): Series of strategy returns.
        drawdown (numpay array): Array of strategy drawdown.

    Returns:
        tuple: Returns a tuple of risk metrics values.
    """
    try:
        # Sharpe Ratio
        mean = str_returns.mean() * period_metric - (riskfree_metric / 100)
        sigma = str_returns.std() * np.sqrt(period_metric)
        if sigma == 0 or np.isnan(sigma):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = mean / sigma

        # Sortino Ratio
        mean = str_returns.mean() * period_metric - (riskfree_metric / 100)
        std_neg = str_returns[str_returns < 0].std() * np.sqrt(period_metric)
        if std_neg == 0 or np.isnan(std_neg):
            sortino_ratio = 0.0
        else:
            sortino_ratio = mean / std_neg

        # Calmar Ratio
        mean = str_returns.mean() * period_metric
        if abs(drawdown.min()) == 0 or np.isnan(abs(drawdown.min())):
            calmar_ratio = 0.0
        else:
            calmar_ratio = mean / abs(drawdown.min())

        return (sharpe_ratio, sortino_ratio, calmar_ratio)

    except Exception as e:
        exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
        exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
        track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
        print(f"{Fore.LIGHTYELLOW_EX}{exception_type}{exception_message}{track_line}")
        return (0, 0, 0)


def get_mc_price_paths(
    seed: int,
    df_diff_factor: int,
    mc_mode: str,
    str_params: list,
    n_sim: int,
    n_steps: int,
    sigma: int | float = 0.5,
    s0: int | float = 20000,
    r: int | float = 0.5,
    dt: float = (1 / 365),
    lambda_: int | float = 0.1,
    mu_y: int | float = 0.02,
    sigma_y: int | float = 0.1,
) -> np.ndarray:
    """
    Generate a price path sequences for Monte Carlo test.

    Args:
        seed (int): Strategy reproductibility.
        df_diff_factor (int): The length difference factor between the strategy input data and the strategy output data.
        mc_mode (string): Monte Carlo test mode.
        str_params (list): The strategy parameters.
        n_sim (int): The number of simulations (lines)
        n_steps (int): The number of steps (len lines)
        sigma: (int | float): Monte Carlo test - Random price volatility (σ) (Black-Scholes and Merton Jump diffusion models).
        s0 (int | float): Monte Carlo test - Initial stock price (Black-Scholes and Merton Jump diffusion models).
        r (int | float): Monte Carlo test - Risk-free rate (Black-Scholes and Merton Jump diffusion models).
        dt (float): Monte Carlo test - Time step (Black-Scholes and Merton Jump diffusion models).
        lambda_ (int | float): Monte Carlo test - Jump intensity (λ) (Merton Jump diffusion model).
        mu_y (int | float): Monte Carlo test - Mean of jump sizes (μ_y) (Merton Jump diffusion model).
        sigma_y (int | float): Monte Carlo test - Standard deviation of jump sizes (σ_y) (Merton Jump diffusion model).

    Returns:
        np.ndarray: The array of price sequences - Monte Carlo test.
    """
    np.random.seed(seed)
    
    n_steps = n_steps + df_diff_factor

    if mc_mode == "random_prices_black_scholes":
        price_paths = (
            s0
            * np.exp(
                np.cumsum(
                    (r - 0.5 * sigma**2) * dt
                    + sigma * np.sqrt(dt) * np.random.standard_normal((n_steps, n_sim)),
                    axis=0,
                )
            )
        ).transpose()

    elif mc_mode == "random_prices_merton_jump_diffusion":
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.standard_normal((n_steps, n_sim))

        # Jump component
        jumps = np.random.poisson(lambda_ * dt, (n_steps, n_sim))
        jump_sizes = mu_y + sigma_y * np.random.standard_normal((n_steps, n_sim))
        jump_diffusion = jumps * jump_sizes

        price_paths = (
            s0
            * np.exp(
                np.cumsum(
                    drift + diffusion + jump_diffusion,
                    axis=0,
                )
            )
        ).transpose()

    elif mc_mode == "random_prices_price_base":

        def np_log_returns(sequence):
            log_returns = np.insert(np.diff(np.log(sequence)), 0, 0)
            return np.round(log_returns, 3)

        log_returns = np.tile(
            np_log_returns(str_params[0].c.values[:n_steps]), (n_sim, 1)
        )
        shuffled_log_returns = np.apply_along_axis(
            np.random.permutation, 1, log_returns
        )

        # Initialize the synthetic price matrix
        price_paths = np.zeros((n_sim, n_steps))

        # Set the initial price
        price_paths[:, 0] = str_params[0].c.values[0]

        # Recreate synthetic prices
        cumulative_returns = np.cumsum(shuffled_log_returns, axis=1)
        price_paths[:, 0:] = price_paths[:, 0][:, np.newaxis] * np.exp(
            cumulative_returns
        )

    elif (
        mc_mode == "random_positions"
        or mc_mode == "random_returns"
        or mc_mode == "random_returns_with_replacement"
        or mc_mode == "random_endings_positions"
        or mc_mode == "random_startings_positions"
    ):
        price = str_params[0].c.values[:n_steps]
        price_paths = np.tile(price, (n_sim, 1))

    return price_paths
