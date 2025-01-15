# LabQuant - A visual tool to support the development of algo-strategies in Quantitative Finance - by fab2112
import datetime
import traceback
import pandas as pd
import pyqtgraph as pg

from colorama import Fore


class DatetimeAxisX2(pg.AxisItem):
    """
    This class reconfigure and rewrites a datetime string on axis "x" of plot "plt_2".
    """

    def __init__(self, *args, **kwargs):
        super(DatetimeAxisX2, self).__init__(*args, **kwargs)
        self.data = kwargs.get("data")
        self.value_var_hist_axis = kwargs.get("value_var_hist_axis")
        self.value_var_time_axis = kwargs.get("value_var_time_axis")

    def tickStrings(self, values, scale, spacing):
        def timestamp_calc(data: pd.Series, n: float):
            """
            Manage and limit timestamps on axes.

            Args:
                data (pandas series): Timestamp series.
                n (float): Plot axes values.
            """
            time_zero = data[0]
            interval = data[1] - time_zero
            new_timestamp = time_zero + interval * (n - 1)
            if new_timestamp < 31536000000:
                new_timestamp = 31536000000
            elif new_timestamp > 2524608000000:
                new_timestamp = 2524608000000
            return new_timestamp

        try:
            # Returns - percentage
            if self.value_var_hist_axis.value == 10:
                ax_values = [str(round(value, 3) * 100) for value in values]

            # Price distibutions
            elif self.value_var_hist_axis.value == 20:
                ax_values = [str(value) for value in values]

            else:
                if self.value_var_time_axis.value == 0:
                    ax_values = [
                        (
                            datetime.datetime.fromtimestamp(
                                int(timestamp_calc(self.data, value)) / 1000
                            ).strftime("%b-%-y")
                        ).capitalize()
                        for value in values
                    ]
                else:
                    ax_values = [str(value) for value in values]

        except Exception as e:
            exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
            exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
            track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
            print(
                f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}{Fore.RESET}"
            )
            ax_values = []
            # raise sys.exc_info()[0]

        return ax_values


class DatetimeAxisX3(pg.AxisItem):
    """
    This class reconfigure and rewrites a datetime string on axis "x" of plot "plt_3".
    """

    def __init__(self, *args, **kwargs):
        super(DatetimeAxisX3, self).__init__(*args, **kwargs)
        self.data = kwargs.get("data")
        self.value_var_hist_axis = kwargs.get("value_var_hist_axis")
        self.value_var_time_axis = kwargs.get("value_var_time_axis")

    def tickStrings(self, values, scale, spacing):
        def timestamp_calc(data: pd.Series, n: float):
            """
            Manage and limit timestamps on axes.

            Args:
                data (pandas series): Timestamp series.
                n (float): Plot axes values.
            """
            time_zero = data[0]
            interval = data[1] - time_zero
            new_timestamp = time_zero + interval * (n - 1)
            if new_timestamp < 31536000000:
                new_timestamp = 31536000000
            elif new_timestamp > 2524608000000:
                new_timestamp = 2524608000000
            return new_timestamp

        try:
            # Price distibutions
            if self.value_var_hist_axis.value == 20:
                ax_values = [str(value) for value in values]

            else:
                if self.value_var_time_axis.value == 0:
                    ax_values = [
                        (
                            datetime.datetime.fromtimestamp(
                                int(timestamp_calc(self.data, value)) / 1000
                            ).strftime("%b-%-y")
                        ).capitalize()
                        for value in values
                    ]
                else:
                    ax_values = [str(value) for value in values]

        except Exception as e:
            exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
            exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
            track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
            print(
                f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}{Fore.RESET}"
            )
            ax_values = []
            # raise sys.exc_info()[0]

        return ax_values
