# LabQuant - A visual tool to support the development of algo-strategies in Quantitative Finance - by fab2112
import re
import os
import sys
import datetime
import tempfile
import traceback
import itertools
import numpy as np
import pandas as pd
import pyqtgraph as pg
import multiprocessing as mp

from scipy import stats
from colorama import Fore
from pyqtgraph.Qt import QtCore
from pyqtgraph import QtWidgets
from typing import Callable, Optional
from multiprocessing.sharedctypes import Value
from sklearn.metrics import accuracy_score, f1_score
from pyqtgraph.Qt.QtGui import QFont, QLinearGradient, QBrush
from pyqtgraph.Qt.QtWidgets import (
    QPushButton,
    QApplication,
    QComboBox,
    QLabel,
    QCheckBox,
)

from .candlestick import CandlestickItem
from .multlines import MultiColorLines, MultiLines
from .datetimeaxis import DatetimeAxisX2, DatetimeAxisX3
from .simulations import ProcessMonteCarlo, ProcessHypSimulations, ThreadHypSimulations
from .utils import (
    apply_tax,
    process_df,
    process_mc_strategy,
    decimal_round,
    get_drawdowns,
    get_equitycurve,
    get_hitrate,
    get_riskmetrics,
    get_mc_price_paths,
)


BAR_TITLE = "LabQuant v0.1.2"


class LabQuant(QtWidgets.QMainWindow):
    """
    This class manages the entire graphical engine of the project based on the PyQtGraph module.
    """

    def __init__(self, df_1: pd.DataFrame, seed: int | None = None, show_roi: bool = False):
        """
        Initialization method.
        
        args:
            df_1 (dataframe): The main dataframe ohlcv processed by strategy.
            seed (int | None): Reproductibility of experiments.
            show_roi (bool | None): Enable or disable region of interest (see ROI - PyQtGraph).
        """
        self.app = QApplication(sys.argv)
        super().__init__()
        #
        self.seed = seed
        self.df_1 = df_1
        self.exec_loop = True
        self.show_roi = show_roi
        self.strategy = None
        self.str_params = None
        # Monte Carlo args
        self.mc_mode = None
        self.mc_paths_colors = None
        self.mc_line_plots = None
        self.mc_dist_bins = None
        self.mc_price_model = None
        self.mc_nsim = None
        self.mc_nsteps = None
        self.mc_sigma = None
        self.mc_s0 = None
        self.mc_r = None
        self.mc_dt = None
        self.mc_lambda_ = None
        self.mc_mu_y = None
        self.mc_sigma_y = None
        # Variables Logic
        self.showplt1 = 1   # Performance | Signals
        self.showplt2 = 1   # Positions & Signals
        self.showplt3 = 1   # Features
        self.showplt4 = 1   # Cumulative Amount
        self.showplt5 = 1   # Equity Curve Scatter
        self.showplt6 = 1   # Monte Carlo Simulation
        self.showplt7 = 1   # Distribution
        self.showplt8 = 1   # DDrown Monte Carlo
        self.showplt9 = 1   # Monte Carlo EquityCurves CDF
        self.showplt10 = 1  # Monte Carlo Plot Expander
        self.showplt11 = 1  # Strategy Returns
        self.showplt12 = 1  # Strategy params simulations
        self.showplt13 = 1  # Monte Carlo CDF | PDF
        self.cumulative_gain_curve_str = 0
        # Object Attributes
        self.equity_curve_true = None
        self.equity_curve_pred = None
        self.initial_pos = None
        self.pct_rate = None
        self.returns = None
        self.strategy_returns_true = None
        self.strategy_returns_pred = None
        self.cumulative_gain_curve_hold = None
        self.market_returns_cum = None
        self.strategy_returns_cum = None
        self.drawdown = None
        self.scatter_long_true = None
        self.scatter_short_true = None
        self.scatter_long_pred = None
        self.scatter_short_pred = None
        self.scatter_exit_true_long = None
        self.scatter_exit_true_short = None
        self.scatter_exit_pred_long = None
        self.scatter_exit_pred_short = None
        self.scatter_exit_gain_true = None
        self.scatter_exit_stop_true = None
        self.scatter_exit_gain_pred = None
        self.scatter_exit_stop_pred = None
        self.scatter_short_pred_plus = None
        self.scatter_short_pred_minus = None
        self.scatter_long_pred_plus = None
        self.scatter_long_pred_minus = None
        self.n_trads = None
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.calmar_ratio = None
        self.grad = None
        self.brush = None
        self.maker_fee = None
        self.risk_free = None
        self.period = None
        self.pos_true = None
        self.pos_pred = None
        self.frame_plot_2 = None
        self.plt_4 = None
        self.plt_5 = None
        self.plt_6 = None
        self.mc_sucess_prob = None
        self.var_value = None
        self.np_mem_1 = None
        self.np_mem_2 = None
        self.workProcess = None
        self.timer_plot = None
        self.mc_average_dd = None
        self.x_line_plt1 = None
        self.y_line_plt1 = None
        self.x_line_plt2 = None
        self.y_line_plt2 = None
        self.x_line_plt3 = None
        self.y_line_plt3 = None
        self.proxy = None
        self.test_ = False
        self.treat_zerodiv_factor = 1e-32
        self.df_diff_factor = 0
        self.df_main = pd.DataFrame()
        self.df_str_params = pd.DataFrame()

        # Reset df_1 index
        self.df_1 = self.df_1.reset_index(drop=True)

        # Set Axis shared var
        self.value_var_time_axis = Value("d")

        # Parse timestamp
        if "time" in self.df_1.columns:
            # Convert date object to to epoch timestamp
            if pd.api.types.is_object_dtype(self.df_1.time):
                # Set time to epoch timestamp
                try:
                    self.df_1["time"] = self.df_1.time.astype(float)
                    # self.df_1.time = pd.to_datetime(self.df_1.time)
                # Convert string unix epoch to float
                except Exception as e:
                    self.df_1.time = pd.to_datetime(self.df_1.time)
                    self.df_1["time"] = (
                        self.df_1["time"] - datetime.datetime(1970, 1, 1)
                    ).dt.total_seconds() * 1000 + 10800000.0
            # Convert date datetime64[ns] to epoch timestamp
            elif pd.api.types.is_datetime64_any_dtype(self.df_1.time):
                self.df_1["time"] = (
                    self.df_1["time"] - datetime.datetime(1970, 1, 1)
                ).dt.total_seconds() * 1000 + 10800000.0

            # Epoch [ms] normalization
            if self.df_1.time.values[-1] < 10000000000:
                self.df_1.time = self.df_1.time * 1000

            # Dateindex column reference - Epoch timestamp or timestamp[ms]
            self.df_1["dateindex"] = 0
            step_ref = self.df_1.time.values[1] - self.df_1.time.values[0]
            time_diff = self.df_1["time"].diff().fillna(step_ref)
            self.df_1["dateindex"] = (time_diff / step_ref).cumsum()
            self.value_var_time_axis.value = 0

        else:
            self.df_1["dateindex"] = self.df_1.index.values
            self.df_1["time"] = self.df_1.index.values
            self.value_var_time_axis.value = 10

        # MainWindow 1
        self.win_1 = QtWidgets.QMainWindow()
        self.win_1.setWindowTitle(BAR_TITLE)
        self.win_1.setGeometry(150, 100, 1400, 900)
        self.win_1.setStyleSheet("background-color: #282828;")

        # Set widgets layouts
        self.plot_widget_0 = pg.GraphicsLayoutWidget()
        self.plot_widget_0_1 = pg.GraphicsLayoutWidget()
        self.plot_widget_1 = pg.GraphicsLayoutWidget()
        self.plot_widget_2 = pg.GraphicsLayoutWidget()
        self.plot_widget_3 = pg.GraphicsLayoutWidget()

        # Fix widget_0 height
        self.plot_widget_0.setFixedHeight(40)

        # Set widgets bg-color
        self.plot_widget_0.setBackground(background="#282828")
        self.plot_widget_0_1.setBackground(background="#282828")
        self.plot_widget_1.setBackground(background="#282828")
        self.plot_widget_2.setBackground(background="#282828")
        self.plot_widget_3.setBackground(background="#282828")

        # Set QSplitter
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Add widgets in splitter
        self.splitter.addWidget(self.plot_widget_0)
        self.splitter.addWidget(self.plot_widget_0_1)
        self.splitter.addWidget(self.plot_widget_1)
        self.splitter.addWidget(self.plot_widget_2)
        self.splitter.addWidget(self.plot_widget_3)

        # Splitter settings
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: black;
                height: 1px; 
            }
        """)

        # Set Axis shared var
        self.value_var_hist_axis = Value("d")
        self.value_var_hist_axis.value = 0

        self.datetimeaxis_plt2 = DatetimeAxisX2(
            orientation="bottom",
            data=self.df_1.time,
            value_var_hist_axis=self.value_var_hist_axis,
            value_var_time_axis=self.value_var_time_axis,
        )
        self.datetimeaxis_plt3 = DatetimeAxisX3(
            orientation="bottom",
            data=self.df_1.time,
            value_var_hist_axis=self.value_var_hist_axis,
            value_var_time_axis=self.value_var_time_axis,
        )
        self.plt_0 = self.plot_widget_0_1.addPlot()
        self.plt_1 = self.plot_widget_1.addPlot()
        self.plt_2 = self.plot_widget_2.addPlot(
            axisItems={"bottom": self.datetimeaxis_plt2}
        )
        self.plt_3 = self.plot_widget_3.addPlot(
            axisItems={"bottom": self.datetimeaxis_plt3}
        )

        # Optimizations
        factor = 5
        self.plt_1.setClipToView(True)
        self.plt_2.setClipToView(True)
        self.plt_3.setClipToView(True)

        self.plt_0.setDownsampling(auto=True, mode="subsample", ds=factor)
        self.plt_1.setDownsampling(auto=True, mode="peak", ds=factor)
        self.plt_2.setDownsampling(auto=True, mode="peak", ds=factor)
        self.plt_3.setDownsampling(auto=True, mode="peak", ds=factor)

        # ROI - Region Of Interest
        if self.show_roi:
            if len(self.df_1) > 100000:
                pct_region = 0.02
            else:
                pct_region = 0.1
            self.plot_widget_0_1.setFixedHeight(50)
            self.roi = pg.LinearRegionItem([0, (pct_region * len(self.df_1.c))])
            self.roi.setBounds([0, len(self.df_1.c) - 1])
            self.roi.sigRegionChanged.connect(self._update_plot_by_roi)
            self.roi.setHoverBrush(pg.mkBrush((pg.mkColor("#D3D3D340"))))
            self.roi.setBrush(pg.mkBrush((pg.mkColor("#D3D3D320"))))
            self.plt_0.setFixedHeight(30)
            self.plt_0.addItem(self.roi)
            self.plt_0.plot(self.df_1.c.values, pen=pg.mkPen("#B0C4DE", width=0.6))
            self.plt_0.enableAutoRange(x=True, y=True)
            self.plt_0.setMouseEnabled(x=False, y=False)
            self.roi_plot_var = 0
            for line in self.roi.lines:
                line.setPen(pg.mkPen(pg.mkColor("grey"), width=1))
                line.setHoverPen(pg.mkPen(pg.mkColor("green"), width=3))

        else:
            self.plot_widget_0_1.hide()

        # Set splitter top margin
        self.splitter.setContentsMargins(0, 0, 0, 0)

        # Centralize splitter in win_1
        self.win_1.setCentralWidget(self.splitter)

        # ">" - Performance
        self.button_1 = QPushButton(self.plot_widget_0)
        self.button_1.clicked.connect(self._show_plot)
        self.button_1.setGeometry(10, 10, 20, 20)
        self.button_1.setText(">")
        self.button_1.show()
        self.button_1.setStyleSheet(
            "font: bold 11pt; color: white; background-color: #483D8B; "
            "border-radius: 1px; border: 1px outset grey;"
        )

        # "-" - Signals / Positions / Pct-change
        self.button_2 = QPushButton(self.plot_widget_0)
        self.button_2.clicked.connect(self._show_signals_positions)
        self.button_2.setGeometry(40, 10, 20, 20)
        self.button_2.setText("-")
        self.button_2.setStyleSheet(
            "font: bold 11pt; color: white; background-color: #483D8B; "
            "border-radius: 1px; border: 1px outset grey;"
        )

        # "X" - Features
        self.button_3 = QPushButton(self.plot_widget_0)
        self.button_3.clicked.connect(self._show_features)
        self.button_3.setGeometry(70, 10, 20, 20)
        self.button_3.setText("X")
        self.button_3.setStyleSheet(
            "font: bold 11pt; color: white; background-color: #483D8B; "
            "border-radius: 1px; border: 1px outset grey;"
        )

        # "A" - Cumulative Amount
        self.button_4 = QPushButton(self.plot_widget_0)
        self.button_4.clicked.connect(self._show_cumulative_gains)
        self.button_4.setGeometry(100, 10, 20, 20)
        self.button_4.setText("A")
        self.button_4.setStyleSheet(
            "font: bold 11pt; color: white; background-color: #483D8B; "
            "border-radius: 1px; border: 1px outset grey;"
        )

        # "MC" - Monte Carlo Simulation
        self.button_5 = QPushButton(self.plot_widget_0)
        self.button_5.clicked.connect(self._show_monte_carlo_simulation)
        self.button_5.setGeometry(200, 10, 30, 20)
        self.button_5.setText("MC")
        self.button_5.setEnabled(False)
        self.button_5.setStyleSheet(
            """
            QPushButton {
                font: bold 11pt; 
                color: white; 
                background-color: #483D8B; 
                border-radius: 1px; 
                border: 1px outset grey;
            }
            QPushButton:disabled {
                color: #8B8B8B;  
                background-color: #404040;  
                border: 1px outset #404040; 
            }
            """
        )

        # "D" - Price Distribution
        self.button_6 = QPushButton(self.plot_widget_0)
        self.button_6.clicked.connect(self._show_pricedistribution)
        self.button_6.setGeometry(100, 10, 20, 20)
        self.button_6.setText("D")
        self.button_6.setStyleSheet(
            "font: bold 11pt; color: white; background-color: #483D8B; "
            "border-radius: 1px; border: 1px outset grey;"
        )

        # "R" - Returns
        self.button_9 = QPushButton(self.plot_widget_0)
        self.button_9.clicked.connect(self._show_returns)
        self.button_9.setGeometry(40, 10, 20, 20)
        self.button_9.setText("R")
        self.button_9.setStyleSheet(
            "font: bold 11pt; color: white; background-color: #483D8B; "
            "border-radius: 1px; border: 1px outset grey;"
        )

        # "S" - Simulation
        self.button_10 = QPushButton(self.plot_widget_0)
        self.button_10.clicked.connect(self._show_hypparams_simulation)
        self.button_10.setGeometry(160, 10, 30, 20)
        self.button_10.setText("SM")
        self.button_10.setEnabled(False)
        self.button_10.setStyleSheet(
            """
            QPushButton {
                font: bold 11pt; 
                color: white; 
                background-color: #483D8B; 
                border-radius: 1px; 
                border: 1px outset grey;
            }
            QPushButton:disabled {
                color: #8B8B8B;  
                background-color: #404040;  
                border: 1px outset #404040; 
            }
            """
        )

        # Set Font
        self.font = QFont("TypeWriter")
        self.font.setPixelSize(13)

        # MC combobox
        self.combobox_mc = QComboBox(self.plot_widget_0)
        self.combobox_mc.setGeometry(245, 9, 305, 22)
        self.combobox_mc.setFont(self.font)
        self.combobox_mc.currentIndexChanged.connect(self._update_mc_mode)
        self.combobox_mc.addItems(
            [
                "RANDOM PRICES PRICE BASE",
                "RANDOM PRICES BLACK SCHOLES",
                "RANDOM PRICES MERTON JUMP DIFFUSION",
                "RANDOM RETURNS",
                "RANDOM RETURNS WITH REPLACEMENT",
                "RANDOM POSITIONS",
                "RANDOM STARTINGS POSITIONS",
                "RANDOM ENDINGS POSITIONS",
            ]
        )
        self.combobox_mc.setStyleSheet("""
            QComboBox {
                background-color: #2E2E2E; /* Cor de fundo do ComboBox */
                color: #B0C4DE; /* Cor do texto */
                border: 1px solid gray; /* Cor da borda */
                selection-color: black;
                selection-background-color: #B0C4DE;
            }
            QComboBox QAbstractItemView {
                background-color: #2E2E2E; /* Cor de fundo das opções */
                color: grey; /* Cor do texto das opções */
            }
            QComboBox:disabled {
                background-color: #424242; /* Fundo intermediário */
                color: #9A9A9A; /* Texto cinza médio */
                border: 1px solid #606060; /* Borda cinza escuro */
            }
        """)
        self.combobox_mc.setEnabled(False)

        # Checkbox
        self.checkbox_scatter = QCheckBox("SCATTERS", self.plot_widget_0)
        self.checkbox_scatter.stateChanged.connect(self._showhide_scatters)
        self.checkbox_scatter.setGeometry(565, 9, 100, 22)
        self.checkbox_scatter.setFont(self.font)
        self.checkbox_scatter.setChecked(True)
        self.checkbox_scatter.setStyleSheet(
            "color: #B0C4DE; background-color: #2E2E2E; "
            "border-radius: 1px; border: 1px solid grey"
        )

        # Scaling limit
        self.plt_1.getViewBox().setLimits(
            xMin=-10000000000000,
            xMax=10000000000000,
            yMin=-10000000000000,
            yMax=10000000000000,
        )
        self.plt_2.getViewBox().setLimits(
            xMin=-1000000000000,
            xMax=10000000000000,
            yMin=-10000000000000,
            yMax=10000000000000,
        )
        self.plt_3.getViewBox().setLimits(
            xMin=-10000000000000,
            xMax=10000000000000,
            yMin=-10000000000000,
            yMax=10000000000000,
        )

        # Risk Metrics
        self.risk_metrics_textitem = pg.TextItem(color="#5EF38C")
        self.risk_metrics_textitem.setParentItem(self.plt_3)
        self.risk_metrics_textitem.setPos(10, 5)
        self.risk_metrics_textitem.setFont(self.font)

        # Profit and Losses
        self.pnl_textitem = pg.TextItem(color="#5EF38C")
        self.pnl_textitem.setParentItem(self.plt_2)
        self.pnl_textitem.setPos(10, 5)
        self.pnl_textitem.setFont(self.font)

        # Returns
        self.returns_textitem = pg.TextItem(color="#5EF38C")
        self.returns_textitem.setParentItem(self.plt_2)
        self.returns_textitem.setPos(10, 5)
        self.returns_textitem.setFont(self.font)

        # Hit-Rate and n-Hits | n-Losses
        self.hit_trads_textitem = pg.TextItem(color="#90EE90")
        self.hit_trads_textitem.setParentItem(self.plt_2)
        self.hit_trads_textitem.setPos(10, 5)
        self.hit_trads_textitem.setFont(self.font)

        # Distribution
        self.dist_textitem = pg.TextItem(color="#99CCFF")
        self.dist_textitem.setParentItem(self.plt_2)
        self.dist_textitem.setPos(10, 5)
        self.dist_textitem.setFont(self.font)

        # Monte Carlo Status
        self.mc_label = QLabel(self.plot_widget_0)
        self.mc_label.setStyleSheet("color: #B0C4DE;")
        self.mc_label.setGeometry(685, 10, 450, 20)
        self.mc_label.setFont(self.font)
        self.mc_label.hide()

        # Simulation Status
        self.sim_label = QLabel(self.plot_widget_0)
        self.sim_label.setStyleSheet("color: #B0C4DE;")
        self.sim_label.setGeometry(685, 10, 450, 20)
        self.sim_label.setFont(self.font)
        self.sim_label.hide()

        # Initializing status
        self.init_label = QLabel(self.plot_widget_0)
        self.init_label.setStyleSheet("color: yellow;")
        self.init_label.setGeometry(685, 10, 400, 20)
        self.init_label.setFont(self.font)
        self.init_label.show()
        self.init_label.setText("STARTING...")

        # Hit-Rate and n-Hits | n-Losses
        self.scores_textitem = pg.TextItem(color="#90EE90")
        self.scores_textitem.setParentItem(self.plt_1)
        self.scores_textitem.setPos(10, 5)
        self.scores_textitem.setFont(self.font)
        self.scores_textitem.hide()

        # Drawndown
        self.drawdown_textitem = pg.TextItem(color="#ff3562")
        self.drawdown_textitem.setParentItem(self.plt_1)
        self.drawdown_textitem.setPos(10, 5)
        self.drawdown_textitem.setFont(self.font)

        # Plot Configs
        self.font_axis = QFont("Arial")
        self.font_axis.setPixelSize(13)
        self.font_axis.setWeight(40)
        # Font Tick
        self.plt_0.getAxis("right").setTickFont(self.font_axis)
        self.plt_1.getAxis("right").setTickFont(self.font_axis)
        self.plt_2.getAxis("right").setTickFont(self.font_axis)
        self.plt_3.getAxis("right").setTickFont(self.font_axis)
        self.plt_3.getAxis("bottom").setTickFont(self.font_axis)
        # Color Tick
        self.plt_0.getAxis("right").setTextPen("#C0C0C0")
        self.plt_1.getAxis("right").setTextPen("#C0C0C0")
        self.plt_2.getAxis("right").setTextPen("#C0C0C0")
        self.plt_3.getAxis("right").setTextPen("#C0C0C0")
        self.plt_3.getAxis("bottom").setTextPen("#C0C0C0")
        # Config Grid
        self.plt_0.showGrid(x=True, y=True, alpha=0.2)
        self.plt_1.showGrid(x=True, y=True, alpha=0.2)
        self.plt_2.showGrid(x=True, y=True, alpha=0.2)
        self.plt_3.showGrid(x=True, y=True, alpha=0.2)
        # Config Axis
        self.plt_0.showAxis("right")
        self.plt_0.showAxis("left")
        self.plt_0.showAxis("top")
        self.plt_0.getAxis("left").setStyle(showValues=False)
        self.plt_0.getAxis("top").setStyle(showValues=False)
        self.plt_0.getAxis("bottom").setStyle(showValues=False)
        self.plt_0.getAxis("right").setStyle(showValues=False)
        self.plt_0.getAxis("right").setWidth(int(65))
        #
        self.plt_1.showAxis("right")
        self.plt_1.showAxis("left")
        self.plt_1.showAxis("top")
        self.plt_1.getAxis("left").setStyle(showValues=False)
        self.plt_1.getAxis("top").setStyle(showValues=False)
        self.plt_1.getAxis("bottom").setStyle(showValues=False)
        self.plt_1.getAxis("right").setWidth(int(65))
        #
        self.plt_2.showAxis("right")
        self.plt_2.showAxis("left")
        self.plt_2.showAxis("top")
        self.plt_2.getAxis("left").setStyle(showValues=False)
        self.plt_2.getAxis("top").setStyle(showValues=False)
        self.plt_2.getAxis("bottom").setStyle(showValues=False)
        self.plt_2.getAxis("right").setWidth(int(65))
        #
        self.plt_3.showAxis("right")
        self.plt_3.showAxis("left")
        self.plt_3.showAxis("top")
        self.plt_3.getAxis("left").setStyle(showValues=False)
        self.plt_3.getAxis("top").setStyle(showValues=False)
        self.plt_3.getAxis("right").setWidth(int(65))
        # Config Frame
        self.plt_0.getAxis("bottom").setPen(pg.mkPen(color="#505050", width=1))
        self.plt_0.getAxis("right").setPen(pg.mkPen(color="#505050", width=1))
        self.plt_0.getAxis("top").setPen(pg.mkPen(color="#505050", width=1))
        self.plt_0.getAxis("left").setPen(pg.mkPen(color="#505050", width=1))
        #
        self.plt_1.getAxis("bottom").setPen(pg.mkPen(color="#606060", width=1))
        self.plt_1.getAxis("right").setPen(pg.mkPen(color="#606060", width=1))
        self.plt_1.getAxis("top").setPen(pg.mkPen(color="#606060", width=1))
        self.plt_1.getAxis("left").setPen(pg.mkPen(color="#606060", width=1))
        #
        self.plt_2.getAxis("bottom").setPen(pg.mkPen(color="#606060", width=1))
        self.plt_2.getAxis("right").setPen(pg.mkPen(color="#606060", width=1))
        self.plt_2.getAxis("top").setPen(pg.mkPen(color="#606060", width=1))
        self.plt_2.getAxis("left").setPen(pg.mkPen(color="#606060", width=1))
        #
        self.plt_3.getAxis("bottom").setPen(pg.mkPen(color="#606060", width=1))
        self.plt_3.getAxis("right").setPen(pg.mkPen(color="#606060", width=1))
        self.plt_3.getAxis("top").setPen(pg.mkPen(color="#606060", width=1))
        self.plt_3.getAxis("left").setPen(pg.mkPen(color="#606060", width=1))
        # Set Link - Axis
        self.plt_1.setXLink(self.plt_3)
        self.plt_2.setXLink(self.plt_3)

        # Update mc_mode
        self._update_mc_mode()

        # MC random
        np.random.seed(self.seed)

        print(f"\n{Fore.LIGHTYELLOW_EX}LOADING DATA...{Fore.RESET}")

    def start(
        self,
        stop_rate: float | int | None = None,
        gain_rate: float | int | None = None,
        opers_fee: float | int | None = None,
        metrics_riskfree: float | int = 10,
        metrics_period: float | int = 365,
        dist_bins: int = 50,
        show_candles: bool = False,
        strategy: Optional[Callable] = None,
        str_params: list | None = None,
        mc_paths_colors: bool = True,
        mc_line_plots: bool = False,
        mc_dist_bins: int = 50,
        mc_nsim: int = 200,
        mc_nsteps: int | None = None,
        mc_sigma: float | int = 0.5,
        mc_s0: float | int | None = None,
        mc_r: float | int = 0.5,
        mc_dt: float = (1 / 365),
        mc_lambda_: float | int = 0.1,
        mc_mu_y: float | int = 0.02,
        mc_sigma_y: float | int = 0.1,
        mc_rndnpositions: int = 10,
        sim_taskmode: str = "process",
        sim_method: str = "grid",
        sim_params: dict = None,
        sim_nbest: int = 10,
        sim_nrandsims: int = 15,
        sim_bayesopt_ncalls: int = 5,
        sim_bayesopt_spaces: list | None = None,
        sim_bayesopt_kwargs: dict = {},):
        """
        This method initializes the project's graphical engine by configuring resources according to user-defined arguments.

        Args:
            stop_rate (int): Stop loss threshold (%).
            gain_rate (int): Take profit threshold (%).
            opers_fee (int): Emulation of operation fee (%).
            metrics_riskfree (int): Risk-free parameter (%) used in Sharpe-Ratio and Sortino-Ratio calculations. Default is 10.
            metrics_period (int): Period parameter for Sharpe-Ratio, Sortino-Ratio, and Calmar-Ratio calculations. Default is 365.
            dist_bins (int): Number of bins for the price distribution plot. Default is 50.
            show_candles (bool): Plot candlesticks. Disable for better performance with large datasets. Default is False.
            strategy (function): Strategy function. Required for Monte Carlo tests and hyperparameter searches.
            str_params (list): Strategy parameters. Necessary for Monte Carlo tests.
            mc_paths_colors (bool): Enable Monte Carlo test path line coloring. Default is True.
            mc_line_plots (bool): Enable Monte Carlo test path line plots. Disable for better performance at scale. Default is False.
            mc_dist_bins (int): Number of bins for Monte Carlo test distribution plots. Default is 50.
            mc_nsim (int): Number of Monte Carlo test simulations. Default is 200.
            mc_nsteps (int): Number of Monte Carlo test steps, based on the length of the time frame. Default is None.
            mc_sigma (float | int): Random price volatility (σ) for Black-Scholes and Merton models. Default is 0.5 (%).
            mc_s0 (float | int): Initial stock price for Monte Carlo test (Black-Scholes and Merton models). Default is None.
            mc_r (float | int): Risk-free rate for Monte Carlo test (Black-Scholes and Merton models). Default is 0.5.
            mc_dt (float): Time step for Monte Carlo test (Black-Scholes and Merton models). Default is (1 / 365).
            mc_lambda_ (float | int): Jump intensity (λ) for Monte Carlo test (Merton model). Default is 0.1.
            mc_mu_y (float | int): Mean of jump sizes (μ_y) for Monte Carlo test (Merton model). Default is 0.02.
            mc_sigma_y (float | int): Standard deviation of jump sizes (σ_y) for Monte Carlo test (Merton model). Default is 0.1.
            mc_rndnpositions (int): Window size to randomize starting or ending positions in Monte Carlo test. Default is 10.
            sim_taskmode (str): Simulation task mode ("process" or "thread"). Default is "process".
            sim_method (str): Hyperparameter simulation method ("grid", "random", or "bayesian-opt"). Default is "grid".
            sim_params (dict): Hyperparameter simulation strategy parameters for "grid" or "random" methods. Default is None.
            sim_nbest (int): Number of best curves to show in hyperparameter search simulations. Default is 10.
            sim_nrandsims (int): Number of random simulations for hyperparameter search simulations. Default is 15.
            sim_bayesopt_ncalls (int): Number of calls for Bayesian optimization (scikit-optimize). Default is 5.
            sim_bayesopt_spaces (list): Search spaces for Bayesian optimization (scikit-optimize). Default is None.
            sim_bayesopt_kwargs (dict): Additional kwargs for Bayesian optimization (scikit-optimize). Default is {}.

        """

        # Set positions | pred & true
        if "pred" not in self.df_1.columns:
            self.df_1["pred"] = self.df_1.positions
        else:
            self.df_1["positions"] = self.df_1.pred

        self.pos_pred = self.df_1.pred.values
        self.stop_rate = stop_rate
        self.gain_rate = gain_rate
        self.opers_fee = opers_fee
        self.risk_free = metrics_riskfree
        self.period = metrics_period
        self.dist_bins = dist_bins
        self.show_candles = show_candles
        self.strategy = strategy
        self.str_params = str_params
        self.mc_paths_colors = mc_paths_colors
        self.mc_line_plots = mc_line_plots
        self.mc_dist_bins = mc_dist_bins
        self.mc_nsim = mc_nsim
        self.mc_nsteps = mc_nsteps
        self.mc_sigma = mc_sigma
        self.mc_s0 = mc_s0
        self.mc_r = mc_r
        self.mc_dt = mc_dt
        self.mc_lambda_ = mc_lambda_
        self.mc_mu_y = mc_mu_y
        self.mc_rndnpositions = mc_rndnpositions
        self.mc_sigma_y = mc_sigma_y
        self.sim_taskmode = sim_taskmode
        self.sim_method = sim_method
        self.sim_params = sim_params
        self.sim_nbest = sim_nbest
        self.sim_nrandsims = sim_nrandsims
        self.sim_bayesopt_spaces = sim_bayesopt_spaces
        self.sim_bayesopt_kwargs = sim_bayesopt_kwargs
        self.sim_bayesopt_ncalls = sim_bayesopt_ncalls
        

        # Set first position
        try:
            if self.df_1.pred.values[0] == 0:
                self.initial_pos = abs(self.pos_pred[self.pos_pred != 0][0])
            else:
                self.initial_pos = abs(self.pos_pred[0])
        except Exception as e:
            exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
            exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
            track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
            print(
                f"{Fore.LIGHTYELLOW_EX}{exception_type}{exception_message}{track_line}"
            )
            print(f"SET INITIAL POSITON TO 1{Fore.RESET}")
            # raise sys.exc_info()[0]
            self.initial_pos = 1

        # Set y-true values
        if "true" in self.df_1.columns:
            self.pos_true = self.df_1.true.values
        else:
            self.pos_true = np.nan

        # Process mouse events
        self._mouse_events()

        # Process df_1
        self.df_1 = process_df(
            self.df_1,
            self.pos_true,
            self.pos_pred,
            self.stop_rate,
            self.gain_rate,
            self.initial_pos,
        )

        # Set crosshair
        self._set_crosshair()

        print(f"\n{Fore.LIGHTYELLOW_EX}INITIALIZING LabQuant...{Fore.RESET}")

        # Plot logic
        if self.test_:
            if self.show_roi:
                self.df_main = self.df_1.copy()
                self.roi_plot_var = 1
                self._update_plot_by_roi()
            else:
                self._show_plot()
        else:
            self.win_1.show()
            if self.show_roi:
                self.df_main = self.df_1.copy()
                if self.str_params is not None:
                    self.df_str_params = self.str_params[0].copy()
                self.roi_plot_var = 1
                QtCore.QTimer.singleShot(100, self._update_plot_by_roi)
            else:
                self.df_main = self.df_1.copy()
                if self.str_params is not None:
                    self.df_str_params = self.str_params[0].copy()
                QtCore.QTimer.singleShot(100, self._show_plot)

        # QApplication Main loop
        if (
            self.exec_loop
            and (sys.flags.interactive != 1)
            or not hasattr(QtCore, "PYQT_VERSION")
        ):
            QApplication.instance().exec_()

    def _show_plot(self):
        """
        Displays the signals interface and switches between the signals and the performance interface.
        """

        self.plt_1.clear()
        self.plt_2.clear()
        self.plt_3.clear()

        self.button_1.show()
        self.risk_metrics_textitem.hide()
        self.pnl_textitem.hide()
        self.drawdown_textitem.hide()
        self.dist_textitem.hide()
        self.scores_textitem.hide()

        self.plt_1.showButtons()
        self.plt_1.showAxis("top")
        self.plt_1.showAxis("bottom")
        self.plt_2.getAxis("bottom").setStyle(showValues=False)

        # Enable | Disable simulations
        self._simulations_logic()

        # Returns pct
        self.returns = self.df_1.c.pct_change()

        # Strategy returns
        self.strategy_returns_pred = (
            self.returns * self.df_1.positions_pred.shift(1)
        ).fillna(0)
        self.strategy_returns_true = (
            self.returns * self.df_1.positions_true.shift(1)
        ).fillna(0)

        # Apply opers fees
        if self.opers_fee is not None:
            strategy_returns_pred_ = apply_tax(
                self.opers_fee,
                self.strategy_returns_pred.values,
                self.df_1.positions.values,
            )
            self.strategy_returns_pred = pd.Series(strategy_returns_pred_)

        # Process Hit-Rate
        hitrate = get_hitrate(self.df_1.signals_pred, self.strategy_returns_pred)
        text = (
            "HIT-RATE: {}%     ".format(decimal_round(hitrate[2], 1))
            + "n-HITS: {}     ".format(str(hitrate[0]))
            + "n-LOSSES: {}    ".format(str(hitrate[1]))
            + "n-TRADS: {} ".format(str(hitrate[3]))
        )
        self.hit_trads_textitem.setText(text=text)
        self.hit_trads_textitem.show()
        self.n_trads = hitrate[3]

        # Show Signals
        if self.showplt1 == 1:
            self.roi_plot_var = 1
            self.value_var_hist_axis.value = 0
            self.plt_1.show()
            self.plt_2.show()
            self.plt_3.show()
            self._process_scatter()
            self.button_2.show()
            self.button_4.hide()
            self.button_6.show()
            self.button_9.hide()
            self.returns_textitem.hide()
            self.plt_2.setXLink(self.plt_3)

            # Scatter plot y_true
            if np.all(np.isnan(self.pos_true)):
                self.plt_2.setTitle("Trade-Analysis")
                self.plt_1.hide()
                self.plt_1.hideAxis("top")
                self.plt_1.hideAxis("bottom")
                self.plt_1.hideButtons()
                self.plot_widget_1.hide()
                self.plot_widget_2.show()
                self.plot_widget_3.show()
                self.plt_1.setTitle("")
                self.plt_1.setContentsMargins(0, 0, 0, 0)
                self.plt_2.setContentsMargins(0, 0, 0, 0)

                self.splitter.setSizes([50, 40, 0, 400, 150])

                if self.show_candles:
                    candlestick_data = list(
                        zip(
                            self.df_1.dateindex.values,
                            self.df_1.o.values,
                            self.df_1.h.values,
                            self.df_1.l.values,
                            self.df_1.c.values,
                        )
                    )
                    item = CandlestickItem(candlestick_data)
                    self.plt_2.addItem(item)
                else:
                    self.plt_2.plot(self.df_1.dateindex.values, self.df_1.c.values)

            else:
                self.plt_1.show()
                self.plt_1.setTitle("y-true signals")
                self.plt_2.setTitle("y-pred signals")
                self.plt_1.plot(self.df_1.dateindex.values, self.df_1.c.values)
                self.plt_2.plot(self.df_1.dateindex.values, self.df_1.c.values)
                self.plot_widget_1.show()
                self.plot_widget_2.show()
                self.plot_widget_3.show()
                self.plt_1.setContentsMargins(0, 0, 0, 0)
                self.plt_2.setContentsMargins(0, 0, 0, 0)

                self.splitter.setSizes([50, 40, 200, 200, 150])
                text = "SCORE: {}     ".format(
                    decimal_round(
                        accuracy_score(self.df_1["true"], self.df_1["pred"]), 2
                    )
                ) + "F1-SCORE: {}     ".format(
                    decimal_round(
                        f1_score(
                            self.df_1["true"], self.df_1["pred"], average="weighted"
                        ),
                        2,
                    )
                )
                self.scores_textitem.setText(text=text)
                self.scores_textitem.show()

            if not np.all(np.isnan(self.pos_true)):
                self.plt_1.addItem(self.scatter_long_true)
                self.plt_1.addItem(self.scatter_short_true)

                self.plt_1.addItem(self.scatter_short_true_plus)
                self.plt_1.addItem(self.scatter_short_true_minus)
                self.plt_1.addItem(self.scatter_long_true_plus)
                self.plt_1.addItem(self.scatter_long_true_minus)

                self.plt_1.addItem(self.scatter_exit_true_long)
                self.plt_1.addItem(self.scatter_exit_true_short)
                self.plt_1.addItem(self.scatter_exit_gain_true)
                self.plt_1.addItem(self.scatter_exit_stop_true)

            self.plt_2.addItem(self.scatter_long_pred)
            self.plt_2.addItem(self.scatter_short_pred)

            self.plt_2.addItem(self.scatter_short_pred_plus)
            self.plt_2.addItem(self.scatter_short_pred_minus)
            self.plt_2.addItem(self.scatter_long_pred_plus)
            self.plt_2.addItem(self.scatter_long_pred_minus)

            self.plt_2.addItem(self.scatter_exit_pred_long)
            self.plt_2.addItem(self.scatter_exit_pred_short)
            self.plt_2.addItem(self.scatter_exit_gain_pred)
            self.plt_2.addItem(self.scatter_exit_stop_pred)

            # y_true x y_pred
            if np.all(np.isnan(self.pos_true)):
                self.plt_3.setTitle("Positions")
            else:
                self.plt_3.setTitle(
                    "y_true positions (green)   |   y_pred positions (red)"
                )
                self.plt_3.plot(
                    self.df_1.dateindex.values,
                    (self.df_1.positions_true * self.initial_pos).values,
                    pen={"color": (127, 200, 0), "width": 0.6},
                )  # GREEN
            self.plt_3.plot(
                self.df_1.dateindex.values,
                (self.df_1.positions_pred * self.initial_pos).values,
                pen={"color": (255, 20, 30)},
            )
            self.showplt1 = 2
            self.showplt2 = 2
            self.showplt11 = 1

            if self.checkbox_scatter.isChecked():
                self._showhide_scatters(True)
            else:
                self._showhide_scatters(False)

        # Show Performance
        else:
            self.showplt1 = 1
            self.roi_plot_var = 2
            self.value_var_hist_axis.value = 0

            self.plt_1.show()
            self.plt_2.show()
            self.plt_3.show()
            self.button_2.hide()
            self.button_4.show()
            self.button_6.hide()
            self.button_9.show()
            self.pnl_textitem.show()
            self.drawdown_textitem.show()
            self.returns_textitem.hide()
            self.hit_trads_textitem.hide()
            self.plot_widget_1.show()
            self.splitter.setSizes([50, 40, 200, 300, 200])

            # Cumulative Gains Reapplied
            self.df_1.cumul_gains_reappl_hold = (
                (1 + self.returns).cumprod() * self.initial_pos
            ).fillna(self.initial_pos)
            self.df_1.cumul_gains_reappl_str = (
                (1 + self.strategy_returns_pred).cumprod() * self.initial_pos
            ).fillna(self.initial_pos)

            # Cumulative Gains
            self.df_1.cumul_gains_hold = (self.returns.cumsum() + 1) * self.initial_pos
            self.df_1.cumul_gains_str = (
                self.strategy_returns_pred.cumsum() + 1
            ) * self.initial_pos

            # Returns cumulative (Strategy x Market)
            self.market_returns_cum = (self.returns.cumsum()).bfill()
            self.strategy_returns_cum = (self.strategy_returns_pred.cumsum()).bfill()

            text = (
                "PNL: "
                + str("%.2f" % (self.strategy_returns_cum.values[-1] * 100))
                + "%"
            )
            if self.strategy_returns_cum.values[-1] > 0:
                self.pnl_textitem.setText(text=text, color="#5EF38C")
            else:
                self.pnl_textitem.setText(text=text, color="#7D7ABC")
            self.pnl_textitem.show()

            self.plt_2.setTitle("Strategy (Green)  |  Hold (Grey)")
            self.plt_2.addLine(x=None, y=0, pen={"color": "red", "width": 0.3})
            self.plt_2.plot(
                self.df_1.dateindex.values,
                self.market_returns_cum.values,
                pen={"color": "#A9A9A9", "width": 0.4},
            )
            self.plt_2.plot(
                self.df_1.dateindex.values,
                self.strategy_returns_cum.values,
                pen={"color": "g", "width": 0.7},
            )

            # Equity Curve
            self.equity_curve_true = (self.strategy_returns_true.cumsum()) + 1
            self.equity_curve_pred = (self.strategy_returns_pred.cumsum()) + 1
            if np.all(np.isnan(self.pos_true)):
                self.plt_3.setTitle("Equity Curve")
            else:
                self.plt_3.setTitle("Equity Curve y-pred (Blue)  |  y-true (Green)")
                self.plt_3.plot(
                    self.df_1.dateindex.values,
                    self.equity_curve_true.values,
                    pen={"color": "g", "width": 0.3},
                )
            self.plt_3.plot(
                self.df_1.dateindex.values,
                self.equity_curve_pred.values,
                pen={"color": "#63B8FF", "width": 0.7},
            )
            # Scatter plot y_pred
            self.showplt5 = 2
            self.showplt4 = 1
            self._process_scatter()
            self.plt_3.addItem(self.scatter_long_pred)
            self.plt_3.addItem(self.scatter_short_pred)

            self.plt_3.addItem(self.scatter_short_pred_plus)
            self.plt_3.addItem(self.scatter_short_pred_minus)
            self.plt_3.addItem(self.scatter_long_pred_plus)
            self.plt_3.addItem(self.scatter_long_pred_minus)

            self.plt_3.addItem(self.scatter_exit_pred_long)
            self.plt_3.addItem(self.scatter_exit_pred_short)
            self.plt_3.addItem(self.scatter_exit_gain_pred)
            self.plt_3.addItem(self.scatter_exit_stop_pred)

            # Drawdowns
            self.drawdown = get_drawdowns(pd.Series(self.equity_curve_pred))
            self.grad = QLinearGradient(0, 0, 0, -100)
            self.grad.setColorAt(0.15, pg.mkColor("#27408B"))
            self.grad.setColorAt(0.4, pg.mkColor("#CD0000"))
            self.brush = QBrush(self.grad)
            self.plt_1.setTitle("Drawdown")
            self.plt_1.setYRange(0.0, -0.4)
            self.plt_1.plot(
                self.df_1.dateindex.values,
                self.drawdown[0] * 100,
                pen={"color": "#1E1E1E", "width": 1.0},
                fillBrush=self.brush,
                fillLevel=0,
            )
            self.plt_1.addLine(x=None, y=0, pen={"color": "#27408B", "width": 0.8})

            text = (
                "DD-MAX: "
                + str("%.2f" % (min(self.drawdown[0]) * -100))
                + "%"
                + "     DD-DURATION: "
                + str(int(max(self.drawdown[1])))
            )
            self.drawdown_textitem.setText(text=text, color="#ff3562")
            self.drawdown_textitem.show()

            # Risk Metrics
            rmetrics = get_riskmetrics(
                self.period,
                self.risk_free,
                self.strategy_returns_pred,
                self.drawdown[0],
            )
            self.sharpe_ratio = rmetrics[0]
            self.sortino_ratio = rmetrics[1]
            self.calmar_ratio = rmetrics[2]
            text = (
                "SHARPE: {}     ".format(decimal_round(self.sharpe_ratio, 2))
                + "SORTINO: {}     ".format(decimal_round(self.sortino_ratio, 2))
                + "CALMAR: {} ".format(decimal_round(self.calmar_ratio, 2))
            )
            self.risk_metrics_textitem.setText(text=text)
            self.risk_metrics_textitem.show()

            self.plt_1.enableAutoRange(x=True, y=True)
            self.plt_2.enableAutoRange(x=True, y=True)
            self.plt_3.enableAutoRange(x=True, y=True)

            if self.checkbox_scatter.isChecked():
                self._showhide_scatters(True)
            else:
                self._showhide_scatters(False)

        # Crosshair update
        self.plt_1.addItem(self.x_line_plt1, ignoreBounds=True)
        self.plt_1.addItem(self.y_line_plt1, ignoreBounds=True)
        self.plt_2.addItem(self.x_line_plt2, ignoreBounds=True)
        self.plt_2.addItem(self.y_line_plt2, ignoreBounds=True)
        self.plt_3.addItem(self.x_line_plt3, ignoreBounds=True)
        self.plt_3.addItem(self.y_line_plt3, ignoreBounds=True)

        self.showplt3 = 1
        self.showplt7 = 1

        self.plt_1.enableAutoRange(x=True, y=True)
        self.plt_2.enableAutoRange(x=True, y=True)
        self.plt_3.enableAutoRange(x=True, y=True)

        if self.test_:
            self.exec_loop = False

        self.init_label.hide()

    def _update_plot_by_roi(self):
        """
        Set the data range (df_1) defined by the ROI (region of interest) window and update plot
        """
        region = self.roi.getRegion()
        start, end = int(region[0]), int(region[1])

        if end - start < 1:
            end = start

        self.df_1 = self.df_main[start : end + 1]
        self.df_1 = self.df_1.reset_index(drop=True)

        if self.roi_plot_var == 1:
            self.showplt1 = 1
        elif self.roi_plot_var == 2:
            self.showplt1 = 0

        if self.showplt3 == 2:
            self.showplt3 = 1
            self._show_features()
        elif self.showplt7 == 2:
            self.showplt7 = 1
            self._show_pricedistribution()

        else:
            self._show_plot()

        self.plt_1.enableAutoRange(x=True, y=True)
        self.plt_2.enableAutoRange(x=True, y=True)
        self.plt_3.enableAutoRange(x=True, y=True)

    def _show_signals_positions(self):
        """
        Switches between "positions", "signals" and "signals_size" on the main interface.
        """
        self.risk_metrics_textitem.hide()
        self.pnl_textitem.hide()
        self.drawdown_textitem.hide()

        if self.showplt3 == 1:
            if self.showplt2 == 1:
                self.plt_3.clear()
                if np.all(np.isnan(self.pos_true)):
                    self.plt_3.setTitle("Positions")
                else:
                    self.plt_3.setTitle(
                        "y_true positions (green)   |   y_pred positions (red)"
                    )
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        (self.df_1.positions_true * self.initial_pos).values,
                        pen={"color": (127, 200, 0), "width": 0.6},
                    )
                self.plt_3.plot(
                    self.df_1.dateindex.values,
                    (self.df_1.positions_pred * self.initial_pos).values,
                    pen={"color": (255, 20, 30)},
                )
                self.showplt2 = 2

            elif self.showplt2 == 2:
                self.plt_3.clear()
                if np.all(np.isnan(self.pos_true)):
                    self.plt_3.setTitle("Signals")
                else:
                    self.plt_3.setTitle(
                        "y_true signals (green)   |   y_pred signals (red)"
                    )
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        self.df_1.signals_true.values,
                        pen={"color": (127, 200, 0), "width": 0.6},
                    )  # GREEN
                self.plt_3.plot(
                    self.df_1.dateindex.values,
                    self.df_1.signals_pred.values,
                    pen={"color": (255, 20, 30)},
                )
                self.showplt2 = 3

            elif self.showplt2 == 3:
                self.plt_3.clear()
                if np.all(np.isnan(self.pos_true)):
                    self.plt_3.setTitle("Signals-Size")
                else:
                    self.plt_3.setTitle(
                        "y_true signals-size (green)   |   y_pred signals-size (red)"
                    )
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        self.df_1.signals_size_true.values,
                        pen={"color": (127, 200, 0), "width": 0.6},
                    )  # GREEN
                self.plt_3.plot(
                    self.df_1.dateindex.values,
                    self.df_1.signals_size_pred.values,
                    pen={"color": (255, 20, 30)},
                )
                self.showplt2 = 4

            else:
                self.plt_3.clear()
                if np.all(np.isnan(self.pos_true)):
                    self.plt_3.setTitle("Positions")
                else:
                    self.plt_3.setTitle(
                        "y_true positions (green)   |   y_pred positions (red)"
                    )
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        (self.df_1.positions_true * self.initial_pos).values,
                        pen={"color": (127, 200, 0), "width": 0.6},
                    )
                self.plt_3.plot(
                    self.df_1.dateindex.values,
                    (self.df_1.positions_pred * self.initial_pos).values,
                    pen={"color": (255, 20, 30)},
                )
                self.showplt2 = 2

            self.plt_3.addItem(self.x_line_plt3, ignoreBounds=True)
            self.plt_3.addItem(self.y_line_plt3, ignoreBounds=True)

    def _show_features(self):
        """
        Process the features visualization defined by strategy.
        """
        self.value_var_hist_axis.value = 0
        self.hit_trads_textitem.show()
        self.risk_metrics_textitem.hide()
        self.pnl_textitem.hide()
        self.drawdown_textitem.hide()
        self.dist_textitem.hide()
        self.hit_trads_textitem.hide()
        self.scores_textitem.hide()
        self.returns_textitem.hide()
        self.checkbox_scatter.show()
        self.button_2.hide()
        self.button_4.hide()
        self.button_6.show()
        self.button_9.hide()

        # MC Simulations | params simulations
        if self.showplt12 == 2:
            self.checkbox_scatter.hide()
        if self.showplt6 == 2:
            self.checkbox_scatter.hide()

        self.plt_1.setTitle("PLT1")
        self.plt_2.setTitle("PLT2")
        self.plt_3.setTitle("PLT3")

        self.showplt1 = 1
        self.showplt7 = 1

        self.plt_1.clear()
        self.plt_2.clear()
        self.plt_3.clear()

        self.plt_1.addItem(self.x_line_plt1, ignoreBounds=True)
        self.plt_1.addItem(self.y_line_plt1, ignoreBounds=True)
        self.plt_2.addItem(self.x_line_plt2, ignoreBounds=True)
        self.plt_2.addItem(self.y_line_plt2, ignoreBounds=True)
        self.plt_3.addItem(self.x_line_plt3, ignoreBounds=True)
        self.plt_3.addItem(self.y_line_plt3, ignoreBounds=True)

        self.plt_1.showAxis("top")
        self.plt_1.showAxis("bottom")
        self.plt_1.showButtons()

        self.plt_2.getAxis("bottom").setStyle(showValues=False)
        self.plt_2.setXLink(self.plt_3)

        self.plt_1.showAxis("bottom")

        self.plt_1.enableAutoRange(x=True, y=True)
        self.plt_2.enableAutoRange(x=True, y=True)
        self.plt_3.enableAutoRange(x=True, y=True)

        # Plot Features
        if self.showplt3 == 1:
            self.showplt3 = 2

            # Set PL2 feature space
            if (
                self.df_1.filter(regex="PLT1.*", axis=1).columns.empty
                and self.df_1.filter(regex="PLT3.*", axis=1).columns.empty
            ):
                self.plot_widget_1.hide()
                self.plot_widget_3.hide()

                self.plt_2.getAxis("bottom").setStyle(showValues=True)
                self.plt_2.getAxis("bottom").setPen(pg.mkPen(color="#606060", width=1))
                self.plt_2.getAxis("bottom").setTickFont(self.font_axis)
                self.plt_2.getAxis("bottom").setTextPen("#C0C0C0")
                self.plt_2.getAxis("right").setWidth(int(65))

            # Set PLT1, PLT2 and PLT3 feature spaces
            else:
                if (
                    not self.df_1.filter(regex="PLT1.*", axis=1).columns.empty
                    and not self.df_1.filter(regex="PLT3.*", axis=1).columns.empty
                ):
                    self.plot_widget_1.show()
                    self.plot_widget_3.show()
                    self.plt_1.show()
                    self.plt_3.show()
                    self.splitter.setSizes([50, 40, 150, 350, 170])

                # Set PLT3 and PLT2 feature spaces
                elif (
                    self.df_1.filter(regex="PLT1.*", axis=1).columns.empty
                    and not self.df_1.filter(regex="PLT3.*", axis=1).columns.empty
                ):
                    self.plot_widget_1.hide()
                    self.plot_widget_3.show()
                    self.plt_3.show()
                    self.splitter.setSizes([50, 40, 0, 400, 150])

                # Set PLT1 and PLT2 feature space
                elif (
                    not self.df_1.filter(regex="PLT1.*", axis=1).columns.empty
                    and self.df_1.filter(regex="PLT3.*", axis=1).columns.empty
                ):
                    self.plot_widget_1.show()
                    self.plot_widget_3.hide()
                    self.plt_1.show()
                    self.splitter.setSizes([50, 40, 150, 400, 0])

                    self.plt_2.getAxis("bottom").setStyle(showValues=True)
                    self.plt_2.getAxis("bottom").setPen(
                        pg.mkPen(color="#606060", width=1)
                    )
                    self.plt_2.getAxis("bottom").setTickFont(self.font_axis)
                    self.plt_2.getAxis("bottom").setTextPen("#C0C0C0")

            features_PLT1 = []
            for i in self.df_1.filter(regex="PLT1.*", axis=1):
                name_string = re.match(r"^(.*?)_", i).group(1)
                color_string = re.search(r"_([^_]+)$", i).group(1)
                strings = f"{name_string}_{color_string}"
                features_PLT1.append(strings)

                if "PLT1_GREEN" in i:
                    self.plt_1.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#00994C", "width": 0.7},
                    )
                if "PLT1_WHITE" in i:
                    self.plt_1.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FFFAF0", "width": 0.7},
                    )
                if "PLT1_BLUE" in i:
                    self.plt_1.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#3399FF", "width": 0.7},
                    )
                if "PLT1_RED" in i:
                    self.plt_1.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FF3333", "width": 0.7},
                    )
                if "PLT1_YELLOW" in i:
                    self.plt_1.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FFFF00", "width": 0.7},
                    )
                if "PLT1_ORANGE" in i:
                    self.plt_1.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FF8000", "width": 0.7},
                    )
                if "PLT1_MAGENTA" in i:
                    self.plt_1.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FF00FF", "width": 0.7},
                    )
                if "PLT1_CYAN" in i:
                    self.plt_1.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#00FFFF", "width": 0.7},
                    )

            if features_PLT1:
                result_plt1 = " | ".join(features_PLT1)
                self.plt_1.setTitle(f"PLT1: {result_plt1}")

            features_PLT2 = []
            for i in self.df_1.filter(regex="PLT2.*", axis=1):
                name_string = re.match(r"^(.*?)_", i).group(1)
                color_string = re.search(r"_([^_]+)$", i).group(1)
                strings = f"{name_string}_{color_string}"
                features_PLT2.append(strings)

                if "PLT2_GREEN" in i:
                    self.plt_2.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#00994C", "width": 0.7},
                    )
                if "PLT2_WHITE" in i:
                    self.plt_2.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FFFAF0", "width": 0.7},
                    )
                if "PLT2_BLUE" in i:
                    self.plt_2.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#3399FF", "width": 0.7},
                    )
                if "PLT2_RED" in i:
                    self.plt_2.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FF3333", "width": 0.7},
                    )
                if "PLT2_YELLOW" in i:
                    self.plt_2.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FFFF00", "width": 0.7},
                    )
                if "PLT2_ORANGE" in i:
                    self.plt_2.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FF8000", "width": 0.7},
                    )
                if "PLT2_MAGENTA" in i:
                    self.plt_2.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FF00FF", "width": 0.7},
                    )
                if "PLT2_CYAN" in i:
                    self.plt_2.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#00FFFF", "width": 0.7},
                    )

            if features_PLT2:
                result_plt2 = " | ".join(features_PLT2)
                self.plt_2.setTitle(f"PLT2: {result_plt2}")

            features_PLT3 = []
            for i in self.df_1.filter(regex="PLT3.*", axis=1):
                name_string = re.match(r"^(.*?)_", i).group(1)
                color_string = re.search(r"_([^_]+)$", i).group(1)
                strings = f"{name_string}_{color_string}"
                features_PLT3.append(strings)

                if "PLT3_GREEN" in i:
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#00994C", "width": 0.7},
                    )
                if "PLT3_WHITE" in i:
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FFFAF0", "width": 0.7},
                    )
                if "PLT3_BLUE" in i:
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#3399FF", "width": 0.7},
                    )
                if "PLT3_RED" in i:
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FF3333", "width": 0.7},
                    )
                if "PLT3_YELLOW" in i:
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FFFF00", "width": 0.7},
                    )
                if "PLT3_ORANGE" in i:
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FF8000", "width": 0.7},
                    )
                if "PLT3_MAGENTA" in i:
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#FF00FF", "width": 0.7},
                    )
                if "PLT3_CYAN" in i:
                    self.plt_3.plot(
                        self.df_1.dateindex.values,
                        self.df_1[i].values,
                        pen={"color": "#00FFFF", "width": 0.7},
                    )

            if features_PLT3:
                result_plt3 = " | ".join(features_PLT3)
                self.plt_3.setTitle(f"PLT3: {result_plt3}")

            # Scatter plot y_pred
            self._process_scatter()
            self.plt_2.plot(self.df_1.dateindex.values, self.df_1.c.values)
            self.plt_2.addItem(self.scatter_long_pred)
            self.plt_2.addItem(self.scatter_short_pred)

            self.plt_2.addItem(self.scatter_short_pred_plus)
            self.plt_2.addItem(self.scatter_short_pred_minus)
            self.plt_2.addItem(self.scatter_long_pred_plus)
            self.plt_2.addItem(self.scatter_long_pred_minus)

            self.plt_2.addItem(self.scatter_exit_pred_long)
            self.plt_2.addItem(self.scatter_exit_pred_short)
            self.plt_2.addItem(self.scatter_exit_gain_pred)
            self.plt_2.addItem(self.scatter_exit_stop_pred)

            self.plt_1.enableAutoRange(x=True, y=True)
            self.plt_2.enableAutoRange(x=True, y=True)
            self.plt_3.enableAutoRange(x=True, y=True)

            if self.checkbox_scatter.isChecked():
                self._showhide_scatters(True)
            else:
                self._showhide_scatters(False)

        elif self.showplt3 == 2:
            self.showplt3 = 1

            self.plt_1.clear()
            self.plt_2.clear()
            self.plt_3.clear()

            self.plt_1.enableAutoRange(x=True, y=True)
            self.plt_2.enableAutoRange(x=True, y=True)
            self.plt_3.enableAutoRange(x=True, y=True)

            self._show_plot()

    def _show_cumulative_gains(self):
        """
        Manage de the cumulative gains in the interface performance.
        """
        self.plt_3.clear()

        if self.showplt4 == 1:
            # Cumulative Amount  Curve
            self.plt_3.setTitle("Cumulative Gains Strategy (Orange)  X  Hold (Grey)")
            self.plt_3.plot(
                self.df_1.dateindex.values,
                self.df_1.cumul_gains_str.values,
                pen={"color": "#FF9933", "width": 0.7},
            )
            self.plt_3.plot(
                self.df_1.dateindex.values,
                self.df_1.cumul_gains_hold.values,
                pen={"color": "#A9A9A9", "width": 0.4},
            )

            # Scatter plot y_pred
            self.showplt5 = 2
            self.showplt4 = 2

        elif self.showplt4 == 2:
            # Cumulative Amount  Curve
            self.plt_3.setTitle(
                "Cumulative Gains Reapplied -  Strategy (Orange)  X  Hold (Grey)"
            )
            self.plt_3.plot(
                self.df_1.dateindex.values,
                self.df_1.cumul_gains_reappl_str.values,
                pen={"color": "#FF9933", "width": 0.7},
            )
            self.plt_3.plot(
                self.df_1.dateindex.values,
                self.df_1.cumul_gains_reappl_hold.values,
                pen={"color": "#A9A9A9", "width": 0.4},
            )

            # Scatter plot y_pred
            self.showplt5 = 2
            self.showplt4 = 3

        else:
            # Equity Curve
            if np.all(np.isnan(self.pos_true)):
                self.plt_3.setTitle("Equity Curve")
            else:
                self.plt_3.setTitle("Equity Curve y-pred (Blue)  |  y-true (Green)")
                self.plt_3.plot(
                    self.df_1.dateindex.values,
                    self.equity_curve_true.values,
                    pen={"color": "g", "width": 0.3},
                )
            self.plt_3.plot(
                self.df_1.dateindex.values,
                self.equity_curve_pred.values,
                pen={"color": "#63B8FF", "width": 0.7},
            )

            # Scatter plot y_pred
            self.showplt5 = 2
            self.showplt4 = 1

        self._process_scatter()
        self.plt_3.addItem(self.scatter_long_pred)
        self.plt_3.addItem(self.scatter_short_pred)

        self.plt_3.addItem(self.scatter_short_pred_plus)
        self.plt_3.addItem(self.scatter_short_pred_minus)
        self.plt_3.addItem(self.scatter_long_pred_plus)
        self.plt_3.addItem(self.scatter_long_pred_minus)

        self.plt_3.addItem(self.scatter_exit_pred_long)
        self.plt_3.addItem(self.scatter_exit_pred_short)
        self.plt_3.addItem(self.scatter_exit_gain_pred)
        self.plt_3.addItem(self.scatter_exit_stop_pred)

        self.plt_3.addItem(self.x_line_plt3, ignoreBounds=True)
        self.plt_3.addItem(self.y_line_plt3, ignoreBounds=True)

        self.plt_1.enableAutoRange()
        self.plt_2.enableAutoRange()
        self.plt_3.enableAutoRange()

        if self.checkbox_scatter.isChecked():
            self._showhide_scatters(True)
        else:
            self._showhide_scatters(False)

    def _show_returns(self):
        """
        Processes and manages the analysis of returns in the performance interface.
        """

        if self.showplt11 == 1:
            self.showplt11 = 2
            self.value_var_hist_axis.value = 0
            self.pnl_textitem.hide()
            self.returns_textitem.hide()
            self.plt_2.clear()
            self.plt_2.setXLink(self.plt_3)
            self.plt_2.setTitle("Returns - Strategy (Green) | Market (Grey)")
            self.plt_2.getAxis("bottom").setStyle(showValues=False)
            self.plt_2.plot(
                self.df_1.dateindex.values,
                (self.strategy_returns_pred * 100).values,
                pen={"color": "g", "width": 0.7},
            )
            self.plt_2.plot(
                self.df_1.dateindex.values,
                (self.returns * 100).values,
                pen={"color": "#A9A9A9", "width": 0.4},
            )

        elif self.showplt11 == 2:
            self.showplt11 = 3
            self.value_var_hist_axis.value = 10
            self.plt_2.clear()
            self.plt_2.setXLink(self.plt_2)
            self.pnl_textitem.hide()
            self.plt_2.setTitle(
                "Returns Distribution - Strategy (Green) | Market (Grey)"
            )
            self.plt_2.getAxis("bottom").setStyle(showValues=True)
            self.plt_2.getAxis("bottom").setPen(pg.mkPen(color="#606060", width=1))
            self.plt_2.getAxis("bottom").setTickFont(self.font_axis)
            self.plt_2.getAxis("bottom").setTextPen("#C0C0C0")
            self.plt_2.getAxis("right").setWidth(int(65))

            text = "STRATEGY  (MEAN: {}%  ".format(
                decimal_round(self.strategy_returns_pred.mean() * 100, 2)
            ) + "STD: {}%)     ".format(
                decimal_round(self.strategy_returns_pred.std() * 100, 2)
            )

            self.returns_textitem.setText(text=text)
            self.returns_textitem.show()

            # Histogram - Strategy
            y_1, x_1 = np.histogram(
                self.strategy_returns_pred.values, bins=self.dist_bins, density=False
            )

            bin_centers = 0.5 * (x_1[:-1] + x_1[1:])
            # Bar plot
            bar_graph = pg.BarGraphItem(
                x=bin_centers,
                height=y_1,
                width=(x_1[1] - x_1[0]),
                brush=pg.mkBrush("green"),
            )
            self.plt_2.addItem(bar_graph)

            # Histogram - Market
            returns_ = self.returns.fillna(0)
            y_1, x_1 = np.histogram(returns_.values, bins=self.dist_bins, density=False)

            bin_centers = 0.5 * (x_1[:-1] + x_1[1:])
            # Bar plot
            bar_graph_ = pg.BarGraphItem(
                x=bin_centers,
                pen="#696969",
                height=y_1,
                width=(x_1[1] - x_1[0]),
                brush=pg.mkBrush("#69696980"),
            )
            self.plt_2.addItem(bar_graph_)

        elif self.showplt11 == 3:
            self.showplt11 = 4
            self.value_var_hist_axis.value = 10
            self.plt_2.clear()
            self.plt_2.setXLink(self.plt_2)
            self.pnl_textitem.hide()
            self.plt_2.setTitle("Trads Returns Strategy - Distribution")
            self.plt_2.getAxis("bottom").setStyle(showValues=True)
            self.plt_2.getAxis("bottom").setPen(pg.mkPen(color="#606060", width=1))
            self.plt_2.getAxis("bottom").setTickFont(self.font_axis)
            self.plt_2.getAxis("bottom").setTextPen("#C0C0C0")
            self.plt_2.getAxis("right").setWidth(int(65))

            # Filter trades (inputs / outputs) and calculates pct_change for each trade
            df_rtn = pd.DataFrame()
            df_rtn["trad_prices"] = self.df_1.c.loc[self.df_1.signals_pred != 0]
            df_rtn["trad_positions"] = self.df_1.signals_pred.loc[
                self.df_1.positions_pred != 0
            ]
            df_rtn["trad_positions"] = df_rtn["trad_positions"].fillna(0)
            df_rtn.reset_index(drop=True, inplace=True)
            prices = df_rtn.trad_prices.values
            signals_positions = df_rtn.trad_positions.values

            # Calculando as variações percentuais de forma vetorizada
            price_diffs = prices[1:] - prices[:-1]
            pct_changes = (price_diffs / prices[:-1]) * signals_positions[:-1]

            self.trads_pct_changes = pct_changes[pct_changes != 0]

            # Treat zero
            if self.trads_pct_changes.size == 0:
                self.trads_pct_changes = np.array([0])

            text = (
                "n-TRADS: {}  ".format(self.n_trads)
                + "(MEAN: {}%  ".format(
                    decimal_round(self.trads_pct_changes.mean() * 100, 2)
                )
                + "STD: {}%)     ".format(
                    decimal_round(self.trads_pct_changes.std() * 100, 2)
                )
            )

            self.returns_textitem.setText(text=text)
            self.returns_textitem.show()

            # Histogram - Strategy
            y_1, x_1 = np.histogram(
                self.trads_pct_changes, bins=self.dist_bins, density=False
            )

            bin_centers = 0.5 * (x_1[:-1] + x_1[1:])
            # Bar plot
            bar_graph = pg.BarGraphItem(
                x=bin_centers,
                height=y_1,
                width=(x_1[1] - x_1[0]),
                brush=pg.mkBrush("#00FF0060"),
            )
            self.plt_2.addItem(bar_graph)

        elif self.showplt11 == 4:
            self.showplt11 = 5
            self.value_var_hist_axis.value = 10
            self.plt_2.clear()
            self.plt_2.setTitle("Trads Returns Strategy - PDF")

            # Probability Distribution Function
            x = np.sort(self.trads_pct_changes)
            pdf = stats.norm.pdf
            self.plt_2.plot(
                x,
                pdf(
                    (x - np.mean(self.trads_pct_changes))
                    / (np.std(self.trads_pct_changes) + self.treat_zerodiv_factor)
                ),
                pen={"color": "#5EF38C", "width": 1},
                fillLevel=0,
                brush="#00FF0020",
            )

        elif self.showplt11 == 5:
            self.showplt11 = 6

            self.value_var_hist_axis.value = 10
            self.plt_2.clear()
            self.plt_2.setXLink(self.plt_2)
            self.pnl_textitem.hide()
            self.plt_2.setTitle("Absolute Trads PDF - Profits (Green) & Losses (Red)")
            self.plt_2.getAxis("bottom").setStyle(showValues=True)
            self.plt_2.getAxis("bottom").setPen(pg.mkPen(color="#606060", width=1))
            self.plt_2.getAxis("bottom").setTickFont(self.font_axis)
            self.plt_2.getAxis("bottom").setTextPen("#C0C0C0")
            self.plt_2.getAxis("right").setWidth(int(65))

            trads_pct_changes_profit = self.trads_pct_changes[
                self.trads_pct_changes > 0
            ]
            trads_pct_changes_loss = (
                self.trads_pct_changes[self.trads_pct_changes < 0] * -1
            )

            # Treat zero
            if trads_pct_changes_profit.size == 0:
                trads_pct_changes_profit = np.array([0])
            if trads_pct_changes_loss.size == 0:
                trads_pct_changes_loss = np.array([0])

            # PDF - Trads Losses
            x = np.sort(trads_pct_changes_loss)
            pdf = stats.norm.pdf
            self.plt_2.plot(
                x,
                pdf(
                    (x - np.mean(trads_pct_changes_loss))
                    / (np.std(trads_pct_changes_loss) + self.treat_zerodiv_factor)
                ),
                pen={"color": "#FF3333", "width": 1},
                fillLevel=0,
                brush="#FF333320",
            )

            # PDF - Trads profits
            x = np.sort(trads_pct_changes_profit)
            pdf = stats.norm.pdf
            self.plt_2.plot(
                x,
                pdf(
                    (x - np.mean(trads_pct_changes_profit))
                    / (np.std(trads_pct_changes_profit) + self.treat_zerodiv_factor)
                ),
                pen={"color": "#5EF38C", "width": 1},
                fillLevel=0,
                brush="#00FF0020",
            )

            text = "P: (MEAN: {}%  STD: {}%)  ".format(
                decimal_round(np.mean(trads_pct_changes_profit) * 100, 2),
                decimal_round(np.std(trads_pct_changes_profit) * 100, 2),
            ) + "L: (MEAN: {}%  STD: {}%)  ".format(
                decimal_round(np.mean(trads_pct_changes_loss) * 100, 2),
                decimal_round(np.std(trads_pct_changes_loss) * 100, 2),
            )

            self.returns_textitem.setText(text=text)
            self.returns_textitem.show()

        elif self.showplt11 == 6:
            self.showplt11 = 1

            self.value_var_hist_axis.value = 0
            self.pnl_textitem.show()
            self.returns_textitem.hide()
            self.plt_2.clear()
            self.plt_2.setXLink(self.plt_3)
            self.plt_2.setTitle("Strategy (Green)  |  Hold (Grey)")
            self.plt_2.getAxis("bottom").setStyle(showValues=False)
            self.plt_2.addLine(x=None, y=0, pen={"color": "red", "width": 0.3})
            self.plt_2.plot(
                self.df_1.dateindex.values,
                self.market_returns_cum.values,
                pen={"color": "#A9A9A9", "width": 0.4},
            )
            self.plt_2.plot(
                self.df_1.dateindex.values,
                self.strategy_returns_cum.values,
                pen={"color": "g", "width": 0.7},
            )

        self.plt_2.addItem(self.x_line_plt2, ignoreBounds=True)
        self.plt_2.addItem(self.y_line_plt2, ignoreBounds=True)

        self.plt_1.enableAutoRange(x=True, y=True)
        self.plt_2.enableAutoRange(x=True, y=True)
        self.plt_3.enableAutoRange(x=True, y=True)

    def _showhide_scatters(self, state: bool):
        """
        Disable and enable scatters symbols on the signals and performance interface.

        Args:
            state (bool): Set the sate of scatters visibility (True: show | False: hide)
        """

        if self.pos_true is not None and not np.all(np.isnan(self.pos_true)):
            # True
            self.scatter_long_true.setVisible(state)
            self.scatter_short_true.setVisible(state)
            #
            self.scatter_short_true_plus.setVisible(state)
            self.scatter_short_true_minus.setVisible(state)
            self.scatter_long_true_plus.setVisible(state)
            self.scatter_long_true_minus.setVisible(state)
            #
            self.scatter_exit_true_long.setVisible(state)
            self.scatter_exit_true_short.setVisible(state)
            self.scatter_exit_gain_true.setVisible(state)
            self.scatter_exit_stop_true.setVisible(state)

        if self.pos_pred is not None:
            # Pred
            self.scatter_long_pred.setVisible(state)
            self.scatter_short_pred.setVisible(state)
            #
            self.scatter_short_pred_plus.setVisible(state)
            self.scatter_short_pred_minus.setVisible(state)
            self.scatter_long_pred_plus.setVisible(state)
            self.scatter_long_pred_minus.setVisible(state)
            #
            self.scatter_exit_pred_long.setVisible(state)
            self.scatter_exit_pred_short.setVisible(state)
            self.scatter_exit_gain_pred.setVisible(state)
            self.scatter_exit_stop_pred.setVisible(state)

    def _show_pricedistribution(self):
        """
        Processes and manages the price distribution interface.
        """
        if self.showplt7 == 1:
            self.value_var_hist_axis.value = 20
            self.plt_1.clear()
            self.plt_1.hide()
            self.hit_trads_textitem.hide()
            self.scores_textitem.hide()
            self.returns_textitem.hide()
            self.plot_widget_1.hide()
            self.plot_widget_2.show()
            self.plot_widget_3.show()
            self.splitter.setSizes([50, 40, 0, 400, 200])

            self.plt_1.enableAutoRange(x=True, y=True)
            self.plt_2.enableAutoRange(x=True, y=True)
            self.plt_3.enableAutoRange(x=True, y=True)

            # Histogram
            y_1, x_1 = np.histogram(
                self.df_1.c.values, bins=self.dist_bins, density=False
            )

            bin_centers = 0.5 * (x_1[:-1] + x_1[1:])
            # Bar plot
            bar_graph = pg.BarGraphItem(
                x=bin_centers,
                height=y_1,
                width=(x_1[1] - x_1[0]),
                brush=pg.mkBrush("#27408B"),
            )
            # Set distribution values
            text = (
                "MEAN: {}     ".format(decimal_round(self.df_1.c.mean(), 1))
                + "MEDIAN: {}     ".format(decimal_round(self.df_1.c.median(), 1))
                + "ASYMMETRY: {}    ".format(decimal_round(self.df_1.c.skew(), 1))
                + "STD: {} ".format(decimal_round(self.df_1.c.std(), 1))
            )
            self.dist_textitem.setText(text=text)
            self.dist_textitem.show()

            self.plt_1.clear()
            self.plt_2.clear()
            self.plt_3.clear()
            self.button_2.hide()
            self.plt_2.addItem(bar_graph)
            self.plt_2.setTitle("Price distribution")
            self.plt_3.setTitle("Price - CDF")
            self.showplt7 = 2
            self.showplt3 = 1
            self.showplt1 = 1

            # Cumulative Distribution Function
            cdf = stats.norm.cdf
            x = np.linspace(
                np.min(self.df_1.c), np.max(self.df_1.c), self.df_1.c.shape[0]
            )
            self.plt_3.plot(
                x,
                cdf((x - np.mean(self.df_1.c)) / np.std(self.df_1.c)) * 100,
                pen={"color": "#99CCFF", "width": 1},
                fillLevel=0,
                brush="#27408B50",
            )
            # Add line 95% confidence
            percentile_95_value = stats.norm.ppf(
                0.95, loc=np.mean(self.df_1.c), scale=np.std(self.df_1.c)
            )
            percentile_95_y = (
                cdf((percentile_95_value - np.mean(self.df_1.c)) / np.std(self.df_1.c))
                * 100
            )
            line_95 = pg.PlotCurveItem(
                x=np.full((50,), percentile_95_value),
                y=np.linspace(0, percentile_95_y, 50),
                pen={"color": "grey", "width": 1},
            )
            self.plt_3.addItem(line_95)
            self.plt_3.show()

            self.plt_2.addItem(self.x_line_plt2, ignoreBounds=True)
            self.plt_2.addItem(self.y_line_plt2, ignoreBounds=True)
            self.plt_3.addItem(self.x_line_plt3, ignoreBounds=True)
            self.plt_3.addItem(self.y_line_plt3, ignoreBounds=True)

        else:
            self.plt_2.clear()
            self.plt_3.clear()
            self.showplt1 = 1
            self.showplt7 = 1
            self._show_plot()

    def _show_monte_carlo_simulation(self):
        """
        Processes and manages the Monte Carlo test inteface.
        """

        def show_mc_ddown__mc_ecurve():
            """
            Switch between mc_drawdowns and mc_equitycurves analyses.
            """
            # Drawdowns
            if self.showplt8 == 1:
                self.showplt10 = 1
                self.showplt8 = 0
                self.showplt13 = 1
                self.mc_plot_widget_1.show()
                self.mc_plot_widget_2.hide()
                self.mc_plot_widget_3.show()
                self.mc_plot_widget_4.hide()
                self.mc_plot_widget_5.show()
                self.mc_plot_widget_6.hide()
                self.mc_plot_widget_7.hide()
                self.mc_range_dd.setParentItem(self.mc_plt_5)
                self.mc_button_1.setText("EC")
                self.mc_button_2.setText("PDF")
                self.mc_splitter.setSizes([50, 200, 200, 200, 200, 200, 200, 200])

            # EquityCurve
            else:
                self.showplt10 = 1
                self.showplt8 = 1
                self.showplt13 = 1
                self.mc_plot_widget_1.show()
                self.mc_plot_widget_2.show()
                self.mc_plot_widget_3.hide()
                self.mc_plot_widget_4.show()
                self.mc_plot_widget_5.hide()
                self.mc_plot_widget_6.hide()
                self.mc_plot_widget_7.hide()
                self.mc_range_ec.setParentItem(self.mc_plt_4)
                self.mc_button_1.setText("DD")
                self.mc_button_2.setText("PDF")
                self.mc_splitter.setSizes([50, 200, 200, 200, 200, 200, 200, 200])

        def show_mc_pdf__mc_cdf():
            """
            Switch between pdf and cdf analyses.
            """

            if self.showplt8 == 1:
                if self.showplt13 == 1:
                    self.mc_button_2.setText("CDF")
                    self.showplt13 = 2
                    self.mc_plot_widget_4.hide()
                    self.mc_plot_widget_6.show()
                    self.mc_range_ec.setParentItem(self.mc_plt_6)
                else:
                    self.mc_button_2.setText("PDF")
                    self.showplt13 = 1
                    self.mc_plot_widget_4.show()
                    self.mc_plot_widget_6.hide()
                    self.mc_range_ec.setParentItem(self.mc_plt_4)
            else:
                if self.showplt13 == 2:
                    self.mc_button_2.setText("PDF")
                    self.showplt13 = 1
                    self.mc_plot_widget_5.show()
                    self.mc_plot_widget_7.hide()
                    self.mc_range_dd.setParentItem(self.mc_plt_5)
                else:
                    self.mc_button_2.setText("CDF")
                    self.showplt13 = 2
                    self.mc_plot_widget_5.hide()
                    self.mc_plot_widget_7.show()
                    self.mc_range_dd.setParentItem(self.mc_plt_7)

        # Process Monte Carlo Simulation
        if self.showplt6 == 1:
            self.showplt6 = 2
            print(
                f"\n{Fore.LIGHTYELLOW_EX}STARTING MONTE CARLO SIMULATION...{Fore.RESET}"
            )
            self.mc_label.show()
            if (
                self.sim_params is not None or self.sim_bayesopt_spaces is not None
            ) and self.strategy is not None:
                self.button_10.setEnabled(False)

            self.mc_label.setText(
                "MONTE CARLO TEST - PROCESSING DATA..." + str(self.mc_nsim) + " PATHS"
            )

            # MainWindow 2
            self.win_2 = QtWidgets.QMainWindow()
            self.win_2.setWindowTitle(BAR_TITLE)
            self.win_2.setGeometry(200, 180, 1300, 700)
            self.win_2.setStyleSheet("background-color: #292929;")

            # Set widgets layouts
            self.mc_plot_widget_0 = pg.GraphicsLayoutWidget()
            self.mc_plot_widget_1 = pg.GraphicsLayoutWidget()
            self.mc_plot_widget_2 = pg.GraphicsLayoutWidget()
            self.mc_plot_widget_3 = pg.GraphicsLayoutWidget()
            self.mc_plot_widget_4 = pg.GraphicsLayoutWidget()
            self.mc_plot_widget_5 = pg.GraphicsLayoutWidget()
            self.mc_plot_widget_6 = pg.GraphicsLayoutWidget()
            self.mc_plot_widget_7 = pg.GraphicsLayoutWidget()

            # Fix widget_0 height
            self.mc_plot_widget_0.setFixedHeight(40)

            # Set widgets bg-color
            self.mc_plot_widget_0.setBackground(background="#292929")
            self.mc_plot_widget_1.setBackground(background="#292929")
            self.mc_plot_widget_2.setBackground(background="#292929")
            self.mc_plot_widget_3.setBackground(background="#292929")
            self.mc_plot_widget_4.setBackground(background="#292929")
            self.mc_plot_widget_5.setBackground(background="#292929")
            self.mc_plot_widget_6.setBackground(background="#292929")
            self.mc_plot_widget_7.setBackground(background="#292929")

            # Set QSplitter
            self.mc_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

            # Add widgets in splitter
            self.mc_splitter.addWidget(self.mc_plot_widget_0)
            self.mc_splitter.addWidget(self.mc_plot_widget_1)
            self.mc_splitter.addWidget(self.mc_plot_widget_2)
            self.mc_splitter.addWidget(self.mc_plot_widget_3)
            self.mc_splitter.addWidget(self.mc_plot_widget_4)
            self.mc_splitter.addWidget(self.mc_plot_widget_5)
            self.mc_splitter.addWidget(self.mc_plot_widget_6)
            self.mc_splitter.addWidget(self.mc_plot_widget_7)

            # Splitter settings
            self.mc_splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: black;
                    height: 1px; 
                }
            """)

            self.mc_plt_1 = self.mc_plot_widget_1.addPlot()
            self.mc_plt_2 = self.mc_plot_widget_2.addPlot()
            self.mc_plt_3 = self.mc_plot_widget_3.addPlot()
            self.mc_plt_4 = self.mc_plot_widget_4.addPlot()
            self.mc_plt_5 = self.mc_plot_widget_5.addPlot()
            self.mc_plt_6 = self.mc_plot_widget_6.addPlot()
            self.mc_plt_7 = self.mc_plot_widget_7.addPlot()

            # Set splitter top margin
            self.mc_splitter.setContentsMargins(0, 0, 0, 0)

            # Centralize splitter in win_2
            self.win_2.setCentralWidget(self.mc_splitter)

            # Set initial config
            self.mc_plot_widget_1.show()
            self.mc_plot_widget_2.show()
            self.mc_plot_widget_3.hide()
            self.mc_plot_widget_4.show()
            self.mc_plot_widget_5.hide()
            self.mc_plot_widget_6.hide()
            self.mc_plot_widget_7.hide()

            # Set sizes
            self.mc_splitter.setSizes([50, 200, 200, 200, 200, 200, 200, 200])

            # "DD" - Monte Carlo Drawdowns
            self.mc_button_1 = QPushButton(self.mc_plot_widget_0)
            self.mc_button_1.clicked.connect(show_mc_ddown__mc_ecurve)
            self.mc_button_1.setGeometry(10, 10, 30, 20)
            self.mc_button_1.setText("DD")
            self.mc_button_1.show()
            self.mc_button_1.setStyleSheet(
                "font: bold 11pt; color: white; background-color: #483D8B; "
                "border-radius: 1px; border: 1px outset grey;"
            )

            # "PDF" - Monte Carlo Drawdowns
            self.mc_button_2 = QPushButton(self.mc_plot_widget_0)
            self.mc_button_2.clicked.connect(show_mc_pdf__mc_cdf)
            self.mc_button_2.setGeometry(50, 10, 40, 20)
            self.mc_button_2.setText("PDF")
            self.mc_button_2.show()
            self.mc_button_2.setStyleSheet(
                "font: bold 11pt; color: white; background-color: #483D8B; "
                "border-radius: 1px; border: 1px outset grey;"
            )

            plots = [
                self.mc_plt_1,
                self.mc_plt_2,
                self.mc_plt_3,
                self.mc_plt_4,
                self.mc_plt_5,
                self.mc_plt_6,
                self.mc_plt_7,
            ]

            # Set MC Plots
            for plot in plots:
                # Font Tick
                plot.getAxis("right").setTickFont(self.font_axis)
                plot.getAxis("bottom").setTickFont(self.font_axis)

                # Color Tick
                plot.getAxis("right").setTextPen("#C0C0C0")
                plot.getAxis("bottom").setTextPen("#C0C0C0")

                # Config Grid
                plot.showGrid(x=True, y=True, alpha=0.2)

                # Config Axis
                plot.showAxis("right")
                plot.showAxis("left")
                plot.showAxis("top")
                plot.getAxis("left").setStyle(showValues=False)
                plot.getAxis("top").setStyle(showValues=False)
                plot.getAxis("right").setWidth(50)

                # Config Frame
                plot.getAxis("bottom").setPen(pg.mkPen(color="#606060", width=1))
                plot.getAxis("right").setPen(pg.mkPen(color="#606060", width=1))
                plot.getAxis("top").setPen(pg.mkPen(color="#606060", width=1))
                plot.getAxis("left").setPen(pg.mkPen(color="#606060", width=1))

                # Disable Auto Range
                plot.disableAutoRange()

                # Performance
                plot.setClipToView(True)
                plot.setDownsampling(auto=True, mode="subsample", ds=2)

            # Set Titles
            mc_title_var = f"{self.mc_mode.title()}"
            self.mc_plt_1.setTitle(f"Monte Carlo Test - EquityCurves  |  {mc_title_var}")
            self.mc_plt_2.setTitle("EquityCurves Distribution")
            self.mc_plt_3.setTitle("Drawdowns Distribution")
            self.mc_plt_4.setTitle("EquityCurves - CDF (Normal)")
            self.mc_plt_5.setTitle("Drawdowns - CDF (Lognormal)")
            self.mc_plt_6.setTitle("EquityCurves - PDF (Normal)")
            self.mc_plt_7.setTitle("Drawdowns - PDF (Lognormal)")

            # Average Drawdown Label
            self.mc_range_dd = pg.TextItem(color="#FF3333")
            self.mc_range_dd.hide()
            self.mc_range_dd.setParentItem(self.mc_plt_5)
            self.mc_range_dd.setPos(10, 5)
            self.mc_range_dd.setFont(self.font)

            # Average EquityCurve Label
            self.mc_range_ec = pg.TextItem(color="#99CCFF")
            self.mc_range_ec.hide()
            self.mc_range_ec.setParentItem(self.mc_plt_4)
            self.mc_range_ec.setPos(10, 5)
            self.mc_range_ec.setFont(self.font)

            # EquityCurve values
            self.mc_ec_values = pg.TextItem(color="#99CCFF")
            self.mc_ec_values.hide()
            self.mc_ec_values.setParentItem(self.mc_plt_2)
            self.mc_ec_values.setPos(10, 5)
            self.mc_ec_values.setFont(self.font)

            # Drawdown values
            self.mc_dd_values = pg.TextItem(color="#FF3333")
            self.mc_dd_values.hide()
            self.mc_dd_values.setParentItem(self.mc_plt_3)
            self.mc_dd_values.setPos(10, 5)
            self.mc_dd_values.setFont(self.font)

            # Set mc_s0
            if self.mc_s0 is None:
                self.mc_s0 = self.df_1.c.values[0]

            # Set default mc_nsteps
            if self.mc_nsteps is None:
                self.mc_nsteps = len(self.df_1) 

            # Clip mc_nsteps
            if self.mc_nsteps > len(self.df_1):
                self.mc_nsteps = len(self.df_1) 
                
            # Process dataframes diffs (strategy data-in x strategy data-out)
            self.df_diff_factor = len(self.str_params[0]) - len(self.strategy(self.str_params))
            self.str_params[0] = self.str_params[0][: self.mc_nsteps + self.df_diff_factor]

            # Set shared var
            self.value_var_mc = Value("d")
            self.value_var_mc.value = 0

            # Create temp files
            temp_file_1 = tempfile.NamedTemporaryFile(delete=True)
            filename_1 = temp_file_1.name
            temp_file_2 = tempfile.NamedTemporaryFile(delete=True)
            filename_2 = temp_file_2.name

            # Map memory shared
            self.np_mem_1 = np.memmap(
                filename_1,
                dtype="float",
                mode="w+",
                shape=(self.mc_nsim, self.mc_nsteps),
            )
            self.np_mem_2 = np.memmap(
                filename_2,
                dtype="float",
                mode="w+",
                shape=(self.mc_nsim, self.mc_nsteps),
            )

            price_paths = get_mc_price_paths(
                seed=self.seed,
                df_diff_factor=self.df_diff_factor,
                mc_mode=self.mc_mode,
                str_params=self.str_params,
                n_sim=self.mc_nsim,
                n_steps=self.mc_nsteps,
                sigma=self.mc_sigma,
                s0=self.mc_s0,
                r=self.mc_r,
                dt=self.mc_dt,
                lambda_=self.mc_lambda_,
                mu_y=self.mc_mu_y,
                sigma_y=self.mc_sigma_y,
            )

            # Monte carlo parallel process
            self.mcProcess = ProcessMonteCarlo(
                self.seed,
                self.mc_mode,
                self.strategy,
                self.str_params,
                process_mc_strategy,
                self.mc_rndnpositions,
                price_paths,
                get_equitycurve,
                self.value_var_mc,
                self.np_mem_1,
                self.np_mem_2,
                get_drawdowns,
                self.df_1,
                self.initial_pos,
                self.df_diff_factor,
            )
            self.mcProcess.start()

            if self.mc_mode == "random_returns":
                self.showplt8 = 1
                self.mc_button_1.hide()
                show_mc_ddown__mc_ecurve()

            def update():
                if self.value_var_mc.value == 1:
                    self.value_var_mc.value = 0

                    self.mc_label.setText(
                        "MONTE CARLO TEST - PLOTTING "
                        + str(self.mc_nsim)
                        + " PATHS "
                        + " WITH "
                        + str(self.mc_nsteps)
                        + " STEPS "
                    )

                    self.mc_equity_curves = np.array(self.np_mem_1[:])
                    self.ddowns = np.array(self.np_mem_2[:])

                    # Path lines
                    # Multiline (paths)
                    if self.mc_line_plots:
                        y_3 = self.mc_equity_curves
                        x_3 = np.tile(np.arange(self.mc_nsteps), (self.mc_nsim, 1))
                        if self.mc_paths_colors:
                            lines = MultiColorLines(x_3, y_3)
                        else:
                            hline = pg.PlotCurveItem(
                                np.amax(self.mc_equity_curves, axis=0),
                                pen={"color": "b", "width": 1},
                            )
                            lline = pg.PlotCurveItem(
                                np.amin(self.mc_equity_curves, axis=0),
                                pen={"color": "b", "width": 1},
                            )
                            pfill = pg.FillBetweenItem(
                                hline, lline, brush=(50, 50, 200, 150)
                            )
                            self.mc_plt_1.addItem(hline)
                            self.mc_plt_1.addItem(lline)
                            self.mc_plt_1.addItem(pfill)
                            lines = MultiLines(x_3, y_3)
                        self.mc_plt_1.addItem(lines)

                    else:
                        hline = pg.PlotCurveItem(
                            np.amax(self.mc_equity_curves, axis=0),
                            pen={"color": "b", "width": 1},
                        )
                        lline = pg.PlotCurveItem(
                            np.amin(self.mc_equity_curves, axis=0),
                            pen={"color": "b", "width": 1},
                        )
                        pfill = pg.FillBetweenItem(
                            hline, lline, brush=(50, 50, 200, 150)
                        )
                        self.mc_plt_1.addItem(hline)
                        self.mc_plt_1.addItem(lline)
                        self.mc_plt_1.addItem(pfill)

                    # EquityCurves
                    cumul_gains_strategy = (
                        self.strategy_returns_pred.cumsum() + 1
                    ) * self.initial_pos
                    nplast_ec = self.mc_equity_curves[:, -1]
                    # Histogram EquityCurves
                    y_1_ec, x_1_ec = np.histogram(
                        nplast_ec, bins=self.mc_dist_bins, density=False
                    )
                    # Bar plot
                    bin_centers = 0.5 * (x_1_ec[:-1] + x_1_ec[1:])
                    bar_graph = pg.BarGraphItem(
                        x=bin_centers,
                        height=y_1_ec,
                        width=(x_1_ec[1] - x_1_ec[0]),
                        brush=pg.mkBrush("#27408B"),
                    )
                    self.mc_plt_2.addItem(bar_graph)
                    # Histogram Lines
                    meanline = pg.PlotCurveItem(
                        x=np.full((np.max(y_1_ec),), np.mean(nplast_ec)),
                        y=np.arange(np.max(y_1_ec)) / 6,
                        pen={"color": "yellow", "width": 2},
                    )
                    self.mc_plt_2.addItem(meanline)
                    # Cumulative Distribution Function
                    cdf = stats.norm.cdf
                    x = np.sort(nplast_ec)
                    self.mc_plt_4.plot(
                        x,
                        cdf((x - np.mean(nplast_ec)) / np.std(nplast_ec)) * 100,
                        pen={"color": "grey", "width": 1.2},
                        fillLevel=0,
                        brush="#27408B",
                    )
                    # Add mean line cdf
                    mean_x_ec = np.mean(nplast_ec)
                    mean_y_ec = (
                        cdf((mean_x_ec - np.mean(nplast_ec)) / np.std(nplast_ec)) * 100
                    )
                    meanline_ec_cdf = pg.PlotCurveItem(
                        x=np.full((51,), mean_x_ec),
                        y=np.linspace(0, mean_y_ec, 51),
                        pen={"color": "yellow", "width": 1},
                    )
                    self.mc_plt_4.addItem(meanline_ec_cdf)
                    # Add line 95% confidence
                    percentile_95_value = stats.norm.ppf(
                        0.95, loc=np.mean(nplast_ec), scale=np.std(nplast_ec)
                    )
                    percentile_95_y = (
                        cdf(
                            (percentile_95_value - np.mean(nplast_ec))
                            / np.std(nplast_ec)
                        )
                        * 100
                    )
                    line_95_ec = pg.PlotCurveItem(
                        x=np.full((51,), percentile_95_value),
                        y=np.linspace(0, percentile_95_y, 51),
                        pen={"color": "grey", "width": 1},
                    )
                    self.mc_plt_4.addItem(line_95_ec)
                    # Probability Distribution Function
                    pdf = stats.norm.pdf
                    self.mc_plt_6.plot(
                        x,
                        pdf((x - np.mean(nplast_ec)) / np.std(nplast_ec)),
                        pen={"color": "grey", "width": 1.2},
                        fillLevel=0,
                        brush="#27408B",
                    )
                    # Add mean line pdf
                    mean_x_ec = np.mean(nplast_ec)
                    mean_y_ec = pdf(
                        (mean_x_ec - np.mean(nplast_ec)) / np.std(nplast_ec)
                    )
                    meanline_ec_pdf = pg.PlotCurveItem(
                        x=np.full((51,), mean_x_ec),
                        y=np.linspace(0, mean_y_ec, 51),
                        pen={"color": "yellow", "width": 1},
                    )
                    self.mc_plt_6.addItem(meanline_ec_pdf)
                    #
                    self.mc_range_ec.setText(
                        f"EC-RANGE($): {str(round(min(nplast_ec), 2))} ~ {str(round(percentile_95_value, 2))} (95% CONFIDENCE)"
                    )
                    self.mc_range_ec.show()
                    text = (
                        "BET($): {}   ".format(str(self.initial_pos))
                        + "STRATEGY($): {}   ".format(
                            decimal_round(cumul_gains_strategy.values[-1], 2)
                        )
                        + "(MEAN($): {}   ".format(decimal_round(mean_x_ec, 2))
                        + "STD($): {})".format(decimal_round(np.std(nplast_ec), 2))
                    )
                    self.mc_ec_values.setText(text)
                    self.mc_ec_values.show()

                    # Drawdowns
                    # Vector of minimals of each drawdown row
                    npamin_dd = np.amin(self.ddowns, axis=1)
                    # Convert ddowns to positive percentages
                    npamin_dd = np.abs(npamin_dd) * 100
                    equity_curve_strategy = (self.strategy_returns_pred.cumsum()) + 1
                    # Remove zero values
                    npamin_dd = npamin_dd[npamin_dd > 0]
                    # Mean lognormal
                    log_data = np.log(npamin_dd)
                    mu = np.mean(log_data)
                    sigma2 = np.var(log_data)
                    mean_lognormal = np.exp(mu + sigma2 / 2)
                    # STD lognormal
                    std_lognormal = np.sqrt(
                        (np.exp(sigma2) - 1) * np.exp(2 * mu + sigma2)
                    )
                    # Histogram Drawdowns
                    y_1_dd, x_1_dd = np.histogram(
                        npamin_dd, bins=self.mc_dist_bins, density=False
                    )
                    # Bar plot
                    bin_centers = 0.5 * (x_1_dd[:-1] + x_1_dd[1:])
                    bar_graph = pg.BarGraphItem(
                        x=bin_centers,
                        height=y_1_dd,
                        width=(x_1_dd[1] - x_1_dd[0]),
                        brush=pg.mkBrush("#3f0e0e"),
                    )
                    self.mc_plt_3.addItem(bar_graph)
                    # Histogram Lines
                    meanline = pg.PlotCurveItem(
                        x=np.full((np.max(y_1_dd),), mean_lognormal),
                        y=np.arange(np.max(y_1_dd)) / 6,
                        pen={"color": "yellow", "width": 2},
                    )
                    self.mc_plt_3.addItem(meanline)
                    # Cumulative Distribution Function - Drawdown
                    # x = np.linspace(np.abs(np.min(self.ddowns) * 100), np.abs(np.max(self.ddowns) * 100), self.ddowns.shape[0])
                    x = np.sort(npamin_dd)
                    shape, loc, scale = stats.lognorm.fit(npamin_dd, floc=0)
                    cdf_log = stats.lognorm.cdf(x, shape, loc=loc, scale=scale) * 100
                    self.mc_plt_5.plot(
                        x,
                        cdf_log,
                        pen={"color": "grey", "width": 1.2},
                        fillLevel=0,
                        brush="#3f0e0e",
                    )
                    # Mean line CDF
                    mean_cdf_log = (
                        stats.lognorm.cdf(mean_lognormal, shape, loc=loc, scale=scale)
                        * 100
                    )
                    meanline_dd_cdf = pg.PlotCurveItem(
                        x=np.full((51,), mean_lognormal),
                        y=np.linspace(0, mean_cdf_log, 51),
                        pen={"color": "yellow", "width": 1.0},
                    )
                    self.mc_plt_5.addItem(meanline_dd_cdf)
                    # Add line 95% confidence
                    percentile_95_value = stats.lognorm.ppf(
                        0.95, shape, loc=loc, scale=scale
                    )
                    percentile_95_y = (
                        stats.lognorm.cdf(
                            percentile_95_value, shape, loc=loc, scale=scale
                        )
                        * 100
                    )
                    line_95 = pg.PlotCurveItem(
                        x=np.full((51,), percentile_95_value),
                        y=np.linspace(0, percentile_95_y, 51),
                        pen={"color": "grey", "width": 1},
                    )
                    self.mc_plt_5.addItem(line_95)
                    # Probability Distribution Function
                    shape, loc, scale = stats.lognorm.fit(npamin_dd, floc=0)
                    pdf_log = stats.lognorm.pdf(x, shape, loc=loc, scale=scale) * 100
                    self.mc_plt_7.plot(
                        x,
                        pdf_log,
                        pen={"color": "grey", "width": 1.2},
                        fillLevel=0,
                        brush="#3f0e0e",
                    )
                    # Mean line PDF
                    mean_cdf_log = (
                        stats.lognorm.pdf(mean_lognormal, shape, loc=loc, scale=scale)
                        * 100
                    )
                    meanline_dd_pdf = pg.PlotCurveItem(
                        x=np.full((51,), mean_lognormal),
                        y=np.linspace(0, mean_cdf_log, 51),
                        pen={"color": "yellow", "width": 1.0},
                    )
                    self.mc_plt_7.addItem(meanline_dd_pdf)
                    #
                    self.mc_range_dd.setText(
                        f"DD-RANGE(%): {decimal_round(min(npamin_dd), 2)} ~ {decimal_round(percentile_95_value, 2)} (95% CONFIDENCE)"
                    )
                    self.mc_range_dd.show()
                    #
                    text = (
                        "STRATEGY(%): {}    ".format(
                            decimal_round(
                                abs(
                                    min(
                                        get_drawdowns(pd.Series(equity_curve_strategy))[
                                            0
                                        ]
                                        * 100
                                    )
                                ),
                                2,
                            )
                        )
                        + "(MEAN(%): {}    ".format(decimal_round(mean_lognormal, 2))
                        + "STD(%): {})".format(decimal_round(std_lognormal, 2))
                    )
                    self.mc_dd_values.setText(text)
                    self.mc_dd_values.show()

                    # Strategy Original EquityCurve
                    equity_line = pg.PlotCurveItem(
                        cumul_gains_strategy.values,
                        pen={"color": "g", "width": 2},
                    )
                    self.mc_plt_1.addItem(equity_line)
                    strline = pg.PlotCurveItem(
                        x=np.full((np.max(y_1_ec),), cumul_gains_strategy.values[-1]),
                        y=np.arange(np.max(y_1_ec)) / 6,
                        pen={"color": "g", "width": 2},
                    )
                    self.mc_plt_2.addItem(strline)

                    # Strategy Original Drawdown
                    strline = pg.PlotCurveItem(
                        x=np.full(
                            (np.max(y_1_dd),),
                            abs(
                                min(
                                    get_drawdowns(pd.Series(equity_curve_strategy))[0]
                                    * 100
                                )
                            ),
                        ),
                        y=np.arange(np.max(y_1_dd)) / 6,
                        pen={"color": "g", "width": 2},
                    )
                    self.mc_plt_3.addItem(strline)

                    self.mc_plt_1.enableAutoRange()
                    self.mc_plt_2.enableAutoRange()
                    self.mc_plt_3.enableAutoRange()
                    self.mc_plt_4.enableAutoRange()
                    self.mc_plt_5.enableAutoRange()
                    self.mc_plt_6.enableAutoRange()
                    self.mc_plt_7.enableAutoRange()

                    if self.mcProcess.is_alive():
                        self.mcProcess.terminate()
                        self.mcProcess.join()
                    self.win_2.show()
                    self.timer_plot.stop()

            self.timer_plot = QtCore.QTimer()
            self.timer_plot.timeout.connect(update)
            self.timer_plot.start()

        elif self.showplt6 == 2:
            self.showplt6 = 1
            if (
                self.sim_params is not None or self.sim_bayesopt_spaces is not None
            ) and self.strategy is not None:
                self.button_10.setEnabled(True)
            self.mc_label.hide()
            self.mc_plt_1.clear()
            self.mc_plt_2.clear()
            self.mc_plt_3.clear()
            self.mc_plt_4.clear()
            self.mc_plt_5.clear()
            if self.timer_plot.isActive():
                self.timer_plot.stop()
                self.timer_plot.deleteLater()
            self.mc_plt_1.close()
            self.mc_plt_2.close()
            self.win_2.close()
            if self.mcProcess.is_alive():
                self.mcProcess.terminate()
            del self.np_mem_1, self.np_mem_2
            # Reset atributtes
            self.mc_nsteps = None
            self.str_params[0] = self.df_str_params.copy()

    def _show_hypparams_simulation(self):
        """
        Processes and manages the hyperparameter simulation inteface.
        """
        # Process Monte Carlo Simulation
        if self.showplt12 == 1:
            self.showplt12 = 2
            print(
                f"\n{Fore.LIGHTYELLOW_EX}STARTING PARAMETERS SIMULATION...{Fore.RESET}"
            )
            self.button_5.setEnabled(False)
            self.sim_label.show()
            # MC random
            np.random.seed(self.seed)

            # MainWindow 2
            self.win_3 = QtWidgets.QMainWindow()
            self.win_3.setWindowTitle(BAR_TITLE)
            self.win_3.setGeometry(200, 240, 1300, 700)
            self.win_3.setStyleSheet("background-color: #292929;")

            # Set widgets layouts
            self.sim_plot_widget_0 = pg.GraphicsLayoutWidget()
            self.sim_plot_widget_1 = pg.GraphicsLayoutWidget()
            self.sim_plot_widget_2 = pg.GraphicsLayoutWidget()

            # Fix widget_0 height
            self.sim_plot_widget_0.setFixedHeight(40)

            # Set widgets bg-color
            self.sim_plot_widget_0.setBackground(background="#292929")
            self.sim_plot_widget_1.setBackground(background="#292929")
            self.sim_plot_widget_2.setBackground(background="#292929")

            # Set QSplitter
            self.sim_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

            # Add widgets in splitter
            self.sim_splitter.addWidget(self.sim_plot_widget_0)
            self.sim_splitter.addWidget(self.sim_plot_widget_1)
            self.sim_splitter.addWidget(self.sim_plot_widget_2)

            # Splitter settings
            self.sim_splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: black;
                    height: 1px; 
                }
            """)

            # Set shared var axissim
            value_var_axissim = Value("d")
            value_var_axissim.value = 0

            datetimeaxis_sim_1 = DatetimeAxisX3(
                orientation="bottom",
                data=self.df_1.time,
                value_var_hist_axis=value_var_axissim,
                value_var_time_axis=self.value_var_time_axis,
            )
            datetimeaxis_sim_2 = DatetimeAxisX3(
                orientation="bottom",
                data=self.df_1.time,
                value_var_hist_axis=value_var_axissim,
                value_var_time_axis=self.value_var_time_axis,
            )
            self.sim_plt_1 = self.sim_plot_widget_1.addPlot(
                axisItems={"bottom": datetimeaxis_sim_1}
            )
            self.sim_plt_2 = self.sim_plot_widget_2.addPlot(
                axisItems={"bottom": datetimeaxis_sim_2}
            )
            self.sim_plt_1.setXLink(self.sim_plt_2)
            self.sim_plt_1.getAxis("bottom").setStyle(showValues=False)

            # Set splitter top margin
            self.sim_splitter.setContentsMargins(0, 0, 0, 0)

            # Centralize splitter in win_2
            self.win_3.setCentralWidget(self.sim_splitter)

            # Set initial config
            self.sim_plot_widget_0.hide()
            self.sim_plot_widget_1.show()
            self.sim_plot_widget_2.show()

            # Set sizes
            self.sim_splitter.setSizes([50, 150, 300])

            plots = [
                self.sim_plt_1,
                self.sim_plt_2,
            ]

            # Set SIM Plots
            for plot in plots:
                # Font Tick
                plot.getAxis("right").setTickFont(self.font_axis)
                plot.getAxis("bottom").setTickFont(self.font_axis)

                # Color Tick
                plot.getAxis("right").setTextPen("#C0C0C0")
                plot.getAxis("bottom").setTextPen("#C0C0C0")

                # Config Grid
                plot.showGrid(x=True, y=True, alpha=0.2)

                # Config Axis
                plot.showAxis("right")
                plot.showAxis("left")
                plot.showAxis("top")
                plot.getAxis("left").setStyle(showValues=False)
                plot.getAxis("top").setStyle(showValues=False)
                plot.getAxis("right").setWidth(50)

                # Config Frame
                plot.getAxis("bottom").setPen(pg.mkPen(color="#606060", width=1))
                plot.getAxis("right").setPen(pg.mkPen(color="#606060", width=1))
                plot.getAxis("top").setPen(pg.mkPen(color="#606060", width=1))
                plot.getAxis("left").setPen(pg.mkPen(color="#606060", width=1))

                # Disable Auto Range
                plot.disableAutoRange()

            self.sim_plt_1.setTitle("Simulated Drawdowns")
            self.sim_plt_2.setTitle("Simulated EquiryCurves")
            self.sim_plt_2.addLegend(pen="#606060", brush="#292929")

            if self.sim_method == "grid":
                values = [self.sim_params[key] for key in self.sim_params]
                params_combinations = list(itertools.product(*values))
            elif self.sim_method == "random":
                values = [self.sim_params[key] for key in self.sim_params]
                all_combinations = list(itertools.product(*values))
                if self.sim_nrandsims is None:
                    self.sim_nrandsims == all_combinations
                    params_combinations = all_combinations
                else:
                    replace_var = (
                        False if self.sim_nrandsims <= len(all_combinations) else True
                    )
                    indices = np.random.choice(
                        len(all_combinations),
                        size=self.sim_nrandsims,
                        replace=replace_var,
                    )
                    params_combinations = [all_combinations[i] for i in indices]
            elif self.sim_method == "bayesian-opt":
                zeros_matrix = np.zeros(
                    (self.sim_bayesopt_ncalls, len(self.sim_bayesopt_spaces))
                )
                params_combinations = [tuple(row) for row in zeros_matrix]

            c_var = "CURVES" if len(params_combinations) > 1 else "CURVE"
            self.sim_label.setText(
                f"{self.sim_method.upper()} SIMULATION - PROCESSING {len(params_combinations)} {c_var}..."
            )

            # Set shared var
            self.value_var_sim = Value("d")
            self.value_var_sim.value = 0

            # Create temp files
            temp_file_3 = tempfile.NamedTemporaryFile(delete=True)
            filename_3 = temp_file_3.name
            temp_file_4 = tempfile.NamedTemporaryFile(delete=True)
            filename_4 = temp_file_4.name

            # Map memory shared
            self.np_mem_3 = np.memmap(
                filename_3,
                dtype="float",
                mode="w+",
                shape=(len(params_combinations), len(self.df_1)),
            )
            self.np_mem_4 = np.memmap(
                filename_4,
                dtype="float",
                mode="w+",
                shape=(len(params_combinations), len(self.df_1)),
            )
            # Params queue
            self.sim_params_queue = mp.Queue()

            # Clip sim_max_curves for grid
            if self.sim_nbest > len(params_combinations):
                self.sim_nbest = len(params_combinations)

            # Parallel (Process) | Concurrent (Thread) Simulations
            if self.sim_taskmode == "thread":
                # Set shared var
                self.value_stopthread_sig = Value("d")
                self.value_stopthread_sig.value = 0
                self.simWorker = ThreadHypSimulations(
                    self.seed,
                    self.df_1,
                    self.sim_method,
                    self.sim_params,
                    self.sim_nbest,
                    self.sim_nrandsims,
                    self.strategy,
                    self.initial_pos,
                    self.returns,
                    self.opers_fee,
                    apply_tax,
                    get_drawdowns,
                    self.np_mem_3,
                    self.np_mem_4,
                    self.value_var_sim,
                    params_combinations,
                    self.sim_bayesopt_ncalls,
                    self.sim_bayesopt_spaces,
                    self.sim_bayesopt_kwargs,
                    self.sim_params_queue,
                    self.value_stopthread_sig,
                )
                self.simWorker.start()

            else:
                self.simWorker = ProcessHypSimulations(
                    self.seed,
                    self.df_1,
                    self.sim_method,
                    self.sim_params,
                    self.sim_nbest,
                    self.sim_nrandsims,
                    self.strategy,
                    self.initial_pos,
                    self.returns,
                    self.opers_fee,
                    apply_tax,
                    get_drawdowns,
                    self.np_mem_3,
                    self.np_mem_4,
                    self.value_var_sim,
                    params_combinations,
                    self.sim_bayesopt_ncalls,
                    self.sim_bayesopt_spaces,
                    self.sim_bayesopt_kwargs,
                    self.sim_params_queue,
                )
                self.simWorker.start()

            # Improve performance
            self.sim_plt_1.setClipToView(True)
            self.sim_plt_2.setClipToView(True)
            #
            self.sim_plt_1.setDownsampling(auto=True, mode="peak", ds=2)
            self.sim_plt_2.setDownsampling(auto=True, mode="peak", ds=2)

            # Set colors
            colors = [
                "#{:02x}{:02x}{:02x}".format(
                    np.random.randint(80, 255),
                    np.random.randint(80, 255),
                    np.random.randint(80, 255),
                )
                for _ in range(len(params_combinations))
            ]

            def update():
                if self.value_var_sim.value == 1:
                    self.value_var_sim.value = 0

                    self.sim_label.setText(
                        f"{self.sim_method.upper()} SIMULATION - PLOTTING {self.sim_nbest} BEST {c_var} OF {len(params_combinations)}"
                    )

                    lines_1 = []
                    lines_2 = []
                    lines_visible = [True] * len(lines_1)
                    selected_indices = set()

                    def get_mouse_press_event(index):
                        def mousePressEvent(event):
                            modifiers = QApplication.keyboardModifiers()
                            if modifiers == QtCore.Qt.ShiftModifier:
                                # If Shift is pressed, toggle the visibility of the clicked line
                                if index in selected_indices:
                                    selected_indices.remove(index)
                                    lines_1[index].setVisible(False)
                                    lines_2[index].setVisible(False)
                                else:
                                    selected_indices.add(index)
                                    lines_1[index].setVisible(True)
                                    lines_2[index].setVisible(True)
                            else:
                                if all(lines_visible):
                                    # Show only the selected line
                                    selected_indices.clear()
                                    selected_indices.add(index)
                                    for i, line in enumerate(lines_1):
                                        line.setVisible(i == index)
                                    for i, line in enumerate(lines_2):
                                        line.setVisible(i == index)
                                    lines_visible[:] = [
                                        i == index for i in range(len(lines_1))
                                    ]
                                else:
                                    # Show all lines
                                    selected_indices.clear()
                                    for line in lines_1:
                                        line.setVisible(True)
                                    for line in lines_2:
                                        line.setVisible(True)
                                    lines_visible[:] = [True] * len(lines_1)
                            legend_2.hide()
                            legend_2.show()

                        return mousePressEvent

                    equity_curves_sim = np.array(self.np_mem_3[:])
                    ddowns_sim = np.array(self.np_mem_4[:])
                    params_sim = self.sim_params_queue.get()

                    # Set data
                    data = [
                        (equity_curves_sim[i], ddowns_sim[i], params_sim[i])
                        for i in range(equity_curves_sim.shape[0])
                    ]
                    # Sort data by EC max
                    data_sorted = sorted(data, key=lambda x: x[1][-1], reverse=True)

                    sim_nbest_logs = []
                    for i in range(self.sim_nbest):
                        drawdowns = data_sorted[i][0]
                        cumul_gains_str = data_sorted[i][1]
                        params = data_sorted[i][2]
                        signal = "+" if cumul_gains_str[-1] > 0 else ""
                        # Drawndowns
                        lines1 = self.sim_plt_1.plot(
                            self.df_1.dateindex,
                            drawdowns,
                            pen={"color": colors[i] + "85", "width": 1.0},
                            name=f"L{i + 1}: {params} DD: {round(np.min(drawdowns), 1)}%",
                            fillBrush=colors[i] + "50",
                            fillLevel=0,
                        )
                        # Cumative Equity Curves
                        lines2 = self.sim_plt_2.plot(
                            self.df_1.dateindex,
                            cumul_gains_str,
                            pen={"color": colors[i] + "97", "width": 2.0},
                            name=f"L{i + 1}_{params}__EC:{signal}{round(cumul_gains_str[-1], 1)}__DD:{round(np.min(drawdowns), 1)}%",
                            # fillBrush=colors[i] + "20",
                            fillLevel=0,
                        )
                        lines_1.append(lines1)
                        lines_2.append(lines2)
                        # Logs
                        text_log = (
                            f"L{i + 1}__Params:{list(params)}__EquityCurve:[{signal}"
                            f"{round(cumul_gains_str[-1], 1)}]__Drawndown:[{round(np.min(drawdowns), 1)}%]"
                        )
                        sim_nbest_logs.append(text_log)

                    # Save logs
                    if not os.path.exists("./Sim_logs"):
                        os.makedirs("./Sim_logs")
                    with open("./Sim_logs/sim_nbest_logs.txt", "w") as f:
                        for item in sim_nbest_logs:
                            f.write("".join(map(str, item)) + "\n")

                    # Set legend sim_plt_2
                    legend_2 = self.sim_plt_2.addLegend()
                    legend_2.anchor((0, 0), (0, 0), offset=(5, 5))
                    legend_2.setColumnCount(1)
                    legend_2.setContentsMargins(0, 0, 0, 0)
                    for i, (_, label) in enumerate(legend_2.items):
                        label.setText(
                            f"<span style='color: {colors[i]}';>{label.text}</span>"
                        )
                        label.mousePressEvent = get_mouse_press_event(i)

                    self.sim_plt_1.enableAutoRange(x=True, y=True)
                    self.sim_plt_2.enableAutoRange(x=True, y=True)

                    if self.simWorker.is_alive():
                        if self.sim_taskmode == "thread":
                            self.simWorker.join()
                        else:
                            self.simWorker.terminate()
                            self.simWorker.join()
                    self.win_3.show()
                    self.timer_plot_sim.stop()

            self.timer_plot_sim = QtCore.QTimer()
            self.timer_plot_sim.timeout.connect(update)
            self.timer_plot_sim.start()

        else:
            self.sim_plt_1.clear()
            self.sim_plt_2.clear()
            self.sim_label.hide()
            if self.timer_plot_sim.isActive():
                self.timer_plot_sim.stop()
                self.timer_plot_sim.deleteLater()
            self.showplt12 = 1
            self.win_3.close()
            if self.simWorker.is_alive():
                if self.sim_taskmode == "thread":
                    self.value_stopthread_sig.value = 1
                else:
                    self.simWorker.terminate()
            if self.str_params is not None and self.strategy is not None:
                self.button_5.setEnabled(True)
                self.combobox_mc.setEnabled(True)
            del self.np_mem_3, self.np_mem_4

    def _process_scatter(self):
        """
        Set and process all scatters symbols.
        """
        # Optimizations
        dynamicRangeLimit = None
        downfactor = 10
        downmethod = "subsample"

        sig_ref = self.df_1.c
        size_1 = 9
        size_2 = 8
        size_3 = 5

        if self.showplt5 == 2 and self.showplt4 == 1:
            self.showplt5 = 1
            sig_ref = self.equity_curve_pred
            size_1 = 4.5
            size_2 = 4.5
            size_3 = 3.5

        elif self.showplt5 == 2 and self.showplt4 == 2:
            self.showplt5 = 1
            sig_ref = self.df_1.cumul_gains_str
            size_1 = 4.5
            size_2 = 4.5
            size_3 = 3.5

        elif self.showplt5 == 2 and self.showplt4 == 3:
            self.showplt5 = 1
            sig_ref = self.df_1.cumul_gains_reappl_str
            size_1 = 4.5
            size_2 = 4.5
            size_3 = 3.5

        # Scatter Short Plus y_true
        if len(self.df_1.dateindex[self.df_1.signals_true_scatter == -3].values) == 1:
            self.scatter_short_true_plus = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == -3].values,
                sig_ref[self.df_1.signals_true_scatter == -3].values,
                pen=pg.mkPen(color="#E60000"),
                brush=pg.mkBrush("#E60000"),
                symbol="t",
                size=size_3,
            )
        else:
            self.scatter_short_true_plus = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == -3].values,
                sig_ref[self.df_1.signals_true_scatter == -3].values,
                pen=None,
                symbol="t",
                symbolSize=size_3,
                symbolBrush="#E60000",
                symbolPen="#E60000",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter Short Minus y_true
        if len(self.df_1.dateindex[self.df_1.signals_true_scatter == -5].values) == 1:
            self.scatter_short_true_minus = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == -5].values,
                sig_ref[self.df_1.signals_true_scatter == -5].values,
                pen=pg.mkPen(color="#E60000"),
                brush=pg.mkBrush("#E60000"),
                symbol="t1",
                size=size_3,
            )
        else:
            self.scatter_short_true_minus = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == -5].values,
                sig_ref[self.df_1.signals_true_scatter == -5].values,
                pen=None,
                symbol="t1",
                symbolSize=size_3,
                symbolBrush="#E60000",
                symbolPen="#E60000",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter Long Plus y_true
        if len(self.df_1.dateindex[self.df_1.signals_true_scatter == 8].values) == 1:
            self.scatter_long_true_plus = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == 8].values,
                sig_ref[self.df_1.signals_true_scatter == 8].values,
                pen=pg.mkPen(color="#04E104"),
                brush=pg.mkBrush("#04E104"),
                symbol="t1",
                size=size_3,
            )
        else:
            self.scatter_long_true_plus = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == 8].values,
                sig_ref[self.df_1.signals_true_scatter == 8].values,
                pen=None,
                symbol="t1",
                symbolSize=size_3,
                symbolBrush="#04E104",
                symbolPen="#04E104",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter Long Minus y_true
        if len(self.df_1.dateindex[self.df_1.signals_true_scatter == 6].values) == 1:
            self.scatter_long_true_minus = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == 6].values,
                sig_ref[self.df_1.signals_true_scatter == 6].values,
                pen=pg.mkPen(color="#04E104"),
                brush=pg.mkBrush("#04E104"),
                symbol="t",
                size=size_3,
            )
        else:
            self.scatter_long_true_minus = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == 6].values,
                sig_ref[self.df_1.signals_true_scatter == 6].values,
                pen=None,
                symbol="t",
                symbolSize=size_3,
                symbolBrush="#04E104",
                symbolPen="#04E104",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter Short Plus y_pred
        if len(self.df_1.dateindex[self.df_1.signals_pred_scatter == -3].values) == 1:
            self.scatter_short_pred_plus = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == -3].values,
                sig_ref[self.df_1.signals_pred_scatter == -3].values,
                pen=pg.mkPen(color="#E60000"),
                brush=pg.mkBrush("#E60000"),
                symbol="t",
                size=size_3,
            )
        else:
            self.scatter_short_pred_plus = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == -3].values,
                sig_ref[self.df_1.signals_pred_scatter == -3].values,
                pen=None,
                symbol="t",
                symbolSize=size_3,
                symbolBrush="#E60000",
                symbolPen="#E60000",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter Short Minus y_pred
        if len(self.df_1.dateindex[self.df_1.signals_pred_scatter == -5].values) == 1:
            self.scatter_short_pred_minus = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == -5].values,
                sig_ref[self.df_1.signals_pred_scatter == -5].values,
                pen=pg.mkPen(color="#E60000"),
                brush=pg.mkBrush("#E60000"),
                symbol="t1",
                size=size_3,
            )
        else:
            self.scatter_short_pred_minus = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == -5].values,
                sig_ref[self.df_1.signals_pred_scatter == -5].values,
                pen=None,
                symbol="t1",
                symbolSize=size_3,
                symbolBrush="#E60000",
                symbolPen="#E60000",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter Long Plus y_pred
        if len(self.df_1.dateindex[self.df_1.signals_pred_scatter == 8].values) == 1:
            self.scatter_long_pred_plus = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == 8].values,
                sig_ref[self.df_1.signals_pred_scatter == 8].values,
                pen=pg.mkPen(color="#04E104"),
                brush=pg.mkBrush("#04E104"),
                symbol="t1",
                size=size_3,
            )
        else:
            self.scatter_long_pred_plus = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == 8].values,
                sig_ref[self.df_1.signals_pred_scatter == 8].values,
                pen=None,
                symbol="t1",
                symbolSize=size_3,
                symbolBrush="#04E104",
                symbolPen="#04E104",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter Long Minus y_pred
        if len(self.df_1.dateindex[self.df_1.signals_pred_scatter == 6].values) == 1:
            self.scatter_long_pred_minus = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == 6].values,
                sig_ref[self.df_1.signals_pred_scatter == 6].values,
                pen=pg.mkPen(color="#04E104"),
                brush=pg.mkBrush("#04E104"),
                symbol="t",
                size=size_3,
            )
        else:
            self.scatter_long_pred_minus = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == 6].values,
                sig_ref[self.df_1.signals_pred_scatter == 6].values,
                pen=None,
                symbol="t",
                symbolSize=size_3,
                symbolBrush="#04E104",
                symbolPen="#04E104",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter1 Long y_true
        if len(self.df_1.dateindex[self.df_1.signals_true_scatter == 1].values) == 1:
            self.scatter_long_true = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == 1].values,
                sig_ref[self.df_1.signals_true_scatter == 1].values,
                pen=pg.mkPen(color="#00FF7F"),
                brush=pg.mkBrush("#008B00"),
                symbol="t1",
                size=size_1,
            )
        else:
            self.scatter_long_true = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == 1].values,
                sig_ref[self.df_1.signals_true_scatter == 1].values,
                pen=None,
                symbol="t1",
                symbolSize=size_1,
                symbolBrush="#008B00",
                symbolPen="#00FF7F",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter1 Short y_true
        if len(self.df_1.dateindex[self.df_1.signals_true_scatter == -1].values) == 1:
            self.scatter_short_true = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == -1].values,
                sig_ref[self.df_1.signals_true_scatter == -1].values,
                pen=pg.mkPen(color="#FF0000"),
                brush=pg.mkBrush("#8B0000"),
                symbol="t",
                size=size_1,
            )
        else:
            self.scatter_short_true = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == -1].values,
                sig_ref[self.df_1.signals_true_scatter == -1].values,
                pen=None,
                symbol="t",
                symbolSize=size_1,
                symbolBrush="#8B0000",
                symbolPen="#FF0000",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter2 Long y_pred
        if len(self.df_1.dateindex[self.df_1.signals_pred_scatter == 1].values) == 1:
            self.scatter_long_pred = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == 1].values,
                sig_ref[self.df_1.signals_pred_scatter == 1].values,
                pen=pg.mkPen(color="#00FF7F"),
                brush=pg.mkBrush("#008B00"),
                symbol="t1",
                size=size_1,
            )
        else:
            self.scatter_long_pred = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == 1].values,
                sig_ref[self.df_1.signals_pred_scatter == 1].values,
                pen=None,
                symbol="t1",
                symbolSize=size_1,
                symbolBrush="#008B00",
                symbolPen="#00FF7F",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter2 Short y_pred
        if len(self.df_1.dateindex[self.df_1.signals_pred_scatter == -1].values) == 1:
            self.scatter_short_pred = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == -1].values,
                sig_ref[self.df_1.signals_pred_scatter == -1].values,
                pen=pg.mkPen(color="#FF0000"),
                brush=pg.mkBrush("#8B0000"),
                symbol="t",
                size=size_1,
            )
        else:
            self.scatter_short_pred = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == -1].values,
                sig_ref[self.df_1.signals_pred_scatter == -1].values,
                pen=None,
                symbol="t",
                symbolSize=size_1,
                symbolBrush="#8B0000",
                symbolPen="#FF0000",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter1 Exit y_true Long
        if len(self.df_1.dateindex[self.df_1.signals_true_scatter == -2].values) == 1:
            self.scatter_exit_true_long = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == -2].values,
                sig_ref[self.df_1.signals_true_scatter == -2].values,
                pen=pg.mkPen(color="#FF0000"),
                brush=pg.mkBrush("black"),
                symbol="t",
                size=size_1,
            )
        else:
            self.scatter_exit_true_long = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == -2].values,
                sig_ref[self.df_1.signals_true_scatter == -2].values,
                pen=None,
                symbol="t",
                symbolSize=size_1,
                symbolBrush="black",
                symbolPen="#FF0000",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter1 Exit y_true Short
        if len(self.df_1.dateindex[self.df_1.signals_true_scatter == 2].values) == 1:
            self.scatter_exit_true_short = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == 2].values,
                sig_ref[self.df_1.signals_true_scatter == 2].values,
                pen=pg.mkPen(color="#00FF7F"),
                brush=pg.mkBrush("black"),
                symbol="t1",
                size=size_1,
            )
        else:
            self.scatter_exit_true_short = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == 2].values,
                sig_ref[self.df_1.signals_true_scatter == 2].values,
                pen=None,
                symbol="t1",
                symbolSize=size_1,
                symbolBrush="black",
                symbolPen="#00FF7F",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter1 Exit y_pred Long
        if len(self.df_1.dateindex[self.df_1.signals_pred_scatter == -2].values) == 1:
            self.scatter_exit_pred_long = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == -2].values,
                sig_ref[self.df_1.signals_pred_scatter == -2].values,
                pen=pg.mkPen(color="#FF0000"),
                brush=pg.mkBrush("black"),
                symbol="t",
                size=size_1,
            )
        else:
            self.scatter_exit_pred_long = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == -2].values,
                sig_ref[self.df_1.signals_pred_scatter == -2].values,
                pen=None,
                symbol="t",
                symbolSize=size_1,
                symbolBrush="black",
                symbolPen="#FF0000",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter1 Exit y_pred Short
        if len(self.df_1.dateindex[self.df_1.signals_pred_scatter == 2].values) == 1:
            self.scatter_exit_pred_short = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == 2].values,
                sig_ref[self.df_1.signals_pred_scatter == 2].values,
                pen=pg.mkPen(color="#00FF7F"),
                brush=pg.mkBrush("black"),
                symbol="t1",
                size=size_1,
            )
        else:
            self.scatter_exit_pred_short = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == 2].values,
                sig_ref[self.df_1.signals_pred_scatter == 2].values,
                pen=None,
                symbol="t1",
                symbolSize=size_1,
                symbolBrush="black",
                symbolPen="#00FF7F",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter exit-gain y_true
        if len(self.df_1.dateindex[self.df_1.signals_true_scatter == 4].values) == 1:
            self.scatter_exit_gain_true = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == 4].values,
                sig_ref[self.df_1.signals_true_scatter == 4].values,
                pen=pg.mkPen(color="#FFFF00"),
                brush=pg.mkBrush("#FFFF00"),
                symbol="+",
                size=size_2,
            )
        else:
            self.scatter_exit_gain_true = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == 4].values,
                sig_ref[self.df_1.signals_true_scatter == 4].values,
                pen=None,
                symbol="+",
                symbolSize=size_2,
                symbolBrush="#FFFF00",
                symbolPen="#FFFF00",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter exit-stop y_true
        if len(self.df_1.dateindex[self.df_1.signals_true_scatter == -4].values) == 1:
            self.scatter_exit_stop_true = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == -4].values,
                sig_ref[self.df_1.signals_true_scatter == -4].values,
                pen=pg.mkPen(color="#FF00FF"),
                brush=pg.mkBrush("#FF00FF"),
                symbol="x",
                size=size_2,
            )
        else:
            self.scatter_exit_stop_true = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_true_scatter == -4].values,
                sig_ref[self.df_1.signals_true_scatter == -4].values,
                pen=None,
                symbol="x",
                symbolSize=size_2,
                symbolBrush="#FF00FF",
                symbolPen="#FF00FF",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter exit-gain y_pred
        if len(self.df_1.dateindex[self.df_1.signals_pred_scatter == 4].values) == 1:
            self.scatter_exit_gain_pred = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == 4].values,
                sig_ref[self.df_1.signals_pred_scatter == 4].values,
                pen=pg.mkPen(color="#FFFF00"),
                brush=pg.mkBrush("#FFFF00"),
                symbol="+",
                size=size_2,
            )
        else:
            self.scatter_exit_gain_pred = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == 4].values,
                sig_ref[self.df_1.signals_pred_scatter == 4].values,
                pen=None,
                symbol="+",
                symbolSize=size_2,
                symbolBrush="#FFFF00",
                symbolPen="#FFFF00",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # Scatter exit-stop y_pred
        if len(self.df_1.dateindex[self.df_1.signals_pred_scatter == -4].values) == 1:
            self.scatter_exit_stop_pred = pg.ScatterPlotItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == -4].values,
                sig_ref[self.df_1.signals_pred_scatter == -4].values,
                pen=pg.mkPen(color="#FF00FF"),
                brush=pg.mkBrush("#FF00FF"),
                symbol="x",
                size=size_2,
            )
        else:
            self.scatter_exit_stop_pred = pg.PlotDataItem(
                self.df_1.dateindex[self.df_1.signals_pred_scatter == -4].values,
                sig_ref[self.df_1.signals_pred_scatter == -4].values,
                pen=None,
                symbol="x",
                symbolSize=size_2,
                symbolBrush="#FF00FF",
                symbolPen="#FF00FF",
                pxMode=True,
                antialias=False,
                downsampleMethod=downmethod,
                downsample=downfactor,
                autoDownsample=True,
                dynamicRangeLimit=dynamicRangeLimit,
            )

        # True
        self.scatter_long_true.setVisible(False)
        self.scatter_short_true.setVisible(False)
        #
        self.scatter_short_true_plus.setVisible(False)
        self.scatter_short_true_minus.setVisible(False)
        self.scatter_long_true_plus.setVisible(False)
        self.scatter_long_true_minus.setVisible(False)
        #
        self.scatter_exit_true_long.setVisible(False)
        self.scatter_exit_true_short.setVisible(False)
        self.scatter_exit_gain_true.setVisible(False)
        self.scatter_exit_stop_true.setVisible(False)

        # Pred
        self.scatter_long_pred.setVisible(False)
        self.scatter_short_pred.setVisible(False)
        #
        self.scatter_short_pred_plus.setVisible(False)
        self.scatter_short_pred_minus.setVisible(False)
        self.scatter_long_pred_plus.setVisible(False)
        self.scatter_long_pred_minus.setVisible(False)
        #
        self.scatter_exit_pred_long.setVisible(False)
        self.scatter_exit_pred_short.setVisible(False)
        self.scatter_exit_gain_pred.setVisible(False)
        self.scatter_exit_stop_pred.setVisible(False)

    def _update_mc_mode(self):
        """
        Update attribute mc_mode by select MC test in combobox.
        """
        self.mc_mode = self.combobox_mc.currentText().lower().replace(" ", "_")

    def _simulations_logic(self):
        """
        Manage the simulations on the interface. Enable and disable buttons and labels.
        """
        # Monte Carlo logics
        if self.str_params and self.strategy:
            self.button_5.setEnabled(True)
            self.combobox_mc.setEnabled(True)
        else:
            self.button_5.setEnabled(False)
            self.combobox_mc.setEnabled(False)

        # Hyperparameters simulation logics
        if self.strategy:
            if self.sim_method == "grid" and isinstance(self.sim_params, dict):
                self.button_10.setEnabled(True)
            elif self.sim_method == "random" and isinstance(self.sim_params, dict):
                self.button_10.setEnabled(True)
            elif self.sim_method == "bayesian-opt" and isinstance(
                self.sim_bayesopt_spaces, list
            ):
                self.button_10.setEnabled(True)
            else:
                self.button_10.setEnabled(False)
        else:
            self.button_10.setEnabled(False)

        # Enable | Disable (MC and Hyper simulations)
        if self.showplt12 == 2:
            self.sim_label.show()
            self.mc_label.hide()
            self.button_5.setEnabled(False)
            self.button_10.setEnabled(True)
        if self.showplt6 == 2:
            self.sim_label.hide()
            self.mc_label.show()
            self.button_5.setEnabled(True)
            self.button_10.setEnabled(False)

    def _mouse_events(self):
        """
        Manage the mouse events.
        """

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

        def mouse_evt1(evt: QtCore.QPointF):
            """
            Manage mouse events in widget plt_1.
            
            Args:
                evt (QtCore.QPointF): Mouse coordinate points in plot plt_1.
            """
            self.x_line_plt2.hide()
            self.y_line_plt2.hide()
            self.x_line_plt3.hide()
            self.y_line_plt3.hide()

            mouse_point_1 = self.plt_1.vb.mapSceneToView(evt)
            self.x_line_plt1.setPos(mouse_point_1.x())
            self.x_line_plt2.setPos(mouse_point_1.x())
            self.y_line_plt1.setPos(mouse_point_1.y())
            self.x_line_plt1.show()
            self.y_line_plt1.show()

            self.x_line_plt2.setPos(mouse_point_1.x())
            self.x_line_plt2.show()

            self.x_line_plt3.setPos(mouse_point_1.x())
            self.x_line_plt3.show()

            self.x_line_plt1.label.show()
            self.x_line_plt2.label.hide()
            self.x_line_plt3.label.hide()

            try:
                self.y_line_plt1.label.setFormat(str(round(mouse_point_1.y(), 2)))
                if self.value_var_hist_axis.value == 20:
                    self.x_line_plt2.label.setFormat(str(round(mouse_point_1.x(), 2)))
                elif self.value_var_time_axis.value == 0:
                    self.x_line_plt1.label.setFormat(
                        datetime.datetime.fromtimestamp(
                            int(
                                timestamp_calc(
                                    self.df_1.time, round(mouse_point_1.x(), 5)
                                )
                            )
                            / 1000
                        )
                        .strftime("%-d %b '%-y  %-H:%M")
                        .upper()
                    )
                else:
                    self.x_line_plt1.label.setFormat(str(round(mouse_point_1.x(), 2)))

            except Exception as e:
                exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
                exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
                track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
                print(
                    f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}{Fore.RESET}"
                )
                pass
                # raise sys.exc_info()[0]

        def mouse_evt2(evt):
            """
            Manage mouse events in widget plt_2.
            
            Args:
                evt (QtCore.QPointF): Mouse coordinate points in plot plt_2.
            """
            self.x_line_plt1.hide()
            self.y_line_plt1.hide()
            self.x_line_plt3.hide()
            self.y_line_plt3.hide()

            mouse_point_2 = self.plt_2.vb.mapSceneToView(evt)
            self.x_line_plt2.setPos(mouse_point_2.x())
            self.y_line_plt2.setPos(mouse_point_2.y())
            self.x_line_plt2.show()
            self.y_line_plt2.show()

            self.x_line_plt1.setPos(mouse_point_2.x())
            self.x_line_plt1.show()

            self.x_line_plt3.setPos(mouse_point_2.x())
            self.x_line_plt3.show()

            self.x_line_plt1.label.hide()
            self.x_line_plt2.label.show()
            self.x_line_plt3.label.hide()

            try:
                self.y_line_plt2.label.setFormat(str(round(mouse_point_2.y(), 2)))
                # Set this for dist plot
                if self.value_var_hist_axis.value == 20:
                    self.x_line_plt2.label.setFormat(str(round(mouse_point_2.x(), 2)))
                elif self.value_var_hist_axis.value == 10:
                    self.x_line_plt2.label.setFormat(
                        str(round(mouse_point_2.x() * 100, 2)) + "%"
                    )
                elif self.value_var_time_axis.value == 0:
                    self.x_line_plt2.label.setFormat(
                        datetime.datetime.fromtimestamp(
                            int(
                                timestamp_calc(
                                    self.df_1.time, round(mouse_point_2.x(), 5)
                                )
                            )
                            / 1000
                        )
                        .strftime("%-d %b '%-y  %-H:%M")
                        .upper()
                    )
                else:
                    self.x_line_plt2.label.setFormat(str(round(mouse_point_2.x(), 2)))

            except Exception as e:
                exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
                exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
                track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
                print(
                    f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}{Fore.RESET}"
                )
                pass
                # raise sys.exc_info()[0]

        def mouse_evt3(evt):
            """
            Manage mouse events in widget plt_3.
            
            Args:
                evt (QtCore.QPointF): Mouse coordinate points in plot plt_3.
            """
            self.x_line_plt1.hide()
            self.y_line_plt1.hide()
            self.x_line_plt2.hide()
            self.y_line_plt2.hide()

            mouse_point_3 = self.plt_3.vb.mapSceneToView(evt)
            self.x_line_plt3.setPos(mouse_point_3.x())
            self.y_line_plt3.setPos(mouse_point_3.y())
            self.x_line_plt3.show()
            self.y_line_plt3.show()

            self.x_line_plt1.setPos(mouse_point_3.x())
            self.x_line_plt1.show()

            self.x_line_plt2.setPos(mouse_point_3.x())
            self.x_line_plt2.show()

            self.x_line_plt1.label.hide()
            self.x_line_plt2.label.hide()
            self.x_line_plt3.label.show()

            try:
                self.y_line_plt3.label.setFormat(str(round(mouse_point_3.y(), 2)))
                if self.value_var_hist_axis.value == 20:
                    self.x_line_plt2.label.setFormat(str(round(mouse_point_3.x(), 2)))
                    self.x_line_plt3.label.setFormat(str(round(mouse_point_3.x(), 2)))
                elif self.value_var_time_axis.value == 0:
                    self.x_line_plt3.label.setFormat(
                        datetime.datetime.fromtimestamp(
                            int(
                                timestamp_calc(
                                    self.df_1.time, round(mouse_point_3.x(), 5)
                                )
                            )
                            / 1000
                        )
                        .strftime("%-d %b '%-y  %-H:%M")
                        .upper()
                    )
                else:
                    self.x_line_plt3.label.setFormat(str(round(mouse_point_3.x(), 2)))
            except Exception as e:
                exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
                exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
                track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
                print(
                    f"{Fore.LIGHTRED_EX}{exception_type}{exception_message}{track_line}{Fore.RESET}"
                )
                pass
                # raise sys.exc_info()[0]

        self.plot_widget_1.scene().sigMouseMoved.connect(mouse_evt1)
        self.plot_widget_2.scene().sigMouseMoved.connect(mouse_evt2)
        self.plot_widget_3.scene().sigMouseMoved.connect(mouse_evt3)

    def _set_crosshair(self):
        """
        Set the infinitelines for crosshair.
        """
        # Infiniteline PLT1
        self.x_line_plt1 = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen={"color": "#969696", "width": 0.4},
            label="{value:0.1f}",
            labelOpts={
                "position": 0.90,
                "color": "#D3D3D3",
                "movable": True,
                "fill": "#2F4F4F",
            },
        )
        self.y_line_plt1 = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen={"color": "#969696", "width": 0.4},
            label="{value:0.1f}",
            labelOpts={
                "position": 0.03,
                "color": "#D3D3D3",
                "movable": True,
                "fill": "#2F4F4F",
            },
        )

        # Infiniteline PLT2
        self.x_line_plt2 = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen={"color": "#969696", "width": 0.4},
            label="{value:0.1f}",
            labelOpts={
                "position": 0.95,
                "color": "#D3D3D3",
                "movable": True,
                "fill": "#2F4F4F",
            },
        )
        self.y_line_plt2 = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen={"color": "#969696", "width": 0.4},
            label="{value:0.1f}",
            labelOpts={
                "position": 0.03,
                "color": "#D3D3D3",
                "movable": True,
                "fill": "#2F4F4F",
            },
        )

        # Infiniteline PLT3
        self.x_line_plt3 = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen={"color": "#969696", "width": 0.4},
            label="{value:0.1f}",
            labelOpts={
                "position": 0.92,
                "color": "#D3D3D3",
                "movable": True,
                "fill": "#2F4F4F",
            },
        )
        self.y_line_plt3 = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen={"color": "#969696", "width": 0.4},
            label="{value:0.1f}",
            labelOpts={
                "position": 0.03,
                "color": "#D3D3D3",
                "movable": True,
                "fill": "#2F4F4F",
            },
        )

        # Infiniteline settings
        font = QFont("TypeWriter")
        font.setPixelSize(9)
        font.setWeight(70)
        self.x_line_plt1.label.setFont(font)
        self.y_line_plt1.label.setFont(font)
        self.x_line_plt2.label.setFont(font)
        self.y_line_plt2.label.setFont(font)
        self.x_line_plt3.label.setFont(font)
        self.y_line_plt3.label.setFont(font)
