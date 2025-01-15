# LabQuant - A visual tool to support the development of algo-strategies in Quantitative Finance - by fab2112
import numpy as np
import pyqtgraph as pg

from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtCore


class CandlestickItem(pg.GraphicsObject):
    """
    This class generate the candlesticks graphicals of the project based on the PyQtGraph module.
    """

    def __init__(self, data: list):
        """
        Initialization method.

        Args:
            data (list): List of tuples (timeindex, o, h, l, c)
        """
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.data = [np.array(tuple_) for tuple_ in self.data]
        self._generate_picture()

    def _generate_picture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = 0.4  # half width of the candlestick bar
        previous_candle_bullish = (
            None  # Track if previous candle was bullish or bearish
        )

        for timestamp, open_, high_, low_, close_ in self.data:
            if any(np.isnan([timestamp, open_, high_, low_, close_])):
                continue  # Skip this data point if any value is NaN

            if open_ == high_ == low_ == close_:
                # Draw a horizontal line if all values are the same
                if previous_candle_bullish is None:
                    # Default color if there is no previous candle
                    p.setPen(pg.mkPen("grey"))
                elif previous_candle_bullish:
                    p.setPen(pg.mkPen("#3399FF"))  # Color of bullish previous candle
                else:
                    p.setPen(pg.mkPen("grey"))  # Color of bearish previous candle
                p.drawLine(
                    QtCore.QPointF(timestamp - w, open_),
                    QtCore.QPointF(timestamp + w, open_),
                )
            else:
                if open_ > close_:
                    p.setBrush(pg.mkBrush("#505050"))  # grey for bearish
                    p.setPen(pg.mkPen("grey"))  # white for bearish lines
                    previous_candle_bullish = False
                else:
                    p.setBrush(pg.mkBrush("#0066CC"))  # blue for bullish
                    p.setPen(pg.mkPen("#3399FF"))  # blue for bullish lines
                    previous_candle_bullish = True
                p.drawLine(
                    QtCore.QPointF(timestamp, low_), QtCore.QPointF(timestamp, high_)
                )
                p.drawRect(QtCore.QRectF(timestamp - w, open_, w * 2, close_ - open_))

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())
