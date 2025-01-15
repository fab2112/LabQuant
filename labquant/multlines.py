# LabQuant - A visual tool to support the development of algo-strategies in Quantitative Finance - by fab2112
import numpy as np
import pyqtgraph as pg

from pyqtgraph.Qt.QtGui import QPen, QColor
from pyqtgraph.Qt.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsItemGroup,
)

class MultiLines(QGraphicsPathItem):
    """
    This class generates graphical lines for Monte Carlo tests based on the Qt graphics engine.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Initialization method.

        Args:
            x (numpy array): Represent the axis "x"
            y (numpy array): Represent the axis "y" (equity curves)
        """
        connect = np.ones(x.shape, dtype=bool)
        connect[:, -1] = 0
        self.path = pg.arrayToQPath(x.flatten(), y.flatten(), connect.flatten())
        QGraphicsPathItem.__init__(self, self.path)
        self.setPen(pg.mkPen(color="#B0C4DE", width=0.4))

    def _bounding_rect(self):
        """
        Returns the bounds of the path to render in the scene.
        """
        return self.path.bounding_rect()


class MultiColorLines(QGraphicsItemGroup):
    """
    This class generates graphical color lines for Monte Carlo tests based on the Qt graphics engine.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, line_width: float = 0.6):
        """
        Initialization method.

        Args:
            x (numpy array): Represent the axis "x".
            y (numpy array): Represent the axis "y" (equity curves).
            line_width (float): The width of the line.
        """
        super().__init__()
        for i in range(x.shape[0]):
            connect = np.ones(x[i].shape, dtype=bool)
            connect[-1] = 0
            path = pg.arrayToQPath(x[i], y[i], connect)
            color = self._random_color()
            pen = QPen(QColor(color), line_width)
            pen.setCosmetic(True)

            path_item = QGraphicsPathItem(path)
            path_item.setPen(pen)
            self.addToGroup(path_item)

    def _random_color(self):
        """
        Generate the random colors in hex format.
        """
        colors = "#{:06x}".format(np.random.randint(0, 0xFFFFFF))
        return colors

