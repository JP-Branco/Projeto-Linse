from pathlib import Path
import sys

from PySide2 import QtWidgets

from .mainwindow import mainwindow
from .driverhardware import driverhardware
from .dataman import dataman
from .figures import MyFigQtGraph

app = QtWidgets.QApplication([])
app.setStyle('Fusion')

drv = driverhardware()
dman = dataman(drv)
mfig = MyFigQtGraph(dman)

mwindow = mainwindow(app, drv, dman, mfig)

mwindow.show()

sys.exit(app.exec_())
