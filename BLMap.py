#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import os

from PySide2 import QtWidgets, QtUiTools, QtCore

import pyqtgraph as pg

pg.setConfigOption('background', 'w')  # Plothintergrund weiß (2D)
pg.setConfigOption('foreground', 'k')  # Plotvordergrund schwarz (2D)
pg.setConfigOptions(antialias=True)  # Enable antialiasing for prettier plots

# to use pyqtgraph with PySide2, see also:
# https://stackoverflow.com/questions/60580391/pyqtgraph-with-pyside-2-and-qtdesigner
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class UiLoader(QtUiTools.QUiLoader):
    def createWidget(self, className, parent=None, name=""):
        if className == "PlotWidget":
            return pg.PlotWidget(parent=parent)
        return super().createWidget(className, parent, name)


def load_ui(fname):
    fd = QtCore.QFile(fname)
    if fd.open(QtCore.QFile.ReadOnly):
        loader = UiLoader()
        window = loader.load(fd)
        fd.close()
        return window


class Map(QtCore.QObject):

    def __init__(self):
        super(Map, self).__init__()
        self.window = load_ui(os.path.join(DIR_PATH, 'BLMap.ui'))

        # Achsenbeschriftung
        self.window.map.setLabel('bottom', text='distance from source / mm')  # X-Achsenname
        self.window.map.setLabel('left', text='vertical beamoffset / mm')  # Y-Achsenname

        # die optische Achse
        self.window.map.plot((17000, 34500), (0, 0), pen=pg.mkPen('k', style=QtCore.Qt.DashLine))

        # BL-Komponenten
        # Blenden 1 (Position auf +- ~100mm geschätzt)
        self.window.map.plot((17800, 17800), (1, 11), pen=pg.mkPen('b'))  # obere Backe
        self.window.map.plot((17800, 17800), (-1, -11), pen=pg.mkPen('b'))  # untere Backe

        # Filter 1 (Position auf +- ~100mm geschätzt)
        self.window.map.plot((18060, 18060), (-5, 5), pen=pg.mkPen('r'))

        # Filter 2 (Position auf +- ~100mm geschätzt)
        self.window.map.plot((18160, 18160), (-5, 5), pen=pg.mkPen('r'))

        # Drahtmonitor M1
        self.window.map.plot((18487, 18487), (-15.5, -14.5), pen=pg.mkPen('r'))

        # DMM mirror 1
        self.window.map.plot((19239, 19559), (0, 0), pen=pg.mkPen('r'))

        # DMM mirror 2
        self.window.map.plot((19809, 20189), (10, 10), pen=pg.mkPen('r'))

        # Drahtmonitor M2
        self.window.map.plot((25542, 25542), (-15.5, -14.5), pen=pg.mkPen('r'))

        # DCM crystal 1
        self.window.map.plot((26750, 26850), (0, 0), pen=pg.mkPen('r'))

        # DCM crystal 2
        self.window.map.plot((26950, 27050), (10, 10), pen=pg.mkPen('r'))

        # Beamstop
        self.window.map.plot((27738, 27738), (-10, 0), pen=pg.mkPen('r', width=4.5))

        # Fluoreszenzschirm M4
        self.window.map.plot((28091, 28091), (-10, 0), pen=pg.mkPen('r'))

        # Blenden 2 (Position auf +- ~100mm geschätzt)
        self.window.map.plot((29950, 29950), (2, 12), pen=pg.mkPen('b'))  # obere Backe
        self.window.map.plot((29950, 29950), (-2, -12), pen=pg.mkPen('b'))  # untere Backe

        # Drahtmonitor M5 (vertical)
        self.window.map.plot((30330, 30330), (-15.5, -14.5), pen=pg.mkPen('r'))

        # Window
        self.window.map.plot((34000, 34000), (-12.5, 12.5), pen=pg.mkPen('r'))

        # Blenden 3 (Position auf +- ~100mm geschätzt)
        self.window.map.plot((34050, 34050), (3, 13), pen=pg.mkPen('b'))  # obere Backe
        self.window.map.plot((34050, 34050), (-3, -13), pen=pg.mkPen('b'))  # untere Backe

    def show(self):
        self.window.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Map()
    main.show()
    sys.exit(app.exec_())
