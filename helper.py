#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import math
import numpy as np
import time

import helper_calc as calc
import evefile as ef                  # only at BAMline

from PySide2 import QtWidgets, QtUiTools, QtCore, QtGui
from PySide2.QtCore import QRunnable, Slot, QThreadPool, QObject, Signal

import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.sources as rs

import pyqtgraph as pg
from epics import caget, caput, camonitor

pg.setConfigOption('background', 'w')  # background color white (2D)
pg.setConfigOption('foreground', 'k')  # foreground color black (2D)
pg.setConfigOptions(antialias=True)  # enable antialiasing for prettier plots

# using pyqtgraph with PySide2, see also:
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


class Worker(QRunnable):
    """
    Worker thread, for more info see:
    https://www.learnpyqt.com/tutorials/multithreading-pyqt-applications-qthreadpool/

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        self.fn(*self.args, **self.kwargs)
        # # Retrieve args/kwargs here; and fire processing using them
        # try:
        #     result = self.fn(*self.args, **self.kwargs)
        # except:
        #     traceback.print_exc()
        #     exctype, value = sys.exc_info()[:2]
        #     self.signals.error.emit((exctype, value, traceback.format_exc()))
        # else:
        #     self.signals.result.emit(result)  # Return the result of the processing
        # finally:
        #     self.signals.finished.emit()  # Done
        self.signals.finished.emit()  # Done


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress or 0/1 fore not done/done

    """
    finished = Signal()
    # error = Signal(tuple)
    # result = Signal(object)
    progress = Signal(int)


class Helper(QtCore.QObject):

    def __init__(self):
        super(Helper, self).__init__()
        self.window = load_ui(os.path.join(DIR_PATH, 'helper.ui'))
        self.window.installEventFilter(self)

        # class variables
        self.flux_xrt_wls = []  # empty flux array at startup
        self.energy_max = 1  # Maximum Energy of the spectrum in keV, determined in spectrum_evaluation

        self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # mouse cursor position
        self.proxy = pg.SignalProxy(self.window.Graph.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

        # plot legends (offset-position in pixel)
        # self.Graph.addLegend(offset=(1100, -600))

        # calculate already at start up
        self.energy = np.linspace(self.window.e_min.value() * 1000, self.window.e_max.value() * 1000,
                                  self.window.e_step.value())
        self.source_spectrum_energy = self.energy
        self.source_calc_thread()  # first plot at startup

        # perform calculations when there was user input
        self.window.calc_fwhm.stateChanged.connect(self.bl_spectrum)
        self.window.fwhm.valueChanged.connect(self.bl_spectrum)
        self.window.line_button.toggled.connect(self.bl_spectrum)

        # change global Energy-Range
        self.window.e_min.valueChanged.connect(self.energy_range)
        self.window.e_max.valueChanged.connect(self.energy_range)
        self.window.e_step.valueChanged.connect(self.energy_range)

        # user input to source parameters
        self.window.hor_mm.valueChanged.connect(self.calc_acceptance)
        self.window.ver_mm.valueChanged.connect(self.calc_acceptance)
        self.window.distance.valueChanged.connect(self.calc_acceptance)
        self.window.calc_source.clicked.connect(self.source_calc_thread)
        self.window.source_in.stateChanged.connect(self.bl_spectrum)

        # user input to filter parameters
        self.window.filter1.currentIndexChanged.connect(self.set_filter_size)
        self.window.filter2.currentIndexChanged.connect(self.set_filter_size)
        self.window.d_filter1.valueChanged.connect(self.bl_spectrum)
        self.window.d_filter2.valueChanged.connect(self.bl_spectrum)

        # user input to dmm parameters
        self.window.dmm_stripe.currentIndexChanged.connect(self.choose_dmm_stripe)
        self.window.dmm_2d.valueChanged.connect(self.new_dmm_parameters)
        self.window.dmm_gamma.valueChanged.connect(self.new_dmm_parameters)
        self.window.layer_pairs.valueChanged.connect(self.bl_spectrum)
        self.window.dmm_slider_theta.valueChanged.connect(self.dmm_slider_theta_conversion)
        self.window.dmm_slider_off.valueChanged.connect(self.dmm_slider_off_conversion)
        self.window.dmm_off.valueChanged.connect(self.bl_spectrum)
        self.window.dmm_theta.valueChanged.connect(self.bl_spectrum)
        self.window.d_top_layer.valueChanged.connect(self.bl_spectrum)
        self.window.dmm_one_ml.stateChanged.connect(self.bl_spectrum)
        self.window.dmm_in.stateChanged.connect(self.bl_spectrum)
        self.window.dmm_off_check.stateChanged.connect(self.bl_spectrum)
        self.window.dmm_with_filters.stateChanged.connect(self.bl_spectrum)

        # user input to dcm parameters
        self.window.dcm_in.stateChanged.connect(self.bl_spectrum)
        self.window.dcm_one_crystal.stateChanged.connect(self.bl_spectrum)
        self.window.dcm_off_check.stateChanged.connect(self.bl_spectrum)
        self.window.dcm_orientation.currentIndexChanged.connect(self.bl_spectrum)
        self.window.dcm_slider_theta.valueChanged.connect(self.dcm_slider_theta_conversion)
        self.window.dcm_theta.valueChanged.connect(self.bl_spectrum)
        self.window.dcm_slider_off.valueChanged.connect(self.dcm_slider_off_conversion)
        self.window.dcm_off.valueChanged.connect(self.bl_spectrum)
        self.window.dcm_harmonics.valueChanged.connect(self.bl_spectrum)

        # status and GoTo parameters
        self.window.get_pos.clicked.connect(self.bl_status)
        self.window.action_button.clicked.connect(self.bl_move)
        self.window.off_ctTable.toggled.connect(self.toggle_expTable_off)
        self.window.off_expTable.toggled.connect(self.toggle_ctTable_off)

        # load h5-File parameters
        self.window.actionLoad_h5.triggered.connect(self.load_h5)

    def view_box(self):

        """pyqtgraph viewbox for testing... without function yet"""

        # linear_region = pg.LinearRegionItem([10, 40],span=(0.5, 1))
        linear_region = pg.LinearRegionItem()
        # linear_region.setZValue(-10)
        self.window.Graph.addItem(linear_region)

    def mouse_moved(self, evt):

        """Update the cursor position text with the mouse coordinates."""

        pos = evt[0]
        mouse_point = self.window.Graph.plotItem.vb.mapSceneToView(pos)
        self.window.cursor_pos.setText("cursor position: x = %0.2f y = %0.2E" % (mouse_point.x(), mouse_point.y()))

    def show(self):

        """Show the main Window."""

        self.window.show()

    def dmm_slider_theta_conversion(self):

        """Converts the slider ticks to int values."""

        self.window.dmm_theta.setValue(self.window.dmm_slider_theta.value() / 1e4)

    def dmm_slider_off_conversion(self):

        """Converts the slider ticks to int values."""

        self.window.dmm_off.setValue(self.window.dmm_slider_off.value() / 1e2)

    def dcm_slider_theta_conversion(self):

        """Converts the slider ticks to int values."""

        self.window.dcm_theta.setValue(self.window.dcm_slider_theta.value() / 1e4)

    def dcm_slider_off_conversion(self):

        """Converts the slider ticks to int values."""

        self.window.dcm_off.setValue(self.window.dcm_slider_off.value() / 1e2)

    def choose_dmm_stripe(self):

        """Sets the original DMM-Stripe parameters."""

        self.window.dmm_2d.setEnabled(1)
        self.window.layer_pairs.setEnabled(1)
        self.window.dmm_gamma.setEnabled(1)

        # gamma: ration of the high absorbing layer (in our case bottom) to the 2D-value
        self.window.dmm_gamma.setValue(0.4)

        if self.window.dmm_stripe.currentText() == 'W / Si':
            self.window.dmm_2d.setValue(6.619)  # 2D-value in nm
            self.window.layer_pairs.setValue(70)
        if self.window.dmm_stripe.currentText() == 'Mo / B4C':
            self.window.dmm_2d.setValue(5.736)  # 2D-value in nm
            self.window.layer_pairs.setValue(180)
        if self.window.dmm_stripe.currentText() == 'Pd':
            self.window.dmm_gamma.setValue(1)
            self.window.dmm_2d.setValue(0)
            self.window.layer_pairs.setValue(0)
            self.window.dmm_2d.setEnabled(0)
            self.window.layer_pairs.setEnabled(0)
            self.window.dmm_gamma.setEnabled(0)

    def new_dmm_parameters(self):

        """Calculate top- and bottom-layer thickness when there was user input."""

        # The original W/Si-multilayer of the BAMline: d(W) / d(W + Si) = 0.4
        # d_W = (6.619 / 2) * 0.4 = 3.3095 * 0.4 = 1.3238 nm
        # d_Si = 3.3095 - 1.3238 = 1.9857 nm
        # 1 nm = 10 angstrom
        d = self.window.dmm_2d.value() * 10 / 2
        d_bottom = d * self.window.dmm_gamma.value()
        d_top = d - d_bottom

        self.window.d_top_layer.setValue(d_top)
        self.window.d_bottom_layer.setValue(d_bottom)

    def calc_acceptance(self):

        """Calculate the angular acceptance at the experiment depending on the source distance."""

        x_prime_max = math.atan(self.window.hor_mm.value() * 1e-3 / (2 * self.window.distance.value())) * 1000
        z_prime_max = math.atan(self.window.ver_mm.value() * 1e-3 / (2 * self.window.distance.value())) * 1000
        self.window.hor_acceptance.setValue(x_prime_max)
        self.window.ver_acceptance.setValue(z_prime_max)

    def set_filter_size(self):

        """Set the original filter thicknesses of the BAMline when users chooses a filter."""

        filter1_text = self.window.filter1.currentText()
        filter2_text = self.window.filter2.currentText()

        if 'none' in filter1_text:
            self.window.d_filter1.setValue(0.)
        if '200' in filter1_text:
            self.window.d_filter1.setValue(200.)
        if '600' in filter1_text:
            self.window.d_filter1.setValue(600.)
        if '1' in filter1_text:
            self.window.d_filter1.setValue(1000.)

        if '50' in filter2_text:
            self.window.d_filter2.setValue(50.)
        if '60' in filter2_text:
            self.window.d_filter2.setValue(60.)
        if '200' in filter2_text:
            self.window.d_filter2.setValue(200.)
        if '500' in filter2_text:
            self.window.d_filter2.setValue(500.)
        if '1' in filter2_text:
            self.window.d_filter2.setValue(1000.)

    def progress_bar(self, done):

        """Use the process bar as an indicator that a separate thread process is running."""

        if not done:
            self.window.progressBar.setRange(0, 0)
        else:
            self.window.progressBar.setRange(0, 1)
            return

    def source_calc_thread(self):

        """Calculate the Source-spectrum in a separate thread in order to not freeze the GUI."""

        worker = Worker(self.xrt_source_wls)  # Any other args, kwargs are passed to the run function
        worker.signals.progress.connect(self.progress_bar)
        worker.signals.finished.connect(self.bl_spectrum)

        # Execute
        self.threadpool.start(worker)

        # self.window.Graph.clear()
        # text = pg.TextItem(text='Calculation of source spectrum is running, please wait.',
        #                    color=(200, 0, 0), anchor=(0.5, 0.5))
        # self.window.Graph.addItem(text)

    def energy_range(self):

        """Calculates the energy range array."""

        self.energy = np.linspace(self.window.e_min.value() * 1000, self.window.e_max.value() * 1000,
                                  self.window.e_step.value())

        # if we don't need a source spectrum the calculation can be done "instantly"
        if self.window.source_in.isChecked() is False:
            self.bl_spectrum()

    def spectrum_evaluation(self, energy_range, spectrum):

        """Evaluate the spectrum: calculate maxima, FWHM, etc. ..."""

        self.energy_max = energy_range[spectrum.argmax()] / 1000

        # FWHM is calculated out of the left- and right-edge (see helper_calc.py module)
        if self.window.calc_fwhm.isChecked() is True:
            left, right, sw = calc.peak_pos(energy_range, spectrum, schwelle=self.window.fwhm.value())
            # if a FWHM calculation is not possible
            if not left or not right or not sw:
                self.window.text_calculations.setText('maximum = %.2E at %.3f keV' % (spectrum.max(), energy_range[
                    spectrum.argmax()] / 1000))
                return
            left_edge = pg.InfiniteLine(movable=False, angle=90, pen=(200, 200, 10), label='left={value:0.3f}',
                                        labelOpts={'position': 0.95, 'color': (200, 200, 10),
                                                   'fill': (200, 200, 200, 50),
                                                   'movable': True})

            left_edge.setPos([left / 1e3, left / 1e3])
            self.window.Graph.addItem(left_edge)

            right_edge = pg.InfiniteLine(movable=False, angle=90, pen=(200, 200, 10), label='right={value:0.3f}',
                                         labelOpts={'position': 0.9, 'color': (200, 200, 10),
                                                    'fill': (200, 200, 200, 50),
                                                    'movable': True})

            right_edge.setPos([right / 1e3, right / 1e3])
            self.window.Graph.addItem(right_edge)

            width = abs(right - left)
            center = 0.5 * (right + left)
            self.window.text_calculations.setText('maximum = %.2E at %.3f keV\nFWHM = %.3f eV\ncenter(FWHM) = %.3f keV'
                                                  % (spectrum.max(), self.energy_max, width, center / 1000))

        else:
            self.window.text_calculations.setText('maximum = %.2E at %.3f keV' % (spectrum.max(), self.energy_max))

    def xrt_source_wls(self, progress_callback):

        """Calculates the source spectrum of a bending magnet."""

        self.window.e_min.setEnabled(0)
        self.window.e_max.setEnabled(0)
        self.window.e_step.setEnabled(0)

        done = False
        progress_callback.emit(done)
        self.window.source_calc_info.setText('Wait, calculation of new source-spectrum is running...')

        x_prime_max = self.window.hor_acceptance.value()
        z_prime_max = self.window.ver_acceptance.value()

        theta = np.linspace(-1.5, 1.5, 20) * x_prime_max * 1e-3
        psi = np.linspace(-1., 1., 51) * z_prime_max * 1e-3
        dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]

        if self.window.dist_e_eV.isChecked() is True:
            dist_e = 'eV'
        else:
            dist_e = 'BW'

        kwargs_wls = dict(eE=self.window.electron_e.value(), eI=self.window.ring_current.value() / 1000,
                          B0=self.window.magnetic_field.value(),
                          distE=dist_e, xPrimeMax=x_prime_max, zPrimeMax=z_prime_max)
        source_wls = rs.BendingMagnet(**kwargs_wls)
        # .intensities_on_mesh() takes the most computation time
        i0_xrt_wls = source_wls.intensities_on_mesh(self.energy, theta, psi)[0]
        self.flux_xrt_wls = i0_xrt_wls.sum(axis=(1, 2)) * dtheta * dpsi

        done = True
        progress_callback.emit(done)
        self.window.source_calc_info.setText('calculated source spectrum / keV: %.3f-%.3f /%d' %
                                             (self.window.e_min.value(), self.window.e_max.value(),
                                              self.window.e_step.value()))

        self.source_spectrum_energy = self.energy

        self.window.e_min.setEnabled(1)
        self.window.e_max.setEnabled(1)
        self.window.e_step.setEnabled(1)

    def bl_spectrum(self):

        # without a source-spectrum the energy_array comes "def energy_range", otherwise from "def xrt_source_wls"
        if self.window.source_in.isChecked() is False:
            energy_array = self.energy
        else:
            energy_array = self.source_spectrum_energy

        # the DMM
        spectrum_dmm = 1
        if self.window.dmm_in.isChecked():
            ml_system = self.window.dmm_stripe.currentText()
            # The original W/Si-multilayer of the BAMline: d(Mo) / d(Mo + B4C) = 0.4
            # d_Mo = (5.736 / 2) * 0.4 = 2.868 * 0.4 = 1.1472 nm
            # d_B4C = 2.868 - 1.1472 = 1.7208 nm
            # 1 nm = 10 angstrom
            # rho == density in g * cm-3 at room temperature

            if ml_system == 'Mo / B4C':
                mt = rm.Material(['B', 'C'], [4, 1], rho=2.52)  # top_layer
                mb = rm.Material('Mo', rho=10.28)  # bottom_layer
                ms = rm.Material('Si', rho=2.336)  # substrate

                # topLayer, thickness topLayer in angstrom, bLayer, thickness bLayer in angstrom, number of layer pairs,
                # substrate
                ml = rm.Multilayer(mt, self.window.d_top_layer.value(), mb, self.window.d_bottom_layer.value(),
                                   self.window.layer_pairs.value(), ms)

            elif ml_system == 'W / Si':
                mt = rm.Material('Si', rho=2.336)  # top_layer
                mb = rm.Material('W', rho=19.25)  # bottom_layer

                # topLayer, thickness topLayer in angstrom, bLayer, thickness bLayer in angstrom, number of layer pairs,
                # substrate (in this case same material as top_layer)
                ml = rm.Multilayer(mt, self.window.d_top_layer.value(), mb, self.window.d_bottom_layer.value(),
                                   self.window.layer_pairs.value(), mt)
            else:
                ml = rm.Material('Pd', rho=11.99)

            # reflection
            theta = self.window.dmm_theta.value()
            dmm_spol, dmm_ppol = ml.get_amplitude(energy_array, math.sin(math.radians(theta)))[0:2]

            # calculate reflection of only one mirror (not 100% sure if that is correct...)
            if self.window.dmm_one_ml.isChecked():
                spectrum_dmm = abs(dmm_spol) ** 2
            else:
                spectrum_dmm = abs(dmm_spol) ** 4

            # autoselect the filters, depending on theta_1
            if self.window.dmm_with_filters.isChecked() and not self.window.dcm_in.isChecked() and ml_system != 'Pd':

                self.window.filter1.blockSignals(True)
                self.window.filter2.blockSignals(True)
                self.window.d_filter1.blockSignals(True)
                self.window.d_filter2.blockSignals(True)

                if ml_system == 'W / Si':
                    if theta >= 1.1119:  # <= 10 keV
                        self.window.filter1.setCurrentIndex(0)
                        self.window.d_filter1.setValue(600.)
                        self.window.filter2.setCurrentIndex(0)
                        self.window.d_filter2.setValue(200.)
                    elif 1.1119 > theta >= 0.4448:  # <= 25 keV
                        self.window.filter1.setCurrentIndex(2)
                        self.window.d_filter1.setValue(200.)
                        self.window.filter2.setCurrentIndex(0)
                        self.window.d_filter2.setValue(200.)
                    elif 0.4448 > theta >= 0.3177:  # <= 35 keV
                        self.window.filter1.setCurrentIndex(0)
                        self.window.d_filter1.setValue(600.)
                        self.window.filter2.setCurrentIndex(3)
                        self.window.d_filter2.setValue(500.)
                    elif 0.3177 > theta >= 0.2647:  # <= 42 keV
                        self.window.filter1.setCurrentIndex(3)
                        self.window.d_filter1.setValue(1000.)
                        self.window.filter2.setCurrentIndex(0)
                        self.window.d_filter2.setValue(200.)
                    elif 0.2647 > theta >= 0.2224:  # <= 50 keV
                        self.window.filter1.setCurrentIndex(3)
                        self.window.d_filter1.setValue(1000.)
                        self.window.filter2.setCurrentIndex(1)
                        self.window.d_filter2.setValue(50.)
                    elif 0.2224 > theta >= 0.1544:  # <= 72 keV
                        self.window.filter1.setCurrentIndex(1)
                        self.window.d_filter1.setValue(200.)
                        self.window.filter2.setCurrentIndex(0)
                        self.window.d_filter2.setValue(200.)
                    elif theta < 0.1544:  # <= 95 keV
                        self.window.filter1.setCurrentIndex(0)
                        self.window.d_filter1.setValue(600.)
                        self.window.filter2.setCurrentIndex(2)
                        self.window.d_filter2.setValue(1000.)
                elif ml_system == 'Mo / B4C':
                    if theta >= 0.8386:  # <= 15 keV
                        self.window.filter1.setCurrentIndex(0)
                        self.window.d_filter1.setValue(600.)
                        self.window.filter2.setCurrentIndex(0)
                        self.window.d_filter2.setValue(200.)
                    elif 0.8386 > theta >= 0.4193:  # <= 30 keV
                        self.window.filter1.setCurrentIndex(2)
                        self.window.d_filter1.setValue(200.)
                        self.window.filter2.setCurrentIndex(0)
                        self.window.d_filter2.setValue(200.)
                    elif 0.4193 > theta >= 0.2995:  # <= 42 keV
                        self.window.filter1.setCurrentIndex(0)
                        self.window.d_filter1.setValue(600.)
                        self.window.filter2.setCurrentIndex(3)
                        self.window.d_filter2.setValue(500.)
                    elif 0.2995 > theta >= 0.2516:  # <= 50 keV
                        self.window.filter1.setCurrentIndex(3)
                        self.window.d_filter1.setValue(1000.)
                        self.window.filter2.setCurrentIndex(0)
                        self.window.d_filter2.setValue(200.)
                    elif 0.2516 > theta >= 0.:  # <=  keV
                        self.window.filter1.setCurrentIndex(3)
                        self.window.d_filter1.setValue(1000.)
                        self.window.filter2.setCurrentIndex(1)
                        self.window.d_filter2.setValue(50.)

                self.window.filter1.blockSignals(False)
                self.window.filter2.blockSignals(False)
                self.window.d_filter1.blockSignals(False)
                self.window.d_filter2.blockSignals(False)

        # the DCM
        spectrum_dcm = 1
        if self.window.dcm_in.isChecked():
            hkl_orientation = (1, 1, 1)
            if self.window.dcm_orientation.currentText() == '311':
                hkl_orientation = (3, 1, 1)
            elif self.window.dcm_orientation.currentText() == '333':
                hkl_orientation = (3, 3, 3)

            # hkl harmonics
            spectrum_dcm = 0
            for i in range(self.window.dcm_harmonics.value()):
                hkl_ebene = tuple(j * (i + 1) for j in hkl_orientation)
                crystal = rm.CrystalSi(hkl=hkl_ebene)
                dcm_spol, dcm_ppol = crystal.get_amplitude(energy_array,
                                                           math.sin(math.radians(self.window.dcm_theta.value())))
                # calculate reflection of only one crystal (not 100% sure if that is correct...)
                if self.window.dcm_one_crystal.isChecked() is True:
                    spectrum_dcm = spectrum_dcm + abs(dcm_spol) ** 2
                else:
                    spectrum_dcm = spectrum_dcm + abs(dcm_spol) ** 4

        # filter constellation
        filter1_text = self.window.filter1.currentText()
        filter2_text = self.window.filter2.currentText()

        transm_f1 = 1
        transm_f2 = 1

        # we need the "using_filter" variable to decide later if we need to search for zero values or not
        using_filter = False

        if 'none' not in filter1_text or 'none' not in filter2_text:
            using_filter = True

            # rho == density in g * cm-3 at room temperature

            if 'Al' in filter1_text:
                filter1 = rm.Material('Al', rho=2.6989)
            elif 'Be' in filter1_text:
                filter1 = rm.Material('Be', rho=1.848)
            elif 'Cu' in filter1_text:
                filter1 = rm.Material('Cu', rho=8.92)
            else:
                filter1 = None

            if 'Al' in filter2_text:
                filter2 = rm.Material('Al', rho=2.6989)
            elif 'Be' in filter2_text:
                filter2 = rm.Material('Be', rho=1.848)
            elif 'Cu' in filter2_text:
                filter2 = rm.Material('Cu', rho=8.92)
            else:
                filter2 = None

            if filter1:
                absorp_koeff_f1 = filter1.get_absorption_coefficient(energy_array)  # in 1 / cm
                filter1_thickness = self.window.d_filter1.value() * 0.0001  # in cm
                transm_f1 = np.exp(-absorp_koeff_f1 * filter1_thickness)

            if filter2:
                absorp_koeff_f2 = filter2.get_absorption_coefficient(energy_array)  # in 1 / cm
                filter2_thickness = self.window.d_filter2.value() * 0.0001  # in cm
                transm_f2 = np.exp(-absorp_koeff_f2 * filter2_thickness)

        transm_f_total = transm_f1 * transm_f2
        # we need to exchange all zero values with the lowest value bigger zero to be able to plot logarithmic
        # find the lowest value bigger zero and replace the zeros with that value
        # maybe this gets obsolete with a newer pyqtgraph version...
        if using_filter:
            m = min(i for i in transm_f_total if i > 0)
            if m < 1e-15:  # otherwise the plot ranges to e-200 ...
                m = 1e-15
            transm_f_total[transm_f_total < 1e-15] = m

        # what constellation do we have?
        if self.window.source_in.isChecked() is False:
            # without a source
            if self.window.dmm_in.isChecked() is False and self.window.dcm_in.isChecked() is False:
                self.window.Graph.setLabel('left', text='Transmittance / a.u.')
            else:
                self.window.Graph.setLabel('left', text='Reflectivity / a.u.')
            spectrum_bl = spectrum_dmm * spectrum_dcm * transm_f_total
        else:
            x_prime_max = self.window.hor_acceptance.value()
            z_prime_max = self.window.ver_acceptance.value()

            if self.window.dist_e_eV.isChecked() is True:
                self.window.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/eV)'
                                           .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))
            else:
                self.window.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/0.1%bw)'
                                           .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))

            spectrum_bl = self.flux_xrt_wls * spectrum_dmm * spectrum_dcm * transm_f_total

        # plot
        self.window.Graph.setLabel('bottom', text='Energy / keV')
        if self.window.line_button.isChecked() is True:
            self.window.Graph.plot(energy_array / 1e3, spectrum_bl, pen='k', clear=True, name='s-pol')
        else:
            self.window.Graph.plot(energy_array / 1e3, spectrum_bl, pen='k', clear=True, name='s-pol', symbol='o')

        self.spectrum_evaluation(energy_range=energy_array, spectrum=spectrum_bl)

        # Calculate the minimum- and maximum-energy plots depending on the dmm-beam-offset.
        hc_e = 1.2398424  # keV/nm
        if self.window.dmm_in.isChecked() and self.window.dmm_stripe.currentText() != 'Pd' and \
                self.window.dmm_off_check.isChecked():

            # this should actually come from EPICS, but we also want to use it offline, still to do...
            z2_llm = 400  # Soft-Low-Limit Z2 (needed for emin)
            z2_hlm = 1082.5  # Soft-High-Limit Z2 (needed for emax)

            # correction factor for EPICS-beamoffset
            dmm_corr = 1.036
            if self.window.dmm_stripe.currentText() == 'Mo / B4C':
                dmm_corr = 1.023

            e_min = (hc_e * dmm_corr * 2 * z2_llm) / (self.window.dmm_2d.value() * self.window.dmm_off.value())
            e_max = (hc_e * dmm_corr * 2 * z2_hlm) / (self.window.dmm_2d.value() * self.window.dmm_off.value())

            dmm_emin_line = pg.InfiniteLine(movable=False, angle=90, pen='r', label='DMM-min={value:0.3f}keV',
                                            labelOpts={'position': 0.95, 'color': 'r', 'fill': (200, 200, 200, 50),
                                                       'movable': True})

            dmm_emax_line = pg.InfiniteLine(movable=False, angle=90, pen='r', label='DMM-max={value:0.3f}keV',
                                            labelOpts={'position': 0.95, 'color': 'r', 'fill': (200, 200, 200, 50),
                                                       'movable': True})

            self.window.Graph.addItem(dmm_emin_line)
            self.window.Graph.addItem(dmm_emax_line)

            dmm_emin_line.setPos([e_min, e_min])
            dmm_emax_line.setPos([e_max, e_max])

        # Calculate the minimum- and maximum-energy plots depending on the dcm-beam-offset.
        if self.window.dcm_in.isChecked() and self.window.dcm_off_check.isChecked():
            dcm_off = self.window.dcm_off.value()

            # this should actually come from EPICS, but we also want to use it offline, still to do...
            y2_hlm = 57.
            y2_llm = 5.6
            if y2_hlm > dcm_off / 2:
                theta_max_y2 = math.degrees(math.acos(dcm_off / (2 * y2_hlm)))
            else:
                theta_max_y2 = 0.

            if y2_llm > dcm_off / 2:
                theta_min_y2 = math.degrees(math.acos(dcm_off / (2 * y2_llm)))
            else:
                theta_min_y2 = 0.

            z2_hlm = 211.5
            z2_llm = 12.5
            if z2_llm > dcm_off / 2:
                theta_max_z2 = math.degrees(math.asin(dcm_off / (2 * z2_llm)))
            else:
                theta_max_z2 = 90.

            if z2_hlm > dcm_off / 2:
                theta_min_z2 = math.degrees(math.asin(dcm_off / (2 * z2_hlm)))
            else:
                theta_min_z2 = 90.

            theta_llm = -0.5
            if theta_llm < theta_min_z2:
                if theta_min_z2 < theta_min_y2:
                    theta_min = theta_min_y2
                else:
                    theta_min = theta_min_z2
            elif theta_llm < theta_min_y2:
                theta_min = theta_min_y2
            else:
                theta_min = theta_llm

            theta_hlm = 31.
            if theta_hlm > theta_max_z2:
                if theta_max_z2 > theta_max_y2:
                    theta_max = theta_max_y2
                else:
                    theta_max = theta_max_z2
            elif theta_hlm > theta_max_y2:
                theta_max = theta_max_y2
            else:
                theta_max = theta_hlm

            if self.window.dcm_orientation.currentText() == '111':
                d_spacing = 0.6271202  # 2d/nm
            else:
                d_spacing = 0.3275029  # 2d/nm

            e_min = hc_e / (d_spacing * math.sin(math.radians(theta_max)))
            e_max = hc_e / (d_spacing * math.sin(math.radians(theta_min)))

            dcm_emin_line = pg.InfiniteLine(movable=False, angle=90, pen='b', label='DCM-min={value:0.3f}keV',
                                            labelOpts={'position': 0.95, 'color': 'b', 'fill': (200, 200, 200, 50),
                                                       'movable': True})

            dcm_emax_line = pg.InfiniteLine(movable=False, angle=90, pen='b', label='DCM-max={value:0.3f}keV',
                                            labelOpts={'position': 0.95, 'color': 'b', 'fill': (200, 200, 200, 50),
                                                       'movable': True})

            self.window.Graph.addItem(dcm_emin_line)
            self.window.Graph.addItem(dcm_emax_line)

            dcm_emin_line.setPos([e_min, e_min])
            dcm_emax_line.setPos([e_max, e_max])

    def bl_status(self):

        """Get the current motor values and put them to the calculator. Only at BAMline."""

        # the source
        self.window.ring_current.setValue(caget('bIICurrent:Mnt1'))
        self.window.magnetic_field.setValue(caget('W7IT1R:rdbk'))

        # the filters
        filter_1 = caget('OMS58:25000004_MnuAct.SVAL')
        filter_2 = caget('OMS58:25000005_MnuAct.SVAL')

        if filter_1 == '600um Be':
            self.window.filter1.setCurrentIndex(0)
        elif filter_1 == '200um Cu':
            self.window.filter1.setCurrentIndex(1)
        elif filter_1 == '200um Al':
            self.window.filter1.setCurrentIndex(2)
        elif filter_1 == '1mm Al':
            self.window.filter1.setCurrentIndex(3)
        elif filter_1 == 'none':
            self.window.filter1.setCurrentIndex(4)

        if filter_2 == '200um Be':
            self.window.filter2.setCurrentIndex(0)
        elif filter_2 == '50um Cu':
            self.window.filter2.setCurrentIndex(1)
        elif filter_2 == '1mm Cu':
            self.window.filter2.setCurrentIndex(2)
        elif filter_2 == '500um Al':
            self.window.filter2.setCurrentIndex(3)
        elif filter_2 == '60um Al':
            self.window.filter2.setCurrentIndex(4)

        # DMM
        # if dmm_y1 < -1mm: the DMM is out
        dmm_y1 = caget('OMS58:25001000.RBV')
        if dmm_y1 < -1:
            self.window.dmm_in.setChecked(0)
        else:
            # which stripe?
            dmm_x = caget('OMS58:25003004.RBV')
            if -27 < dmm_x < -12.5:
                # Pd stripe
                self.window.dmm_stripe.setCurrentIndex(2)
            elif -12.5 <= dmm_x <= 12.5:
                # W / Si stripe
                self.window.dmm_stripe.setCurrentIndex(0)
            elif 12.5 < dmm_x < 27:
                # Mo / B4C stripe
                self.window.dmm_stripe.setCurrentIndex(1)

            # angle first mirror (set also the slider-position)
            dmm_theta_1 = caget('OMS58:25000007.RBV')
            self.window.dmm_slider_theta.setValue(dmm_theta_1 * 1e4)
            self.window.dmm_theta.setValue(dmm_theta_1)
            # dmm offset
            dmm_offset = caget('Energ:25000007y2.B')
            self.window.dmm_slider_off.setValue(dmm_offset * 1e2)
            self.window.dmm_off.setValue(dmm_offset)

            self.window.dmm_in.setChecked(1)

        # DCM
        # if dcm_y < -1mm and dcm_theta < 1: the DCM is out
        dcm_y = caget('OMS58:25001007.RBV')
        dcm_theta = caget('OMS58:25002000.RBV')

        if dcm_y < -1. and dcm_theta < 1.:
            self.window.dcm_in.setChecked(0)
        else:
            crystal = caget('Energ:25002000selectCrystal')
            # which crystal orientation?
            if crystal == 0:
                self.window.dcm_orientation.setCurrentIndex(0)
            elif crystal == 1:
                self.window.dcm_orientation.setCurrentIndex(1)
            elif crystal == 1:
                self.window.dcm_orientation.setCurrentIndex(2)

            # crystal angle (set also the slider-position)
            self.window.dcm_slider_theta.setValue(dcm_theta * 1e4)
            self.window.dcm_theta.setValue(dcm_theta)
            # dcm offset
            dcm_offset = caget('Energ:25002000z2.B')
            self.window.dcm_slider_off.setValue(dcm_offset * 1e2)
            self.window.dcm_off.setValue(dcm_offset)

            self.window.dcm_in.setChecked(1)

    def toggle_expTable_off(self):

        """Only EXP_TISCH or CT-Table_Y can be checked."""

        self.window.off_expTable.blockSignals(True)
        self.window.off_expTable.setChecked(False)
        self.window.off_expTable.blockSignals(False)

    def toggle_ctTable_off(self):

        """Only EXP_TISCH or CT-Table_Y can be checked."""

        self.window.off_ctTable.blockSignals(True)
        self.window.off_ctTable.setChecked(False)
        self.window.off_ctTable.blockSignals(False)

    def bl_move(self):

        """Move the BL motors to the desired calculator positions."""

        bl_offset = 0
        dmm_off = 0
        move_motor_list = {}
        bl_offset_diff = 0
        white_beam = False
        destination_text = 'Confirm following movements:\n'

        # the filters
        filter1 = self.window.filter1.currentText()
        filter2 = self.window.filter2.currentText()
        filter1_epics = caget('OMS58:25000004_MnuAct.SVAL')
        filter2_epics = caget('OMS58:25000005_MnuAct.SVAL')
        if filter1 != filter1_epics:
            destination_text = destination_text + '\nmoving Filter 1:\t %s --> %s' % (filter1_epics, filter1)
            move_motor_list['OMS58:25000004_Mnu'] = filter1
        if filter2 != filter2_epics:
            destination_text = destination_text + '\nmoving Filter 2:\t %s --> %s' % (filter2_epics, filter2)
            move_motor_list['OMS58:25000005_Mnu'] = filter2

        if self.window.dmm_in.isChecked():

            # if the DMM is out, drive it in (DMM_Y_1 -> 0)
            dmm_y1 = round(caget('OMS58:25001000.RBV'), 2)
            if dmm_y1 < -1.:
                destination_text = destination_text + '\nmoving DMM-Y_1:\t %.2f --> 0' % dmm_y1
                move_motor_list['OMS58:25001000'] = 0

            # user wants the following stripe
            dmm_stripe = self.window.dmm_stripe.currentText()

            # what DMM calculation is currently set?
            dmm_band_epics = caget('Energ:25000007selectBand')
            # the energy calculation in EPICS is either W/Si (dmm_band_epics=0) or Mo/B4C (dmm_band_epics=1), no Pd
            if dmm_band_epics == 0:
                dmm_band_epics = 'W / Si'
            else:
                dmm_band_epics = 'Mo / B4C'

            if dmm_stripe != 'Pd':
                # tell the user only if there is really a change of selectBand
                if dmm_stripe != dmm_band_epics:
                    destination_text = destination_text + '\nsetting EPICS calculation DMM-Stripe:\t %s --> %s' % \
                                       (dmm_band_epics, dmm_stripe)

                # but we need to process selectBand anyway to drive DMM-X
                if dmm_stripe == 'W / Si':
                    move_motor_list['Energ:25000007selectBand'] = 0.
                else:
                    move_motor_list['Energ:25000007selectBand'] = 1.

            # what's with the offset?
            dmm_off = self.window.dmm_off.value()
            bl_offset += dmm_off
            dmm_off_epics = caget('Energ:25000007y2.B')

            if dmm_off != dmm_off_epics:
                destination_text = destination_text + '\nsetting DMM-Offset:\t %.2f --> %.2f' % (dmm_off_epics, dmm_off)
                move_motor_list['Energ:25000007y2.B'] = dmm_off

            if dmm_stripe == 'Pd':
                dmm_x = round(caget('OMS58:25003004.RBV'), 2)
                if dmm_x != -23.:
                    destination_text = destination_text + '\nmoving DMM-X:\t %.2f --> -23' % dmm_x
                    move_motor_list['OMS58:25003004'] = -23.

            if self.window.goto_e_max.isChecked() and dmm_stripe != 'Pd' and not self.window.dcm_in.isChecked():
                # forward energy_max to the EPICS-energy-record
                dmm_energy = round(caget('Energ:25000007rbv'), 3)
                if dmm_energy != round(self.energy_max, 3):
                    destination_text = destination_text + '\nsetting DMM-Energy:\t %.3f --> %.3f' \
                                       % (dmm_energy, round(self.energy_max, 3))
                    move_motor_list['Energ:25000007cff'] = round(self.energy_max, 3)
            else:
                # forward the theta-angle
                dmm_theta = self.window.dmm_theta.value()
                dmm_theta_epics = round(caget('OMS58:25000007.RBV'), 5)
                if dmm_theta != dmm_theta_epics:
                    destination_text = destination_text + '\nmoving DMM-Theta-1:\t %.4f --> %.4f' \
                                                          '\nmoving DMM-Theta-2 to\t %.4f' \
                                       % (dmm_theta_epics, dmm_theta, dmm_theta)
                    move_motor_list['OMS58:25000007'] = dmm_theta
                    move_motor_list['OMS58:25001003'] = dmm_theta

                # calculate the corresponding dmm_z2
                dmm_z2 = round(dmm_off / math.tan(math.radians(2 * dmm_theta)), 2)
                dmm_z2_epics = round(caget('OMS58:25001002.RBV'), 2)
                dmm_z2_hlm = round(caget('OMS58:25001002.HLM'), 2)

                if dmm_z2 > dmm_z2_hlm:
                    dmm_off_needed = round(dmm_z2_hlm * math.tan(math.radians(2 * dmm_theta)), 2)
                    destination_text = 'The calculated DMM_Z2 = %.2f exceeds the High-Limit. You need a ' \
                                       'DMM-Offset = %.2f or lower. Please recalculate!' % (dmm_z2, dmm_off_needed)

                if dmm_z2 != dmm_z2_epics:
                    destination_text = destination_text + '\nmoving DMM-Z_2:\t %.2f --> %.2f' % (dmm_z2_epics, dmm_z2)
                    move_motor_list['OMS58:25001002'] = dmm_z2
        else:
            # take the DMM out if necessary
            dmm_y1 = round(caget('OMS58:25001000.RBV'), 2)
            if dmm_y1 > -5.:
                destination_text = destination_text + '\nmoving DMM out:'
                destination_text = destination_text + '\nmoving DMM-Y_1 to\t -5'
                move_motor_list['OMS58:25001000'] = -5.
                destination_text = destination_text + '\nmoving DMM-Theta-1+2 to ~0'
                move_motor_list['OMS58:25000007'] = 0.05
                move_motor_list['OMS58:25001003'] = 0.
                destination_text = destination_text + '\nmoving DMM-Y_2 to\t 15'
                move_motor_list['OMS58:25001004'] = 15.

        if self.window.dcm_in.isChecked():

            # if the DCM is out, drive it in
            # put the DCM to the DMM-Offset if necessary
            dcm_y = round(caget('OMS58:25001007.RBV'), 2)
            if self.window.dmm_in.isChecked():
                dmm_off = self.window.dmm_off.value()
                if dcm_y != dmm_off:
                    destination_text = destination_text + '\nmoving DCM-Y:\t %.2f --> %.2f' % (dcm_y, dmm_off)
                    move_motor_list['OMS58:25001007'] = dmm_off
            else:
                if dcm_y != 0.:
                    destination_text = destination_text + '\nmoving DCM-Y:\t %.2f --> 0' % dcm_y
                    move_motor_list['OMS58:25001007'] = 0

            # user wants the following crystalorientation
            dcm_orientation = self.window.dcm_orientation.currentText()

            # what DCM orientation is currently set?
            dcm_orientation_epics = caget('Energ:25002000selectCrystal')
            if dcm_orientation_epics == 0:
                dcm_orientation_epics = '111'
            elif dcm_orientation_epics == 1:
                dcm_orientation_epics = '311'
            elif dcm_orientation_epics == 2:
                dcm_orientation_epics = '333'

            if dcm_orientation != dcm_orientation_epics:
                destination_text = destination_text + '\nsetting EPICS DCM calculation Crystal:\t %s --> %s' % \
                                   (dcm_orientation_epics, dcm_orientation)
                if dcm_orientation == '311':
                    move_motor_list['Energ:25002000selectCrystal'] = 1
                elif dcm_orientation == '333':
                    move_motor_list['Energ:25002000selectCrystal'] = 2
                else:
                    move_motor_list['Energ:25002000selectCrystal'] = 0

            # what's with the offset?
            dcm_off = self.window.dcm_off.value()
            bl_offset += dcm_off
            dcm_off_epics = caget('Energ:25002000z2.B')

            if dcm_off != dcm_off_epics:
                destination_text = destination_text + '\nsetting DCM-Offset:\t %.2f --> %.2f' % (dcm_off_epics, dcm_off)
                move_motor_list['Energ:25002000z2.B'] = dcm_off

            dcm_energy = round(caget('Energ:25002000rbv'), 3)
            if dcm_energy != round(self.energy_max, 3):
                destination_text = destination_text + '\nsetting DCM-Energy:\t %.3f --> %.3f' \
                                   % (dcm_energy, round(self.energy_max, 3))
                move_motor_list['Energ:25002000cff'] = round(self.energy_max, 3)

        else:
            # take the DCM out if necessary
            dcm_y = round(caget('OMS58:25001007.RBV'), 2)
            if dcm_y > -5.:
                destination_text = destination_text + '\nmoving DCM out:'
                destination_text = destination_text + '\nmoving DCM-Y to\t -5'
                move_motor_list['OMS58:25001007'] = -5.
                destination_text = destination_text + '\nmoving DCM-Theta to\t ~0'
                move_motor_list['OMS58:25002000'] = 0.05
                destination_text = destination_text + '\nmoving DCM-Y_2 to\t 40'
                move_motor_list['OMS58:25002003'] = 40.

        # move the Beamstop a bit under the beam
        beamstop = round(caget('OMS58:25003001.RBV'), 2)
        beamstop_hlm = round(caget('OMS58:25003001.HLM'), 2)
        beamstop_at_hlm = caget('OMS58:25003001.HLS')
        s1_ver_size = round(caget('Slot:25000002gapSize.RBV'), 2)
        beamstop_req = round(bl_offset - s1_ver_size / 2 - 2., 2)
        if beamstop != beamstop_req:
            if beamstop_req < beamstop_hlm:
                destination_text = destination_text + '\nmoving Beamstop:\t %.2f --> %.2f' % (beamstop, beamstop_req)
                move_motor_list['OMS58:25003001'] = beamstop_req
            elif beamstop_at_hlm == 0:
                destination_text = destination_text + '\nmoving Beamstop:\t %.2f --> %.2f (~High-Limit)' \
                                   % (beamstop, beamstop_hlm)
                move_motor_list['OMS58:25003001'] = beamstop_hlm

        # move s2_ver_pos to the total bl_offset
        s2_ver_pos = round(caget('Slot:25003006gapPos.RBV'), 2)
        if s2_ver_pos != bl_offset:
            destination_text = destination_text + '\nmoving S2_verPos:\t %.2f --> %.2f' % (s2_ver_pos, bl_offset)
            move_motor_list['Slot:25003006gapPos'] = bl_offset
            # use s2_ver_pos as trigger to move extra motors
            bl_offset_diff = bl_offset - s2_ver_pos

        # move the window to the total bl_offset
        window = round(caget('OMS58:25003007.RBV'), 2)
        if window != bl_offset:
            destination_text = destination_text + '\nmoving Window:\t %.2f --> %.2f' % (window, bl_offset)
            move_motor_list['OMS58:25003007'] = bl_offset

        if not self.window.dmm_in.isChecked() and not self.window.dcm_in.isChecked():
            destination_text = destination_text + '\n\nATTENTION! You are setting the white beam!\nOnly possible with' \
                                                  ' closed NebenBeamShutter.'
            white_beam = True

        # move extra motors
        if bl_offset_diff != 0:
            if self.window.off_expTable.isChecked():
                exp_table = round(caget('OMS58:25004004.RBV'), 2)
                exp_table_new = exp_table + bl_offset_diff
                destination_text = destination_text + '\nmoving EXP_TISCH:\t %.2f --> %.2f' % (exp_table, exp_table_new)
                move_motor_list['OMS58:25004004'] = exp_table_new
            if self.window.off_ctTable.isChecked():
                ct_table = round(caget('OMS58:25004003.RBV'), 2)
                ct_table_new = ct_table + bl_offset_diff
                destination_text = destination_text + '\nmoving CT-Table_Y:\t %.2f --> %.2f' % (ct_table, ct_table_new)
                move_motor_list['OMS58:25004003'] = ct_table_new
            if self.window.off_custom.toPlainText():
                print(self.window.off_custom.toPlainText())

        # show a message box to confirm movement
        msg_box = QtGui.QMessageBox()
        msg_box.setWindowTitle('Go there!')
        msg_box.setIcon(QtGui.QMessageBox.Warning)
        if destination_text == 'Confirm following movements:\n':
            destination_text = 'There is nothing to move, we are already there.'
            msg_box.setStandardButtons(QtGui.QMessageBox.Ok)
        else:
            msg_box.setStandardButtons(QtGui.QMessageBox.Cancel | QtGui.QMessageBox.Ok)
            msg_box.setEscapeButton(QtGui.QMessageBox.Cancel)
            msg_box.setDefaultButton(QtGui.QMessageBox.Ok)
            msg_box.setInformativeText('Are you sure you want to proceed ?')
        msg_box.setText(destination_text)
        retval = msg_box.exec_()
        # Cancel = 4194304
        # Ok = 1024
        if destination_text != 'There is nothing to move, we are already there.':
            if retval == 1024:
                # user confirmed
                if white_beam:
                    # NebenBeamShutter-State: 2 == opened, 1 == closed
                    nbs = caget('BS02R02U102L:State')
                    if nbs != 1:
                        info_box = QtGui.QMessageBox()
                        info_box.setWindowTitle('Close the NebenBeamShutter.')
                        info_box.setIcon(QtGui.QMessageBox.Warning)
                        info_box.setStandardButtons(QtGui.QMessageBox.Ok)
                        info_box.setText('Close the NebenBeamShutter and retry.')
                        info_box.exec_()
                        return
                for i in move_motor_list:
                    caput(i, move_motor_list[i])
                    # wait a bit because it is not good to send requests in such high frequency to the VME-IOC
                    time.sleep(0.1)
                    #print("caput(%s, %s)" % (i, move_motor_list[i]))
                return
            # anything else than OK
            return

    def load_h5(self):

        """Loads a h5-File using evefile package form PTB."""

        path = '/messung/'
        path = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', path, '*.h5')
        if path == '':
            return
        self.window.pathLine.setText(path[0])

        if not os.path.isfile(path[0]):
            return

        # this list contains the necessary motor-names to load the beamline status into BAMline-helper
        motor_list = ['FILTER_1_disc', 'FILTER_2_disc', 'DMM_THETA_1', 'DMM_Y_2', 'DMM_X']

        efile = ef.EveFile(path[0])
        mdl = efile.get_metadata(ef.Section.Snapshot)
        for md in mdl:
            elem = efile.get_data(md)
            column_name = elem.columns.tolist()
            position = elem.iloc[0][0]  # <class 'numpy.float64'>
            #print(type(column_name[0]))  # <class 'str'>
            print("%s: at position %s\n" % (column_name[0], position))


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)  # console warning fix
    app = QtWidgets.QApplication(sys.argv)
    main = Helper()
    main.show()
    sys.exit(app.exec_())
