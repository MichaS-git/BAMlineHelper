#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import math
import numpy as np
import glob
import re

import window_loader
import helper_calc as calc
import device_selection
import dmm_parameter
import evefile as ef  # only at BAMline

from PySide2 import QtWidgets, QtCore, QtGui
from PySide2.QtCore import QRunnable, Slot, QThreadPool, QObject, Signal

import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.sources as rs
import xraydb

import pyqtgraph as pg
from epics import caget

pg.setConfigOption('background', 'w')  # background color white (2D)
pg.setConfigOption('foreground', 'k')  # foreground color black (2D)
pg.setConfigOptions(antialias=True)  # enable antialiasing for prettier plots

# using pyqtgraph with PySide2, see also:
# https://stackoverflow.com/questions/60580391/pyqtgraph-with-pyside-2-and-qtdesigner
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


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


class Helper(QtWidgets.QMainWindow):

    def __init__(self):
        super(Helper, self).__init__()
        self.window = window_loader.load_ui(os.path.join(DIR_PATH, 'helper.ui'))
        self.window.installEventFilter(self)

        # these are BAMline dmm start-parameters
        self.dmm_param = dmm_parameter.DMMParam()
        self.layer_pairs_wsi = 70
        self.layer_pairs_mob4c = 180

        # class variables
        self.flux_xrt_wls = []  # empty flux array at startup
        self.energy_max = 1  # Maximum Energy of the spectrum in keV, determined in spectrum_evaluation

        # load PVs from xsubst-File
        # dictionary with the beamline device-names and their corresponding PVs (see func.: initialize_pvs)
        self.bl_pvs = {}
        self.initialize_pvs()

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
        self.window.plot_elements.stateChanged.connect(self.bl_spectrum)

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
        self.window.action_button.clicked.connect(self.choose_move)
        self.window.off_ctTable.toggled.connect(self.toggle_exp_table_off)
        self.window.off_expTable.toggled.connect(self.toggle_ct_table_off)

        # load h5-File parameters
        self.efile = False
        self.window.actionLoad_h5.triggered.connect(self.load_path)
        self.window.pathLine.textChanged.connect(self.load_h5)
        self.window.h5_first.clicked.connect(self.h5_navigate)
        self.window.h5_prev.clicked.connect(lambda: self.h5_navigate(direction=1))
        self.window.h5_next.clicked.connect(lambda: self.h5_navigate(direction=2))
        self.window.h5_last.clicked.connect(lambda: self.h5_navigate(direction=3))

        # new
        self.window.actionDMM_Param.triggered.connect(self.dmm_window)

    # def closeEvent(self, event):
    #
    #     for window in QtWidgets.QApplication.topLevelWidgets():
    #         window.close()

        # QtCore.QCoreApplication.instance().quit()
        # self.dmm_param = None
        # print('closed!')

    def dmm_window(self):

        self.dmm_param.show()

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

        if self.window.dmm_stripe.currentText() == 'W/Si':
            self.window.dmm_2d.setValue(6.619)  # 2D-value in nm
            self.window.layer_pairs.setValue(70)
        if self.window.dmm_stripe.currentText() == 'Mo/B4C':
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

        # without a source-spectrum the energy_array comes from "def energy_range", otherwise from "def xrt_source_wls"
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

            if ml_system == 'Mo/B4C':
                mt = rm.Material(['B', 'C'], [4, 1], rho=2.52)  # top_layer
                mb = rm.Material('Mo', rho=10.28)  # bottom_layer
                ms = rm.Material('Si', rho=2.336)  # substrate

                # topLayer, thickness topLayer in angstrom, bLayer, thickness bLayer in angstrom, number of layer pairs,
                # substrate
                ml = rm.Multilayer(mt, self.window.d_top_layer.value(), mb, self.window.d_bottom_layer.value(),
                                   self.layer_pairs_mob4c, ms)

            elif ml_system == 'W/Si':
                mt = rm.Material('Si', rho=2.336)  # top_layer
                mb = rm.Material('W', rho=19.25)  # bottom_layer

                # topLayer, thickness topLayer in angstrom, bLayer, thickness bLayer in angstrom, number of layer pairs,
                # substrate (in this case same material as top_layer)
                ml = rm.Multilayer(mt, self.window.d_top_layer.value(), mb, self.window.d_bottom_layer.value(),
                                   self.layer_pairs_wsi, mt)
            else:
                ml = rm.Material('Pd', rho=11.99)

            # reflection
            theta = self.window.dmm_theta.value()
            dmm_spol, dmm_ppol = ml.get_amplitude(energy_array, math.sin(math.radians(theta)))[0:2]

            # calculate reflection of only one mirror
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

                if ml_system == 'W/Si':
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
                elif ml_system == 'Mo/B4C':
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
            if self.window.dcm_orientation.currentText() == 'Si 311':
                hkl_orientation = (3, 1, 1)
            elif self.window.dcm_orientation.currentText() == 'Si 333':
                hkl_orientation = (3, 3, 3)

            # hkl harmonics
            spectrum_dcm = 0
            for i in range(self.window.dcm_harmonics.value()):
                hkl_ebene = tuple(j * (i + 1) for j in hkl_orientation)
                crystal = rm.CrystalSi(hkl=hkl_ebene)
                dcm_spol, dcm_ppol = crystal.get_amplitude(energy_array,
                                                           math.sin(math.radians(self.window.dcm_theta.value())))
                # calculate reflection of only one crystal
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
            self.window.Graph.plot(energy_array / 1e3, spectrum_bl, pen=pg.mkPen('k', width=1), clear=True,
                                   name='s-pol')
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
            if self.window.dmm_stripe.currentText() == 'Mo/B4C':
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

            if self.window.dcm_orientation.currentText() == 'Si 111':
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

        # plot element edges from the xray-database, syntax is: Cu [K L1] + Fe [K] + Al
        if self.window.plot_elements.isChecked() and self.window.elements.toPlainText():
            text = self.window.elements.toPlainText()

            # separate the elements
            elements = text.split('+')

            # go through the elements and separate the edges
            label_pos = 1.
            for element in elements:
                # if no edges given, show all edges within the chosen energy-range
                if '[' not in element:
                    all_edges = xraydb.xray_edges(element.strip())
                    for edge in all_edges:
                        energy = float(all_edges[edge].energy) / 1000.
                        if self.window.e_min.value() > energy or energy > self.window.e_max.value():
                            continue
                        label = element + ' ' + edge + ' ' + str(energy)
                        label_pos -= 0.025
                        edge_line = pg.InfiniteLine(movable=False, angle=90, pen=(255, 0, 0, 255), label=label,
                                                    labelOpts={'position': label_pos, 'color': 'b',
                                                               'fill': (200, 200, 200, 50), 'movable': True})
                        self.window.Graph.addItem(edge_line)
                        edge_line.setPos([energy, energy])
                else:
                    edges = element[element.find("[") + 1:element.find("]")].split(',')
                    element = element.rsplit()[0]
                    for edge in edges:
                        energy = xraydb.xray_edge(element, edge.strip(), energy_only=True) / 1000.
                        label = element + ' ' + edge + ' ' + str(energy)
                        label_pos -= 0.025
                        edge_line = pg.InfiniteLine(movable=False, angle=90, pen=(255, 0, 0, 255), label=label,
                                                    labelOpts={'position': label_pos, 'color': 'b',
                                                               'fill': (200, 200, 200, 50), 'movable': True})
                        self.window.Graph.addItem(edge_line)
                        edge_line.setPos([energy, energy])

    def bl_status(self):

        """Get the current motor values and put them to the calculator. Only at BAMline (EPICS environment)."""

        # turn off all signal-events
        self.block_signals_to_bl_spectrum(block=True)

        # the source, get the ring current and the magnetic field; if PVs are not accessible, use default values
        ring_current = caget(self.bl_pvs['Bessy_Ringstrom']['PV'])
        if not ring_current:
            ring_current = 300.
        self.window.ring_current.setValue(ring_current)
        magnetic_field = caget(self.bl_pvs['MagneticFluxDensity']['PV'])
        if not magnetic_field:
            magnetic_field = 7.
        self.window.magnetic_field.setValue(magnetic_field)

        # the filters
        self.window.filter1.setCurrentText(caget(self.bl_pvs['FILTER_1_disc']['PV'], as_string=True))
        self.window.filter2.setCurrentText(caget(self.bl_pvs['FILTER_2_disc']['PV'], as_string=True))

        # DMM
        # if dmm_y1 < -1mm: the DMM is out
        if round(caget(self.bl_pvs['DMM_Y_1']['PV']), 2) < -1:
            self.window.dmm_in.setChecked(0)
        else:
            # which stripe?
            self.window.dmm_stripe.setCurrentText(caget(self.bl_pvs['DMM_X_disc']['PV'], as_string=True))

            # angle first mirror (set also the slider-position)
            dmm_theta_1 = round(caget(self.bl_pvs['DMM_THETA_1']['PV']), 4)
            self.window.dmm_slider_theta.setValue(dmm_theta_1 * 1e4)
            self.window.dmm_theta.setValue(dmm_theta_1)
            # dmm offset
            dmm_offset = round(caget(self.bl_pvs['dmmBeamOffset']['PV']), 2)
            self.window.dmm_slider_off.setValue(dmm_offset * 1e2)
            self.window.dmm_off.setValue(dmm_offset)

            # autoselect filters on?
            if caget(self.bl_pvs['DMM_Filter_Mode']['PV'], as_string=True) == 'Energy only':
                self.window.dmm_with_filters.setChecked(0)
            else:
                self.window.dmm_with_filters.setChecked(1)

            self.window.dmm_in.setChecked(1)

        # DCM
        # if dcm_y < -1mm and dcm_theta < 1: the DCM is out
        dcm_theta = round(caget(self.bl_pvs['DCM_THETA']['PV']), 4)
        if round(caget(self.bl_pvs['DCM_Y']['PV']), 2) < -1. and dcm_theta < 1.:
            self.window.dcm_in.setChecked(0)
        else:
            self.window.dcm_orientation.setCurrentText(caget(self.bl_pvs['dcmCrystalOrientation']['PV'],
                                                             as_string=True))

            # crystal angle (set also the slider-position)
            self.window.dcm_slider_theta.setValue(dcm_theta * 1e4)
            self.window.dcm_theta.setValue(dcm_theta)
            # dcm offset
            dcm_offset = round(caget(self.bl_pvs['dcmBeamOffset']['PV']), 2)
            self.window.dcm_slider_off.setValue(dcm_offset * 1e2)
            self.window.dcm_off.setValue(dcm_offset)

            self.window.dcm_in.setChecked(1)

        # turn on all signal-events
        self.block_signals_to_bl_spectrum(block=False)

        # calculate and plot the new spectrum
        self.bl_spectrum()

    def toggle_exp_table_off(self):

        """Only EXP_TISCH or CT-Table_Y can be checked."""

        self.window.off_expTable.blockSignals(True)
        self.window.off_expTable.setChecked(False)
        self.window.off_expTable.blockSignals(False)

    def toggle_ct_table_off(self):

        """Only EXP_TISCH or CT-Table_Y can be checked."""

        self.window.off_ctTable.blockSignals(True)
        self.window.off_ctTable.setChecked(False)
        self.window.off_ctTable.blockSignals(False)

    def block_signals_to_bl_spectrum(self, block):

        """Turn OFF or ON the signals of all the GUI objects that execute bl_spectrum when user-input happens.
        :param block True = block signals
        :param block False = send signals"""

        # perform calculations when there was user input
        self.window.calc_fwhm.blockSignals(block)
        self.window.fwhm.blockSignals(block)
        self.window.line_button.blockSignals(block)

        # change global Energy-Range
        self.window.source_in.blockSignals(block)

        # user input to filter parameters
        self.window.d_filter1.blockSignals(block)
        self.window.d_filter2.blockSignals(block)
        self.window.plot_elements.blockSignals(block)

        # user input to dmm parameters
        self.window.layer_pairs.blockSignals(block)
        self.window.dmm_off.blockSignals(block)
        self.window.dmm_theta.blockSignals(block)
        self.window.d_top_layer.blockSignals(block)
        self.window.dmm_one_ml.blockSignals(block)
        self.window.dmm_in.blockSignals(block)
        self.window.dmm_off_check.blockSignals(block)
        self.window.dmm_with_filters.blockSignals(block)

        # user input to dcm parameters
        self.window.dcm_in.blockSignals(block)
        self.window.dcm_one_crystal.blockSignals(block)
        self.window.dcm_off_check.blockSignals(block)
        self.window.dcm_orientation.blockSignals(block)
        self.window.dcm_theta.blockSignals(block)
        self.window.dcm_off.blockSignals(block)
        self.window.dcm_harmonics.blockSignals(block)

    def initialize_pvs(self):

        """Load all the motor-names and motor-pv's from the bamline_main.xsubst file (only available at BAMline)."""

        xsubsts = '/messung/eve/xml/xsubst/bamline_main.xsubst', '/messung/rfa/rfa.xsubst', '/messung/ct/ct.xsubst'\
            , '/messung/eve/xml/xsubst/ringstrom.xsubst', '/messung/eve/xml/xsubst/bamline_topo.xsubst'

        # indicator for the main BL devices, 'exp' will be added to other devices to mark them for later use
        main_device = True

        for xsubst in xsubsts:

            if not xsubst == '/messung/eve/xml/xsubst/bamline_main.xsubst':
                main_device = False

            with open(xsubst) as f:
                content = f.read().splitlines()

            for i in content:
                # do not implement outcommented devices
                if i.startswith('#'):
                    continue
                # we do not need the slit motors, we use their combination: slots
                if 'Class="Slits"' in i:
                    continue
                pv = re.search('PV="(.+?)"', i)
                name = re.search('Name="(.+?)"', i)

                # if it is a dicrete-position motor, take out the nominal motor. we assume, that the nominal motor is
                # already present in bl_pvs (nominal motors come first in the xsubst-files)
                if '_disc' in i:
                    self.bl_pvs.pop(name.group(1).replace('_disc', ''), None)

                if pv and name:
                    # if its a discretePosition Motor '_Mnu' is added to the PV
                    if 'discPos3-14' in i:
                        self.bl_pvs[name.group(1)] = {'PV': pv.group(1) + '_Mnu'}
                    else:
                        self.bl_pvs[name.group(1)] = {'PV': pv.group(1)}
                    # indicate that it is a switch
                    if 'Menu' in i:
                        self.bl_pvs[name.group(1)]['switch'] = 'yes'
                    # indicate if it is a device from the user experiment
                    if not main_device:
                        self.bl_pvs[name.group(1)]['exp'] = 'yes'

        # add some additional PVs that are not directly present in the bamline_main.xsubst
        self.bl_pvs['dmmBeamOffset'] = {'PV': self.bl_pvs['DMM_Energy']['PV'] + 'y2.B'}
        self.bl_pvs['dcmBeamOffset'] = {'PV': self.bl_pvs['DCM_Energy']['PV'] + 'z2.B'}
        self.bl_pvs['dcmCrystalOrientation'] = {'PV': self.bl_pvs['DCM_Energy']['PV'] + 'selectCrystal'}
        self.bl_pvs['dcmCrystalOrientation']['switch'] = 'yes'

        # add 'cff' to the energy pseudomotors
        self.bl_pvs['DMM_Energy'] = {'PV': self.bl_pvs['DMM_Energy']['PV'] + 'cff'}
        self.bl_pvs['DCM_Energy'] = {'PV': self.bl_pvs['DCM_Energy']['PV'] + 'cff'}

    def choose_move(self):

        """Ask the user whether to move to positions from h5-file or from the GUI."""

        if self.efile:

            msg_box = QtGui.QMessageBox()
            msg_box.setWindowTitle('Choose positions')
            msg_box.setIcon(QtGui.QMessageBox.Warning)
            msg_box.setStandardButtons(QtGui.QMessageBox.Cancel | QtGui.QMessageBox.Ok)
            h5_button = msg_box.button(QtGui.QMessageBox.Ok)
            h5_button.setText('Use h5-File')
            gui_button = msg_box.button(QtGui.QMessageBox.Cancel)
            gui_button.setText('Use nominal')
            msg_box.setEscapeButton(QtGui.QMessageBox.Cancel)
            msg_box.setDefaultButton(QtGui.QMessageBox.Ok)
            msg_box.setInformativeText('The following h5-File is loaded:\n%s\nUse the positions from the h5-File or '
                                       'use the nominal GUI positions?' % self.window.pathLine.text())

            retval = msg_box.exec_()
            # Cancel = 4194304
            # Ok = 1024
            if retval == 1024:
                self.bl_move(source='h5')
            else:
                self.bl_move()
        else:
            self.bl_move()

    def bl_move(self, source='gui'):

        """Get the destination positions from the GUI or from the h5-File and open the device selection window."""

        if source == 'h5':

            for mdl in self.efile.get_metadata(ef.Section.Snapshot):
                if mdl.getName() in self.bl_pvs:
                    element = self.efile.get_metadata(ef.Section.Snapshot, name=mdl.getName())
                    position = self.efile.get_data(element[0])
                    position = position.iloc[0][0]
                    try:
                        float(position)
                        self.bl_pvs[mdl.getName()]['destination'] = round(position, 4)
                    except ValueError:
                        self.bl_pvs[mdl.getName()]['destination'] = position

        else:
            bl_offset = 0

            # the filters
            self.bl_pvs['FILTER_1_disc']['destination'] = self.window.filter1.currentText()
            self.bl_pvs['FILTER_2_disc']['destination'] = self.window.filter2.currentText()

            if self.window.dmm_with_filters.isChecked():
                self.bl_pvs['DMM_Filter_Mode']['destination'] = 'Energy & Filter'
            else:
                self.bl_pvs['DMM_Filter_Mode']['destination'] = 'Energy only'

            # the DMM
            if self.window.dmm_in.isChecked():

                # if the DMM is out, drive it in (DMM_Y_1 -> 0)
                self.bl_pvs['DMM_Y_1']['destination'] = 0

                # user wants the following stripe
                self.bl_pvs['DMM_X_disc']['destination'] = self.window.dmm_stripe.currentText()

                # how is the dmmOffset?
                self.bl_pvs['dmmBeamOffset']['destination'] = self.window.dmm_off.value()
                bl_offset += self.window.dmm_off.value()

                if self.window.goto_e_max.isChecked() and self.bl_pvs['DMM_X_disc']['destination'] != 'Pd' and not \
                        self.window.dcm_in.isChecked():
                    # forward energy_max to the DMM-energy-record
                    self.bl_pvs['DMM_Energy']['destination'] = round(self.energy_max, 3)
                else:
                    # forward the theta-angle
                    self.bl_pvs['DMM_THETA_1']['destination'] = self.window.dmm_theta.value()
                    self.bl_pvs['DMM_THETA_2']['destination'] = self.window.dmm_theta.value()

                    # calculate the corresponding dmm_z2
                    self.bl_pvs['DMM_Z_2']['destination'] = \
                        round(self.bl_pvs['dmmBeamOffset']['destination'] /
                              math.tan(math.radians(2 * self.bl_pvs['DMM_THETA_1']['destination'])), 2)
                    dmm_z2_hlm = round(caget(self.bl_pvs['DMM_Z_2']['PV'] + '.HLM'), 2)

                    # break if recalculation because of DMM-Offset is needed
                    if self.bl_pvs['DMM_Z_2']['destination'] > dmm_z2_hlm:
                        dmm_off_needed = round(dmm_z2_hlm *
                                               math.tan(math.radians(2 * self.bl_pvs['DMM_THETA_1']['destination'])), 2)
                        wrong_pd_off_text = 'The calculated DMM_Z2 = %.2f exceeds the High-Limit. You need a ' \
                                            'DMM-Offset = %.2f or lower. Please recalculate!' % \
                                            (self.bl_pvs['DMM_Z_2']['destination'], dmm_off_needed)

                        info_box = QtGui.QMessageBox()
                        info_box.setWindowTitle('Recalculate DMM-Offset.')
                        info_box.setIcon(QtGui.QMessageBox.Warning)
                        info_box.setStandardButtons(QtGui.QMessageBox.Ok)
                        info_box.setText(wrong_pd_off_text)
                        info_box.exec_()
                        return

            else:
                # take the DMM out if necessary
                if round(caget(self.bl_pvs['DMM_Y_1']['PV']), 2) > -5.:
                    self.bl_pvs['DMM_Y_1']['destination'] = -5
                    self.bl_pvs['DMM_THETA_1']['destination'] = 0.05
                    self.bl_pvs['DMM_THETA_2']['destination'] = 0
                    self.bl_pvs['DMM_Y_2']['destination'] = 15

            # the DCM
            if self.window.dcm_in.isChecked():

                # if the DCM is out, drive it in, put the DCM to the DMM-Offset if necessary
                if self.window.dmm_in.isChecked():
                    self.bl_pvs['DCM_Y']['destination'] = self.window.dmm_off.value()
                else:
                    self.bl_pvs['DCM_Y']['destination'] = 0

                # user wants the following crystalorientation
                self.bl_pvs['dcmCrystalOrientation']['destination'] = self.window.dcm_orientation.currentText()

                # what's with the offset?
                self.bl_pvs['dcmBeamOffset']['destination'] = self.window.dcm_off.value()
                bl_offset += self.window.dcm_off.value()

                # forward energy_max to the DCM-energy-record
                self.bl_pvs['DCM_Energy']['destination'] = round(self.energy_max, 4)

            else:
                # take the DCM out if necessary
                if round(caget(self.bl_pvs['DCM_Y']['PV']), 2) > -5.:
                    self.bl_pvs['DCM_Y']['destination'] = -5
                    self.bl_pvs['DCM_THETA']['destination'] = 0.05
                    self.bl_pvs['DCM_Y_2']['destination'] = 40

            # move the Beamstop a bit under the beam
            beamstop_hlm = round(caget(self.bl_pvs['BEAMSTOP']['PV'] + '.HLM'), 2)
            self.bl_pvs['BEAMSTOP']['destination'] = round(bl_offset -
                                                           round(caget(self.bl_pvs['S1_verSize']['PV']), 2) / 2 - 2., 2)
            if self.bl_pvs['BEAMSTOP']['destination'] > beamstop_hlm:
                self.bl_pvs['BEAMSTOP']['destination'] = beamstop_hlm

            # move s2_ver_pos to the total bl_offset
            self.bl_pvs['S2_verPos']['destination'] = bl_offset

            # move the window to the total bl_offset
            self.bl_pvs['WINDOW']['destination'] = bl_offset
            # use s2_ver_pos as trigger to move extra motors
            bl_offset_diff = bl_offset - round(caget(self.bl_pvs['S2_verPos']['PV']), 2)

            # move extra motors if there was a difference to the total bl_offset
            if bl_offset_diff != 0:
                if self.window.off_expTable.isChecked():
                    self.bl_pvs['EXP_TISCH']['destination'] = bl_offset_diff + \
                                                              round(caget(self.bl_pvs['EXP_TISCH']['PV']), 2)
                if self.window.off_ctTable.isChecked():
                    self.bl_pvs['CT-Table_Y']['destination'] = bl_offset_diff + \
                                                               round(caget(self.bl_pvs['CT-Table_Y']['PV']), 2)

        # show a message box to select axes and to confirm movement
        dial = device_selection.DeviceDialog(self.bl_pvs)
        if dial.exec_() == QtWidgets.QDialog.Accepted:
            dial.move_selected_devices()

        # delete the destination keys for the next pass
        for name in self.bl_pvs:
            self.bl_pvs[name].pop('destination', None)

    def load_path(self):

        """User selects h5-File to be opened."""

        directory = '/messung/'

        fname = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', directory, '*.h5')

        fname = fname[0]

        if not fname:
            return

        self.window.pathLine.setText('%s' % fname)

    def h5_navigate(self, direction=0):

        """Takes the loaded path-string, replaces the filenumber (depending on the direction) and puts the new string
        to the path-entry field."""

        fname = self.window.pathLine.text()

        if not fname:
            return

        if fname.find('.h5') == -1:
            return

        # get the h5-Filename
        h5_file_name = fname.split('/')[-1]

        # get the Filename
        file_name = h5_file_name.split('.')[0]

        orig_str_number = ''
        # loop through the Filename from back until first char
        for c in reversed(file_name):
            if c.isnumeric():
                orig_str_number += c
            else:
                break

        orig_str_number = orig_str_number[::-1]
        str_number_length = len(orig_str_number)
        # create number 1 .h5-File if the button h5-first was clicked
        new_number = '1'.zfill(str_number_length)
        orig_number = int(orig_str_number)

        if direction == 1:  # prev

            if orig_number <= 1:
                return
            new_number = str(orig_number - 1).zfill(str_number_length)

        if direction == 2:  # next

            new_number = str(orig_number + 1).zfill(str_number_length)
            directory = fname.replace(h5_file_name, '')
            counter = len(glob.glob1(directory, '*.h5'))
            if int(new_number) > counter:
                return

        if direction == 3:  # last

            directory = fname.replace(h5_file_name, '')
            counter = len(glob.glob1(directory, '*.h5'))
            new_number = str(counter).zfill(str_number_length)
            # self.window.pathLine.setText('')

        new_fname = fname.replace(str(orig_str_number).zfill(str_number_length), new_number)

        self.window.pathLine.setText('%s' % new_fname)

    def load_h5(self):

        """Loads the path-entry-field-string, opens the chosen h5-File using evefile package from PTB (only at BAMline!)
        and sets the corresponding BAMlineHelper fields (spectrum)."""

        path = self.window.pathLine.text()

        if not path:
            return
        if os.path.isfile(path) is False:
            return

        # turn off all signal-events
        self.block_signals_to_bl_spectrum(block=True)

        # this list contains the necessary motor-names to load the beamline status into BAMline-helper
        motor_list = ['FILTER_1_disc', 'FILTER_2_disc', 'DMM_X_disc', 'DMM_THETA_1', 'DMM_Y_2', 'DCM_THETA', 'DCM_Z_2']

        # load the h5-file with eveFile
        self.efile = ef.EveFile(path)

        # get the Scan-Start-Time first
        fmd = self.efile.get_file_metadata()
        self.window.h5_start_time.setText(fmd.getStartTime())

        # first: look up which monochromator(s) were in use
        mdl = self.efile.get_metadata(ef.Section.Snapshot, name='DMM_Y_1')
        elem = self.efile.get_data(mdl[0])
        dmm_y_1_pos = elem.iloc[0][0]
        if dmm_y_1_pos > -1:
            self.window.dmm_in.setChecked(1)
        else:
            self.window.dmm_in.setChecked(0)

        mdl = self.efile.get_metadata(ef.Section.Snapshot, name='DCM_Y')
        elem = self.efile.get_data(mdl[0])
        dcm_y_pos = elem.iloc[0][0]
        if dcm_y_pos > -1:
            self.window.dcm_in.setChecked(1)
        else:
            self.window.dcm_in.setChecked(0)

        # for older h5-files that did not contain the pseudo-motor 'DMM_X_disc', we need to look at DMM_X when
        # 'DMM_X_disc' was not found
        dmm_stripe_pseudo_found = False

        # now loop through the meta-data and set the proper positions/settings
        mdl = self.efile.get_metadata(ef.Section.Snapshot)
        for md in mdl:
            elem = self.efile.get_data(md)
            column_name = elem.columns.tolist()
            # print(type(column_name[0]))  # <class 'str'>
            if column_name[0] in motor_list:
                position = elem.iloc[0][0]
                # print("%s: at position %s\n" % (column_name[0], position))
                if column_name[0] == 'FILTER_1_disc':
                    index = self.window.filter1.findText(position, QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.window.filter1.setCurrentIndex(index)
                if column_name[0] == 'FILTER_2_disc':
                    index = self.window.filter2.findText(position, QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.window.filter2.setCurrentIndex(index)
                if dmm_y_1_pos > -1:
                    if column_name[0] == 'DMM_X_disc':
                        dmm_stripe_pseudo_found = True
                        index = self.window.dmm_stripe.findText(position, QtCore.Qt.MatchFixedString)
                        if index >= 0:
                            self.window.dmm_stripe.setCurrentIndex(index)
                    if column_name[0] == 'DMM_THETA_1':
                        self.window.dmm_theta.setValue(position)
                    if column_name[0] == 'DMM_Y_2':
                        self.window.dmm_off.setValue(position)
                if dcm_y_pos > -1:
                    if column_name[0] == 'DCM_THETA':
                        self.window.dcm_theta.setValue(position)
                    if column_name[0] == 'DCM_Z_2':
                        dcm_z_2 = position
                        mdl = self.efile.get_metadata(ef.Section.Snapshot, name='DCM_THETA')
                        elem = self.efile.get_data(mdl[0])
                        dcm_theta = elem.iloc[0][0]
                        dcm_offset = dcm_z_2 * 2 * math.sin(math.radians(dcm_theta))
                        self.window.dcm_off.setValue(dcm_offset)

        if dmm_y_1_pos > -1 and not dmm_stripe_pseudo_found:
            mdl = self.efile.get_metadata(ef.Section.Snapshot, name='DMM_X')
            elem = self.efile.get_data(mdl[0])
            position = elem.iloc[0][0]
            if -25 < position < -12.5:
                self.window.dmm_stripe.setCurrentIndex(2)
            if -12.5 < position < 12.5:
                self.window.dmm_stripe.setCurrentIndex(0)
            if 12.5 < position < 25:
                self.window.dmm_stripe.setCurrentIndex(1)

        # turn on all signal-events
        self.block_signals_to_bl_spectrum(block=False)

        # calculate and plot the new spectrum
        self.bl_spectrum()


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)  # console warning fix
    app = QtWidgets.QApplication(sys.argv)
    main = Helper()
    main.window.show()
    #app.aboutToQuit.connect(main.closeEvent)
    sys.exit(app.exec_())
