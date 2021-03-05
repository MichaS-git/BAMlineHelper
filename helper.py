#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import math
import numpy as np

import helper_calc as calc

from PySide2 import QtWidgets, QtUiTools, QtCore
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
        self.window.dmm_corr.valueChanged.connect(self.bl_spectrum)
        self.window.layer_pairs.valueChanged.connect(self.bl_spectrum)
        self.window.dmm_slider_theta.valueChanged.connect(self.dmm_slider_theta_conversion)
        self.window.dmm_slider_off.valueChanged.connect(self.dmm_slider_off_conversion)
        self.window.dmm_off.valueChanged.connect(self.bl_spectrum)
        self.window.dmm_theta.valueChanged.connect(self.bl_spectrum)
        self.window.d_top_layer.valueChanged.connect(self.bl_spectrum)
        self.window.dmm_one_ml.stateChanged.connect(self.bl_spectrum)
        self.window.dmm_in.stateChanged.connect(self.bl_spectrum)
        self.window.dmm_off_check.stateChanged.connect(self.bl_spectrum)

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

        # buttons that connect to EPICS
        self.window.get_pos.clicked.connect(self.bl_status)

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
        self.window.dmm_corr.setEnabled(1)
        self.window.dmm_gamma.setEnabled(1)

        # gamma: ration of the high absorbing layer (in our case bottom) to the 2D-value
        self.window.dmm_gamma.setValue(0.4)

        if self.window.dmm_stripe.currentText() == 'W / Si':
            self.window.dmm_2d.setValue(6.619)  # 2D-value in nm
            self.window.layer_pairs.setValue(70)
            self.window.dmm_corr.setValue(1.036)
        if self.window.dmm_stripe.currentText() == 'Mo / B4C':
            self.window.dmm_2d.setValue(5.736)  # 2D-value in nm
            self.window.layer_pairs.setValue(180)
            self.window.dmm_corr.setValue(1.023)
        if self.window.dmm_stripe.currentText() == 'Pd':
            self.window.dmm_gamma.setValue(1)
            self.window.dmm_2d.setValue(0)
            self.window.layer_pairs.setValue(0)
            self.window.dmm_corr.setValue(1)
            self.window.dmm_2d.setEnabled(0)
            self.window.layer_pairs.setEnabled(0)
            self.window.dmm_corr.setEnabled(0)
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

        if 'none' in filter2_text:
            self.window.d_filter2.setValue(0.)
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
            self.window.text_calculations.setText('maximum = %.2E at %.3f keV\nFWHM = %.3f eV; center(FWHM) = %.3f keV'
                                                  % (spectrum.max(), energy_range[spectrum.argmax()] / 1000, width,
                                                     center / 1000))

        else:
            self.window.text_calculations.setText('maximum = %.2E at %.3f keV' % (spectrum.max(),
                                                                                  energy_range[spectrum.argmax()] /
                                                                                  1000))

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

        theta = np.linspace(-1., 1., 51) * x_prime_max * 1e-3
        psi = np.linspace(-1., 1., 51) * z_prime_max * 1e-3
        dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]

        if self.window.dist_e_eV.isChecked() is True:
            dist_e = 'eV'
        else:
            dist_e = 'BW'

        kwargs_wls = dict(eE=self.window.electron_e.value(), eI=self.window.beam_current.value() / 1000,
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

        # filter constellation
        filter1_text = self.window.filter1.currentText()
        filter2_text = self.window.filter2.currentText()

        # if nothing is selected --> return
        if 'none' in filter1_text and 'none' in filter2_text and self.window.source_in.isChecked() is False and \
                self.window.dmm_in.isChecked() is False and self.window.dcm_in.isChecked() is False:
            self.window.Graph.clear()
            text = pg.TextItem(text='Nothing to plot... choose your settings!', color=(200, 0, 0), anchor=(0.5, 0.5))
            self.window.Graph.addItem(text)
            # self.window.Graph.plotItem.enableAutoRange(enable=False)
            return

        # without a source-spectrum the energy_array comes "def energy_range", otherwise from "def xrt_source_wls"
        if self.window.source_in.isChecked() is False:
            energy_array = self.energy
        else:
            energy_array = self.source_spectrum_energy

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
            # tungsten is experimental only, BAMline doesn't have this filter
            elif 'W' in filter1_text:
                filter1 = rm.Material('W', rho=19.25)
            else:
                filter1 = None

            if 'Al' in filter2_text:
                filter2 = rm.Material('Al', rho=2.6989)
            elif 'Be' in filter2_text:
                filter2 = rm.Material('Be', rho=1.848)
            elif 'Cu' in filter2_text:
                filter2 = rm.Material('Cu', rho=8.92)
            # tungsten is experimental only, BAMline doesn't have this filter
            elif 'W' in filter2_text:
                filter2 = rm.Material('W', rho=19.25)
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
            if self.window.dmm_one_ml.isChecked() is True:
                spectrum_dmm = abs(dmm_spol) ** 2
            else:
                spectrum_dmm = abs(dmm_spol) ** 4

        # the DCM
        spectrum_dcm = 1
        if self.window.dcm_in.isChecked():
            if self.window.dcm_orientation.currentText() == '111':
                hkl_orientation = (1, 1, 1)
            else:
                hkl_orientation = (3, 1, 1)

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

            # this should actually come from EPICS ...
            z2_llm = 400  # Soft-Low-Limit Z2 (needed for emin)
            z2_hlm = 1082.5  # Soft-High-Limit Z2 (needed for emax)

            e_min = (hc_e * self.window.dmm_corr.value() * 2 * z2_llm) / (self.window.dmm_2d.value() *
                                                                          self.window.dmm_off.value())
            e_max = (hc_e * self.window.dmm_corr.value() * 2 * z2_hlm) / (self.window.dmm_2d.value() *
                                                                          self.window.dmm_off.value())

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

            # this should actually come from EPICS ...
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

        """Get the current motor values and put them to the calculator."""

        filter_1 = caget('OMS58:25000004_MnuAct.SVAL')
        filter_2 = caget('OMS58:25000005_MnuAct.SVAL')

        print(filter_1)
        print(filter_2)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Helper()
    main.show()
    sys.exit(app.exec_())
