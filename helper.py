#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import math
import helper_calc as calc

from PySide2 import QtWidgets, QtUiTools, QtCore
import numpy as np
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.sources as rs

import pyqtgraph as pg
from epics import caget, caput, camonitor

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


class Helper(QtCore.QObject):

    def __init__(self):
        super(Helper, self).__init__()
        self.window = load_ui(os.path.join(DIR_PATH, 'helper.ui'))
        self.window.installEventFilter(self)

        # Position vom Maus-Cursor
        self.proxy = pg.SignalProxy(self.window.Graph.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

        # Aussehen des Plots
        # self.Graph.addLegend(offset=(1100, -600))  # Plots mit Legenden (Offsetposition in Pixeln)

        # beim Start die erste Funktion ausführen
        self.global_energy_range()  # set global Energy-Range
        self.choose_function()  # erster Plot

        self.window.function_box.currentIndexChanged.connect(self.choose_function)  # welche Funktion rechnen?
        self.window.calc_fwhm.stateChanged.connect(self.choose_function)
        self.window.fwhm.valueChanged.connect(self.choose_function)

        # chnage global Energy-Range
        self.window.e_min.valueChanged.connect(self.global_energy_range)
        self.window.e_max.valueChanged.connect(self.global_energy_range)
        self.window.e_step.valueChanged.connect(self.global_energy_range)

        # Source Funktionen
        self.window.hor_mm.valueChanged.connect(self.calc_acceptance)
        self.window.ver_mm.valueChanged.connect(self.calc_acceptance)
        self.window.distance.valueChanged.connect(self.calc_acceptance)
        self.window.calc_source.clicked.connect(self.xrt_source_wls)
        self.window.source_out.stateChanged.connect(self.choose_function)

        # Filter Funktionen
        self.window.filter1.currentIndexChanged.connect(self.set_filter_size)
        self.window.filter2.currentIndexChanged.connect(self.set_filter_size)
        self.window.d_filter1.valueChanged.connect(self.choose_function)
        self.window.d_filter2.valueChanged.connect(self.choose_function)

        # DMM Funktionen
        self.window.dmm_stripe.currentIndexChanged.connect(self.choose_dmm_stripe)  # passt die DMM-Parameter an
        self.window.dmm_2d.valueChanged.connect(self.new_dmm_parameters)  # rechnet den DMM neu
        self.window.dmm_gamma.valueChanged.connect(self.new_dmm_parameters)  # rechnet den DMM neu
        self.window.layer_pairs.valueChanged.connect(self.choose_function)
        self.window.dmm_slider.valueChanged.connect(self.dmm_slider_umrechnung)  # Slider in ein Integer umrechnen
        self.window.dmm_theta.valueChanged.connect(self.choose_function)
        self.window.d_top_layer.valueChanged.connect(self.choose_function)
        self.window.dmm_one_ml.stateChanged.connect(self.choose_function)
        self.window.dmm_out.stateChanged.connect(self.choose_function)

        # DCM Funktionen
        self.window.dcm_out.stateChanged.connect(self.choose_function)
        self.window.dcm_one_crystal.stateChanged.connect(self.choose_function)
        self.window.dcm_orientation.currentIndexChanged.connect(self.choose_function)
        self.window.dcm_slider.valueChanged.connect(self.dcm_slider_umrechnung)  # Slider in ein Integer umrechnen
        self.window.dcm_theta.valueChanged.connect(self.choose_function)
        self.window.dcm_harmonics.valueChanged.connect(self.choose_function)

        # class variables
        self.flux_xrt_wls = []  # empty flux array at startup

    def view_box(self):

        """pyqtgraph viewbox for testing"""

        # linear_region = pg.LinearRegionItem([10, 40],span=(0.5, 1))
        linear_region = pg.LinearRegionItem()
        # linear_region.setZValue(-10)
        self.window.Graph.addItem(linear_region)

    def mouse_moved(self, evt):
        pos = evt[0]
        mouse_point = self.window.Graph.plotItem.vb.mapSceneToView(pos)
        self.window.cursor_pos.setText("cursor position: x = %0.2f y = %0.2E" % (mouse_point.x(), mouse_point.y()))

    def show(self):
        self.window.show()

    def dmm_slider_umrechnung(self):
        self.window.dmm_theta.setValue(self.window.dmm_slider.value() / 1e4)

    def dcm_slider_umrechnung(self):
        self.window.dcm_theta.setValue(self.window.dcm_slider.value() / 1e4)

    def choose_dmm_stripe(self):

        """Setzt die ursprünglichen DMM-Stripe Parameter."""

        self.window.dmm_2d.setEnabled(1)
        self.window.layer_pairs.setEnabled(1)
        self.window.dmm_corr.setEnabled(1)
        self.window.dmm_gamma.setEnabled(1)

        self.window.dmm_gamma.setValue(0.4)  # Gamma-Wert: das Verhältnis hochabsorbierender Layer (bottom) zum 2D-Wert

        if self.window.dmm_stripe.currentText() == 'W / Si':
            self.window.dmm_2d.setValue(6.619)  # 2D-Wert in nm
            self.window.layer_pairs.setValue(70)
            self.window.dmm_corr.setValue(1.036)
        if self.window.dmm_stripe.currentText() == 'Mo / B4C':
            self.window.dmm_2d.setValue(5.736)  # 2D-Wert in nm
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

        """Berechnet top- und bottom-layer Dicken."""

        # Für unseren W/Si-ML gilt: d(W) / d(W + Si) = 0.4
        # d_W = (6.619 / 2) * 0.4 = 3.3095 * 0.4 = 1.3238 nm
        # d_Si = 3.3095 - 1.3238 = 1.9857 nm
        # 1 nm = 10 Angstrom
        d = self.window.dmm_2d.value() * 10 / 2
        d_bottom = d * self.window.dmm_gamma.value()
        d_top = d - d_bottom

        self.window.d_top_layer.setValue(d_top)
        self.window.d_bottom_layer.setValue(d_bottom)

    def calc_acceptance(self):

        # Öffnungswinkel in mrad aus Fläche und Quellabstand
        x_prime_max = math.atan(self.window.hor_mm.value() * 1e-3 / (2 * self.window.distance.value())) * 1000
        z_prime_max = math.atan(self.window.ver_mm.value() * 1e-3 / (2 * self.window.distance.value())) * 1000
        self.window.hor_acceptance.setValue(x_prime_max)
        self.window.ver_acceptance.setValue(z_prime_max)

    def set_filter_size(self):

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

    def choose_function(self):  # welche Funktion rechnen?

        if self.window.function_box.currentText() == 'DMM beamOffset E-Range':
            self.window.Graph.setLabel('bottom', text='DMM-beamOffset / mm')  # X-Achsenname
            self.window.Graph.setLabel('left', text='Energy / keV')  # Y-Achsenname

            self.dmm_beam_offsets()

        if self.window.function_box.currentText() == 'XRT-BAMline':

            # if there is no spectrum, calculate it (at startup)
            if len(self.flux_xrt_wls) == 0:
                self.xrt_source_wls()

            self.bl_spektrum()

        if self.window.function_box.currentText() == 'XRT-Filter compare':
            self.xrt_filter_compare()

    def spektrum_auswertung(self, energy_range, spektrum):

        """Rechnet und plottet Maxima, FWHM, etc. ..."""

        # FWHM
        if self.window.calc_fwhm.isChecked() is True:
            links, rechts, sw = calc.peak_pos(energy_range, spektrum, schwelle=self.window.fwhm.value())
            if not links or not rechts or not sw:  # fals keine FWHM-Berechnung durchgeführt werden kann
                self.window.text_calculations.setText('Maximum = %.2E at %.3f keV' % (spektrum.max(),
                                                                               energy_range[
                                                                                   spektrum.argmax()] / 1000))
                return
            linke_kante = pg.InfiniteLine(movable=False, angle=90, pen=(200, 200, 10), label='left={value:0.3f}',
                                          labelOpts={'position': 0.95, 'color': (200, 200, 10),
                                                     'fill': (200, 200, 200, 50),
                                                     'movable': True})

            linke_kante.setPos([links / 1e3, links / 1e3])
            self.window.Graph.addItem(linke_kante)

            rechte_kante = pg.InfiniteLine(movable=False, angle=90, pen=(200, 200, 10), label='right={value:0.3f}',
                                           labelOpts={'position': 0.9, 'color': (200, 200, 10),
                                                      'fill': (200, 200, 200, 50),
                                                      'movable': True})

            rechte_kante.setPos([rechts / 1e3, rechts / 1e3])
            self.window.Graph.addItem(rechte_kante)

            width = abs(rechts - links)
            center = 0.5 * (rechts + links)
            self.window.text_calculations.setText('Maximum = %.2E at %.3f keV\nFWHM = %.3f eV; center(FWHM) = %.3f keV' 
                                                  % (spektrum.max(), energy_range[spektrum.argmax()] / 1000, width, 
                                                     center / 1000))

        else:
            self.window.text_calculations.setText('Maximum = %.2E at %.3f keV' % (spektrum.max(),
                                                                                  energy_range[spektrum.argmax()] /
                                                                                  1000))

    # ab hier Funktionen des Dropdown-Menüs

    def dmm_beam_offsets(self):

        """Berechnung der minimal und maximal anzufahrenden Energie abhängig vom Beamoffset."""

        hc_e = 1.2398424  # keV/nm

        offset_spanne = np.linspace(2.5, 50, 100)  # Beamoffsetspanne

        z2_llm = 400  # das Soft-Low-Limit von Z2 (für emin)
        z2_hlm = 1082.5  # das Soft-High-Limit von Z2 (für emax)

        e_min_liste = (hc_e * self.window.dmm_corr.value() * 2 * z2_llm) / (self.window.dmm_2d.value() * offset_spanne)
        e_max_liste = (hc_e * self.window.dmm_corr.value() * 2 * z2_hlm) / (self.window.dmm_2d.value() * offset_spanne)

        self.window.Graph.plot(offset_spanne, e_min_liste, pen='b', clear=True, name='E_min')
        self.window.Graph.plot(offset_spanne, e_max_liste, pen='r', name='E_max')

        offset_strich = pg.InfiniteLine(movable=True, angle=90, pen='k', label='beamOffset={value:0.2f}\nDrag me!',
                                        labelOpts={'position': 0.8, 'color': (0, 0, 0),
                                                   'fill': (200, 200, 200, 50),
                                                   'movable': True})

        def new_energyrange():  # wenn der Nutzer den single_offset-Strich versetzt
            single_offset = offset_strich.value()
            e_min = (hc_e * self.window.dmm_corr.value() * 2 * z2_llm) / (self.window.dmm_2d.value() * single_offset)
            e_max = (hc_e * self.window.dmm_corr.value() * 2 * z2_hlm) / (self.window.dmm_2d.value() * single_offset)

            self.window.text_calculations.setText('Energyrange for beamOffset = %.2f\nE_min = %.3f\nE_max = %.3f' %
                                           (single_offset, e_min, e_max))

        offset_strich.sigPositionChanged.connect(new_energyrange)

        offset_strich.setPos([17, 17])  # anfangs erstmal auf beamOffset = 17
        self.window.Graph.addItem(offset_strich)
        # self.window.Graph.setXRange(5, 35)
        # self.window.Graph.setYRange(0, 100)

    def global_energy_range(self):

        # Energiebereich
        self.energy = np.linspace(self.window.e_min.value() * 1000, self.window.e_max.value() * 1000,
                                  self.window.e_step.value())

        # rechne schonmal alles, falls ohne Source
        if self.window.source_out.isChecked() is True:
            self.choose_function()

    def xrt_source_wls(self):

        # Energiebereich
        energy = np.linspace(self.window.e_min.value() * 1000, self.window.e_max.value() * 1000, 
                             self.window.e_step.value())

        print('Wait, calculation of new source-spectrum is running.')

        # die Quelle
        x_prime_max = self.window.hor_acceptance.value()
        z_prime_max = self.window.ver_acceptance.value()

        theta = np.linspace(-1., 1., 51) * x_prime_max * 1e-3
        psi = np.linspace(-1., 1., 51) * z_prime_max * 1e-3
        dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]

        if self.window.dist_e.currentText() == '0.1% bandwidth':
            dist_e = 'BW'
        else:
            dist_e = 'eV'

        kwargs_wls = dict(eE=self.window.electron_e.value(), eI=self.window.beam_current.value() / 1000,
                          B0=self.window.magnetic_field.value(),
                          distE=dist_e, xPrimeMax=x_prime_max, zPrimeMax=z_prime_max)
        source_wls = rs.BendingMagnet(**kwargs_wls)
        i0_xrt_wls = source_wls.intensities_on_mesh(energy, theta, psi)[0]
        self.flux_xrt_wls = i0_xrt_wls.sum(axis=(1, 2)) * dtheta * dpsi
        print(len(self.flux_xrt_wls))

        # plottet noch einen 1.3T bending magnet und beendet die Rechnung
        if self.window.compare_magnet.isChecked() is True:  
            kwargs_bending_magnet = dict(eE=self.window.electron_e.value(), eI=self.window.beam_current.value() / 1000, 
                                         B0=1.3, distE=dist_e, xPrimeMax=x_prime_max, zPrimeMax=z_prime_max)
            source_bending_magnet = rs.BendingMagnet(**kwargs_bending_magnet)
            i0_xrt_bending_magnet = source_bending_magnet.intensities_on_mesh(energy, theta, psi)[0]
            flux_xrt_bending_magnet = i0_xrt_bending_magnet.sum(axis=(1, 2)) * dtheta * dpsi

            # self.window.Graph.setLogMode(False, True)
            # self.window.Graph.setYRange(7, 10, padding=0)
            
            if self.window.dist_e.currentText() == '0.1% bandwidth':  # Y-Achsenname
                self.window.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/0.1%bw)'
                                           .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))
            else:
                self.window.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/eV)'
                                           .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))

            self.window.Graph.setLabel('bottom', text='Energy / keV')  # X-Achsenname
            self.window.Graph.plot(energy / 1e3, self.flux_xrt_wls, pen='k', clear=True, name='7T bending magnet')
            self.window.Graph.plot(energy / 1e3, flux_xrt_bending_magnet, pen='b', name='1.3T bending magnet')
            return

        print('Calculation of new source-spectrum is finished.')
        self.window.source_calc_info.setText('range[keV] calculated: %.3f-%.3f /%d' % (self.window.e_min.value(), 
                                                                                       self.window.e_max.value(), 
                                                                                       self.window.e_step.value()))
        self.choose_function()  # plot it

    def bl_spektrum(self):

        # die Filter
        filter1_text = self.window.filter1.currentText()
        filter2_text = self.window.filter2.currentText()

        # wenn nichts ausgewählt --> return
        if 'none' in filter1_text and 'none' in filter2_text and self.window.source_out.isChecked() is True and \
                self.window.dmm_out.isChecked() is True and self.window.dcm_out.isChecked() is True:
            self.window.Graph.clear()
            text = pg.TextItem(text='Nothing to plot... chose your settings!', color=(200, 0, 0), anchor=(0, 0))
            self.window.Graph.addItem(text)
            self.window.Graph.plotItem.enableAutoRange(enable=False)
            return

        transm_f1 = 1
        transm_f2 = 1
        using_filter = False
        if 'none' not in filter1_text or 'none' not in filter2_text:
            using_filter = True

            if 'Al' in filter1_text:
                filter1 = rm.Material('Al', rho=self.window.rho_al.value())
            elif 'Be' in filter1_text:
                filter1 = rm.Material('Be', rho=self.window.rho_be.value())
            elif 'Cu' in filter1_text:
                filter1 = rm.Material('Cu', rho=self.window.rho_cu.value())
            elif 'W' in filter1_text:
                filter1 = rm.Material('W', rho=self.window.rho_w_filter.value())
            else:
                filter1 = None

            if 'Al' in filter2_text:
                filter2 = rm.Material('Al', rho=self.window.rho_al.value())
            elif 'Be' in filter2_text:
                filter2 = rm.Material('Be', rho=self.window.rho_be.value())
            elif 'Cu' in filter2_text:
                filter2 = rm.Material('Cu', rho=self.window.rho_cu.value())
            elif 'W' in filter2_text:
                filter2 = rm.Material('W', rho=self.window.rho_w_filter.value())
            else:
                filter2 = None

            if filter1:
                absorp_koeff_f1 = filter1.get_absorption_coefficient(self.energy)  # in 1 / cm
                filter1_thickness = self.window.d_filter1.value() * 0.0001  # in cm
                transm_f1 = np.exp(-absorp_koeff_f1 * filter1_thickness)

            if filter2:
                absorp_koeff_f2 = filter2.get_absorption_coefficient(self.energy)  # in 1 / cm
                filter2_thickness = self.window.d_filter2.value() * 0.0001  # in cm
                transm_f2 = np.exp(-absorp_koeff_f2 * filter2_thickness)

        transm_f_total = transm_f1 * transm_f2
        # we need to exchange all zero values with the lowest value bigger zero to be able to plot logarithmic
        # find the lowest value bigger zero and replace the zeros with that
        if using_filter:
            m = min(i for i in transm_f_total if i > 0)
            if m < 1e-15:  # otherwise the plot ranges to e-200 ...
                m = 1e-15
            transm_f_total[transm_f_total < 1e-15] = m

        # der DMM
        spektrum_dmm = 1
        if self.window.dmm_out.isChecked() is False:
            ml_system = self.window.dmm_stripe.currentText()
            # Für unseren ML gilt z.B.: d(Mo) / d(Mo + B4C) = 0.4
            # d_Mo = (5.736 / 2) * 0.4 = 2.868 * 0.4 = 1.1472 nm
            # d_B4C = 2.868 - 1.1472 = 1.7208 nm
            # 1 nm = 10 Angstrom

            if ml_system == 'Mo / B4C':  # Mo/B4C Multilayer
                mt = rm.Material(['B', 'C'], [4, 1], rho=self.window.rho_b4c.value())  # top_layer
                mb = rm.Material('Mo', rho=self.window.rho_mo.value())  # bottom_layer
                ms = rm.Material('Si', rho=self.window.rho_si.value())  # Substrat
                # topLayer, Dicke topLayer in Angstrom, bLayer, Dicke bLayer in Angstrom, number of layer pairs,
                # Substrat
                ml = rm.Multilayer(mt, self.window.d_top_layer.value(), mb, self.window.d_bottom_layer.value(),
                                   self.window.layer_pairs.value(), ms)
            elif ml_system == 'W / Si':  # W/Si Multilayer
                mt = rm.Material('Si', rho=self.window.rho_si.value())  # top_layer
                mb = rm.Material('W', rho=self.window.rho_w_ml.value())  # bottom_layer
                ms = mt  # Substrat
                # topLayer, Dicke topLayer in Angstrom, bLayer, Dicke bLayer in Angstrom, number of layer pairs,
                # Substrat
                ml = rm.Multilayer(mt, self.window.d_top_layer.value(), mb, self.window.d_bottom_layer.value(),
                                   self.window.layer_pairs.value(), ms)
            else:
                ml = rm.Material('Pd', rho=self.window.rho_pd.value())

            # Reflektion
            theta = self.window.dmm_theta.value()
            dmm_spol, dmm_ppol = ml.get_amplitude(self.energy, math.sin(math.radians(theta)))[0:2]

            if self.window.dmm_one_ml.isChecked() is True:
                spektrum_dmm = abs(dmm_spol) ** 2
            else:
                spektrum_dmm = abs(dmm_spol) ** 4

        # der DCM
        spektrum_dcm = 1
        if self.window.dcm_out.isChecked() is False:
            if self.window.dcm_orientation.currentText() == '111':
                hkl_orientation = (1, 1, 1)
            else:
                hkl_orientation = (3, 1, 1)

            # gehe die Harmonischen durch
            spektrum_dcm = 0
            for i in range(self.window.dcm_harmonics.value()):
                hkl_ebene = tuple(j * (i + 1) for j in hkl_orientation)
                crystal = rm.CrystalSi(hkl=hkl_ebene)
                dcm_spol, dcm_ppol = crystal.get_amplitude(self.energy, 
                                                           math.sin(math.radians(self.window.dcm_theta.value())))

                if self.window.dcm_one_crystal.isChecked() is True:
                    spektrum_dcm = spektrum_dcm + abs(dcm_spol) ** 2
                else:
                    spektrum_dcm = spektrum_dcm + abs(dcm_spol) ** 4

        # welche Konstellation?
        if self.window.source_out.isChecked() is True:
            if self.window.dmm_out.isChecked() is True and self.window.dcm_out.isChecked() is True:
                self.window.Graph.setLabel('left', text='Transmittance / a.u.')  # Y-Achsenname
            else:
                self.window.Graph.setLabel('left', text='Reflectivity / a.u.')  # Y-Achsenname
            spektrum_bl = spektrum_dmm * spektrum_dcm * transm_f_total
        else:
            x_prime_max = self.window.hor_acceptance.value()
            z_prime_max = self.window.ver_acceptance.value()
            if self.window.dist_e.currentText() == '0.1% bandwidth':  # Y-Achsenname
                self.window.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/0.1%bw)'
                                           .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))
            else:
                self.window.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/eV)'
                                           .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))
            spektrum_bl = self.flux_xrt_wls * spektrum_dmm * spektrum_dcm * transm_f_total

        # Plot
        self.window.Graph.setLabel('bottom', text='Energy / keV')  # X-Achsenname
        self.window.Graph.plot(self.energy / 1e3, spektrum_bl, pen='k', clear=True, name='s-pol')
        self.spektrum_auswertung(energy_range=self.energy, spektrum=spektrum_bl)

    def xrt_filter_compare(self):

        """Plot aller 3 Filter-Tables."""

        e_ch_total = np.linspace(self.window.e_min.value() * 1000, self.window.e_max.value() * 1000, 
                                 self.window.e_step.value())
        e_henke = np.linspace(10, 30000, 1000)
        e_chantler = np.linspace(11, 100000, 10000)
        e_brco = np.linspace(30, 100000, 10000)
        filter1_text = self.window.filter1.currentText()
        dicke = self.window.d_filter1.value() * 0.0001  # in cm

        if 'Al' in filter1_text:
            filter1_text = 'Al'
            density = self.window.rho_al.value()
        elif 'Be' in filter1_text:
            filter1_text = 'Be'
            density = self.window.rho_be.value()
        elif 'Cu' in filter1_text:
            filter1_text = 'Cu'
            density = self.window.rho_cu.value()
        elif 'W' in filter1_text:
            filter1_text = 'W'
            density = self.window.rho_w_filter.value()
        else:
            return

        filter1_ch_total = rm.Material(filter1_text, rho=density)
        filter1_henke = rm.Material(filter1_text, rho=density, table='Henke')
        filter1_chantler = rm.Material(filter1_text, rho=density, table='Chantler')
        filter1_brco = rm.Material(filter1_text, rho=density, table='BrCo')

        mu_ch_total = filter1_ch_total.get_absorption_coefficient(e_ch_total)  # in 1 / cm
        mu_henke = filter1_henke.get_absorption_coefficient(e_henke)  # in 1 / cm
        mu_chantler = filter1_chantler.get_absorption_coefficient(e_chantler)  # in 1 / cm
        mu_brco = filter1_brco.get_absorption_coefficient(e_brco)  # in 1 / cm

        # transm_ch_total = np.exp(-mu_ch_total * dicke)
        # transm_henke = np.exp(-mu_henke * dicke)
        # transm_chantler = np.exp(-mu_chantler * dicke)
        # transm_brco = np.exp(-mu_brco * dicke)

        # Plot
        self.window.Graph.setLogMode(True, True)
        self.window.Graph.setLabel('bottom', text='Energy / keV')  # X-Achsenname
        self.window.Graph.setLabel('left', text='linear absorption coefficient / cm-1')  # Y-Achsenname

        # self.window.Graph.plot(e_ch_total, transm_ch_total, pen='k', clear=True, name='Chantler total')
        # self.window.Graph.plot(e_henke, transm_henke, pen='r', name='Henke')
        # self.window.Graph.plot(e_chantler, transm_chantler, pen='g', name='Chantler')
        # self.window.Graph.plot(e_brco, transm_brco, pen='b', name='BrCo')

        self.window.Graph.plot(e_ch_total / 1e3, mu_ch_total, pen='k', clear=True, name='Chantler total')
        self.window.Graph.plot(e_henke / 1e3, mu_henke, pen='r', name='Henke')
        self.window.Graph.plot(e_chantler / 1e3, mu_chantler, pen='g', name='Chantler')
        self.window.Graph.plot(e_brco / 1e3, mu_brco, pen='b', name='BrCo')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Helper()
    main.show()
    sys.exit(app.exec_())
