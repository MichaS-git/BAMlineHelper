#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import math
import helper_calc as calc

from PyQt5.uic import loadUiType  # PyQt5 für GUI
from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np
import threading  # Timer für den DCM-Controller
import time
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.sources as rs

import pyqtgraph as pg
import csv                              # csv-Package zum speichern/öffnen von csv-Files
from epics import caget, caput, camonitor

pg.setConfigOption('background', 'w')  # Plothintergrund weiß (2D)
pg.setConfigOption('foreground', 'k')  # Plotvordergrund schwarz (2D)
pg.setConfigOptions(antialias=True)  # Enable antialiasing for prettier plots

Ui_HelperWindow, QHelperWindow = loadUiType('helper.ui')  # GUI vom Hauptfenster
Ui_MapWindow, QMapWindow = loadUiType('BLMap.ui')  # GUI vom Mapfenster
Ui_XafscsvWindow, QXafscsvWindow = loadUiType('xafs_csv.ui')


class Helper(Ui_HelperWindow, QHelperWindow):

    def __init__(self):
        super(Helper, self).__init__()
        self.setupUi(self)

        # Position vom Maus-Cursor
        self.proxy = pg.SignalProxy(self.Graph.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

        # Aussehen des Plots
        # self.Graph.addLegend(offset=(1100, -600))  # Plots mit Legenden (Offsetposition in Pixeln)
        self.Map = Map()  # Verknüpfung zu Map
        self.Xafscsv = Xafscsv()

        # beim Start die erste Funktion ausführen
        # self.xrt_source_wls()  # berechne ein Source-Spektrum
        self.global_energy_range()  # set global Energy-Range
        # self.choose_function()  # erster Plot

        self.function_box.currentIndexChanged.connect(self.choose_function)  # welche Funktion rechnen?
        self.actionBLMap.triggered.connect(self.show_map)  # öffnet das BL-Map Fenster
        self.actionXAFS_CSV_List.triggered.connect(self.show_xafs_csv)
        self.calc_fwhm.stateChanged.connect(self.choose_function)
        self.fwhm.valueChanged.connect(self.choose_function)

        # chnage global Energy-Range
        self.e_min.valueChanged.connect(self.global_energy_range)
        self.e_max.valueChanged.connect(self.global_energy_range)
        self.e_step.valueChanged.connect(self.global_energy_range)

        # Source Funktionen
        self.hor_mm.valueChanged.connect(self.calc_acceptance)
        self.ver_mm.valueChanged.connect(self.calc_acceptance)
        self.distance.valueChanged.connect(self.calc_acceptance)
        self.calc_source.clicked.connect(self.xrt_source_wls)
        self.source_out.stateChanged.connect(self.choose_function)

        # Filter Funktionen
        self.filter1.currentIndexChanged.connect(self.set_filter_size)
        self.filter2.currentIndexChanged.connect(self.set_filter_size)
        self.d_filter1.valueChanged.connect(self.choose_function)
        self.d_filter2.valueChanged.connect(self.choose_function)

        # DMM Funktionen
        self.dmm_stripe.currentIndexChanged.connect(self.choose_dmm_stripe)  # passt die DMM-Parameter an
        self.dmm_2d.valueChanged.connect(self.new_dmm_parameters)  # rechnet den DMM neu
        self.dmm_gamma.valueChanged.connect(self.new_dmm_parameters)  # rechnet den DMM neu
        self.layer_pairs.valueChanged.connect(self.choose_function)
        self.dmm_slider.valueChanged.connect(self.dmm_slider_umrechnung)  # Slider in ein Integer umrechnen
        self.dmm_theta.valueChanged.connect(self.choose_function)
        self.d_top_layer.valueChanged.connect(self.choose_function)
        self.dmm_one_ml.stateChanged.connect(self.choose_function)
        self.dmm_out.stateChanged.connect(self.choose_function)

        # DCM Funktionen
        self.dcm_out.stateChanged.connect(self.choose_function)
        self.dcm_one_crystal.stateChanged.connect(self.choose_function)
        self.dcm_orientation.currentIndexChanged.connect(self.choose_function)
        self.dcm_slider.valueChanged.connect(self.dcm_slider_umrechnung)  # Slider in ein Integer umrechnen
        self.dcm_theta.valueChanged.connect(self.choose_function)
        self.dcm_harmonics.valueChanged.connect(self.choose_function)

    def view_box(self):

        """pyqtgraph viewbox for testing"""

        # linear_region = pg.LinearRegionItem([10, 40],span=(0.5, 1))
        linear_region = pg.LinearRegionItem()
        # linear_region.setZValue(-10)
        self.Graph.addItem(linear_region)

    def mouse_moved(self, evt):
        pos = evt[0]
        mouse_point = self.Graph.plotItem.vb.mapSceneToView(pos)
        self.cursor_pos.setText("cursor position: x = %0.2f y = %0.2E" % (mouse_point.x(), mouse_point.y()))

    def show_map(self):

        # öffnet das Mapfenster (siehe weiter in der Map-Klasse: class Map(Ui_MapWindow, QMapWindow): ...)
        self.Map.show()

    def show_xafs_csv(self):

        # öffnet den XAFS-CSV-Listengenerator
        self.Xafscsv.show()

    def dmm_slider_umrechnung(self):
        self.dmm_theta.setValue(self.dmm_slider.value() / 1e4)

    def dcm_slider_umrechnung(self):
        self.dcm_theta.setValue(self.dcm_slider.value() / 1e4)

    def choose_dmm_stripe(self):

        """Setzt die ursprünglichen DMM-Stripe Parameter."""

        self.dmm_2d.setEnabled(1)
        self.layer_pairs.setEnabled(1)
        self.dmm_corr.setEnabled(1)
        self.dmm_gamma.setEnabled(1)

        self.dmm_gamma.setValue(0.4)  # Gamma-Wert: das Verhältnis hochabsorbierender Layer (bottom) zum 2D-Wert

        if self.dmm_stripe.currentText() == 'W / Si':
            self.dmm_2d.setValue(6.619)  # 2D-Wert in nm
            self.layer_pairs.setValue(70)
            self.dmm_corr.setValue(1.036)
        if self.dmm_stripe.currentText() == 'Mo / B4C':
            self.dmm_2d.setValue(5.736)  # 2D-Wert in nm
            self.layer_pairs.setValue(180)
            self.dmm_corr.setValue(1.023)
        if self.dmm_stripe.currentText() == 'Pd':
            self.dmm_2d.setEnabled(0)
            self.layer_pairs.setEnabled(0)
            self.dmm_corr.setEnabled(0)
            self.dmm_gamma.setEnabled(0)
            self.dmm_2d.setValue(80)
            self.dmm_gamma.setValue(0)

    def new_dmm_parameters(self):

        """Berechnet top- und bottom-layer Dicken."""

        # Für unseren W/Si-ML gilt: d(W) / d(W + Si) = 0.4
        # d_W = (6.619 / 2) * 0.4 = 3.3095 * 0.4 = 1.3238 nm
        # d_Si = 3.3095 - 1.3238 = 1.9857 nm
        # 1 nm = 10 Angstrom
        d = self.dmm_2d.value() * 10 / 2
        d_bottom = d * self.dmm_gamma.value()
        d_top = d - d_bottom

        self.d_top_layer.setValue(d_top)
        self.d_bottom_layer.setValue(d_bottom)

    def calc_acceptance(self):

        # Öffnungswinkel in mrad aus Fläche und Quellabstand
        x_prime_max = math.atan(self.hor_mm.value() * 1e-3 / (2 * self.distance.value())) * 1000
        z_prime_max = math.atan(self.ver_mm.value() * 1e-3 / (2 * self.distance.value())) * 1000
        self.hor_acceptance.setValue(x_prime_max)
        self.ver_acceptance.setValue(z_prime_max)

    def set_filter_size(self):

        filter1_text = self.filter1.currentText()
        filter2_text = self.filter2.currentText()

        if 'none' in filter1_text:
            self.d_filter1.setValue(0.)
        if '200' in filter1_text:
            self.d_filter1.setValue(200.)
        if '600' in filter1_text:
            self.d_filter1.setValue(600.)
        if '1' in filter1_text:
            self.d_filter1.setValue(1000.)

        if 'none' in filter2_text:
            self.d_filter2.setValue(0.)
        if '50' in filter2_text:
            self.d_filter2.setValue(50.)
        if '60' in filter2_text:
            self.d_filter2.setValue(60.)
        if '200' in filter2_text:
            self.d_filter2.setValue(200.)
        if '500' in filter2_text:
            self.d_filter2.setValue(500.)
        if '1' in filter2_text:
            self.d_filter2.setValue(1000.)

    def choose_function(self):  # welche Funktion rechnen?

        if self.function_box.currentText() == 'DMM E_min / E_max vs. beamOffset':
            self.Graph.setLabel('bottom', text='DMM-beamOffset / mm')  # X-Achsenname
            self.Graph.setLabel('left', text='Energy / keV')  # Y-Achsenname

            self.dmm_beam_offsets()

        if self.function_box.currentText() == 'XRT-BAMline-Spektrum':
            self.bl_spektrum()

        if self.function_box.currentText() == 'XRT-Filter compare':
            self.xrt_filter_compare()

        if self.function_box.currentText() == 'XRT-DMM-envelope':
            self.text_calculations.setText('DMM-envelope caculation is running, see console for progress. '
                                           'BAMlineHelper will not respond during this procedure.')
            # items = ("C", "C++", "Java", "Python")
            # item, ok = QtGui.QInputDialog.getItem(self, "select input dialog",
            #                                 "list of languages", items, 0, False)
            #
            # if ok and item:
            #     self.le.setText(item)

            self.xrt_dmm_envelope()

        if self.function_box.currentText() == 'XRT-DCM-envelope':
            self.text_calculations.setText('DCM-envelope caculation is running, see console for progress. '
                                           'BAMlineHelper will not respond during this procedure.')
            self.xrt_dcm_envelope()

        if self.function_box.currentText() == 'XRT-DMM-1M-FWHM':
            self.xrt_dmm_1m_fwhm()

    def spektrum_auswertung(self, energy_range, spektrum):

        """Rechnet und plottet Maxima, FWHM, etc. ..."""

        # FWHM
        if self.calc_fwhm.isChecked() is True:
            links, rechts, sw = calc.peak_pos(energy_range, spektrum, schwelle=self.fwhm.value())
            if not links or not rechts or not sw:  # fals keine FWHM-Berechnung durchgeführt werden kann
                self.text_calculations.setText('Maximum = %.2E at %.3f keV' % (spektrum.max(),
                                                                               energy_range[
                                                                                   spektrum.argmax()] / 1000))
                return
            linke_kante = pg.InfiniteLine(movable=False, angle=90, pen=(200, 200, 10), label='left={value:0.3f}',
                                          labelOpts={'position': 0.95, 'color': (200, 200, 10),
                                                     'fill': (200, 200, 200, 50),
                                                     'movable': True})

            linke_kante.setPos([links / 1e3, links / 1e3])
            self.Graph.addItem(linke_kante)

            rechte_kante = pg.InfiniteLine(movable=False, angle=90, pen=(200, 200, 10), label='right={value:0.3f}',
                                           labelOpts={'position': 0.9, 'color': (200, 200, 10),
                                                      'fill': (200, 200, 200, 50),
                                                      'movable': True})

            rechte_kante.setPos([rechts / 1e3, rechts / 1e3])
            self.Graph.addItem(rechte_kante)

            width = abs(rechts - links)
            center = 0.5 * (rechts + links)
            self.text_calculations.setText('Maximum = %.2E at %.3f keV\nFWHM = %.3f eV; center(FWHM) = %.3f keV' %
                                           (spektrum.max(), energy_range[spektrum.argmax()] / 1000, width,
                                            center / 1000))

        else:
            self.text_calculations.setText('Maximum = %.2E at %.3f keV' % (spektrum.max(),
                                                                           energy_range[spektrum.argmax()] / 1000))

    # ab hier Funktionen des Dropdown-Menüs

    def dmm_beam_offsets(self):

        """Berechnung der minimal und maximal anzufahrenden Energie abhängig vom Beamoffset."""

        hc_e = 1.2398424  # keV/nm

        offset_spanne = np.linspace(2.5, 50, 100)  # Beamoffsetspanne

        z2_llm = 400  # das Soft-Low-Limit von Z2 (für emin)
        z2_hlm = 1082.5  # das Soft-High-Limit von Z2 (für emax)

        e_min_liste = (hc_e * self.dmm_corr.value() * 2 * z2_llm) / (self.dmm_2d.value() * offset_spanne)
        e_max_liste = (hc_e * self.dmm_corr.value() * 2 * z2_hlm) / (self.dmm_2d.value() * offset_spanne)

        self.Graph.plot(offset_spanne, e_min_liste, pen='b', clear=True, name='E_min')
        self.Graph.plot(offset_spanne, e_max_liste, pen='r', name='E_max')

        offset_strich = pg.InfiniteLine(movable=True, angle=90, pen='k', label='beamOffset={value:0.2f}\nDrag me!',
                                        labelOpts={'position': 0.8, 'color': (0, 0, 0),
                                                   'fill': (200, 200, 200, 50),
                                                   'movable': True})

        def new_energyrange():  # wenn der Nutzer den single_offset-Strich versetzt
            single_offset = offset_strich.value()
            e_min = (hc_e * self.dmm_corr.value() * 2 * z2_llm) / (self.dmm_2d.value() * single_offset)
            e_max = (hc_e * self.dmm_corr.value() * 2 * z2_hlm) / (self.dmm_2d.value() * single_offset)

            self.text_calculations.setText('Energyrange for beamOffset = %.2f\nE_min = %.3f\nE_max = %.3f' %
                                           (single_offset, e_min, e_max))

        offset_strich.sigPositionChanged.connect(new_energyrange)

        offset_strich.setPos([17, 17])  # anfangs erstmal auf beamOffset = 17
        self.Graph.addItem(offset_strich)
        # self.Graph.setXRange(5, 35)
        # self.Graph.setYRange(0, 100)

    def global_energy_range(self):

        # Energiebereich
        self.energy = np.linspace(self.e_min.value() * 1000, self.e_max.value() * 1000, self.e_step.value())

        # rechne schonmal alles, falls ohne Source
        if self.source_out.isChecked() is True:
            self.choose_function()

    def xrt_source_wls(self):

        # Energiebereich
        energy = np.linspace(self.e_min.value() * 1000, self.e_max.value() * 1000, self.e_step.value())

        print('Wait, calculation of new source-spectrum is running.')

        # die Quelle
        x_prime_max = self.hor_acceptance.value()
        z_prime_max = self.ver_acceptance.value()

        theta = np.linspace(-1., 1., 51) * x_prime_max * 1e-3
        psi = np.linspace(-1., 1., 51) * z_prime_max * 1e-3
        dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]

        if self.dist_e.currentText() == '0.1% bandwidth':
            dist_e = 'BW'
        else:
            dist_e = 'eV'

        kwargs_wls = dict(eE=self.electron_e.value(), eI=self.beam_current.value() / 1000,
                          B0=self.magnetic_field.value(),
                          distE=dist_e, xPrimeMax=x_prime_max, zPrimeMax=z_prime_max)
        source_wls = rs.BendingMagnet(**kwargs_wls)
        i0_xrt_wls = source_wls.intensities_on_mesh(energy, theta, psi)[0]
        self.flux_xrt_wls = i0_xrt_wls.sum(axis=(1, 2)) * dtheta * dpsi

        if self.compare_magnet.isChecked() is True:  # plottet noch einen 1.3T bending magnet und beendet die Rechnung
            kwargs_bending_magnet = dict(eE=self.electron_e.value(), eI=self.beam_current.value() / 1000, B0=1.3,
                                         distE=dist_e, xPrimeMax=x_prime_max, zPrimeMax=z_prime_max)
            source_bending_magnet = rs.BendingMagnet(**kwargs_bending_magnet)
            i0_xrt_bending_magnet = source_bending_magnet.intensities_on_mesh(energy, theta, psi)[0]
            flux_xrt_bending_magnet = i0_xrt_bending_magnet.sum(axis=(1, 2)) * dtheta * dpsi

            # self.Graph.setLogMode(False, True)
            # self.Graph.setYRange(7, 10, padding=0)
            
            if self.dist_e.currentText() == '0.1% bandwidth':  # Y-Achsenname
                self.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/0.1%bw)'
                                    .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))
            else:
                self.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/eV)'
                                    .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))

            self.Graph.setLabel('bottom', text='Energy / keV')  # X-Achsenname
            self.Graph.plot(energy / 1e3, self.flux_xrt_wls, pen='k', clear=True, name='7T bending magnet')
            self.Graph.plot(energy / 1e3, flux_xrt_bending_magnet, pen='b', name='1.3T bending magnet')

        print('Calculation of new source-spectrum is finished.')
        self.source_calc_info.setText('range[keV] calculated: %.3f-%.3f /%d' % (self.e_min.value(), self.e_max.value(),
                                                                                self.e_step.value()))

    def bl_spektrum(self):

        # die Filter
        filter1_text = self.filter1.currentText()
        filter2_text = self.filter2.currentText()

        # wenn nichts ausgewählt --> return
        if 'none' in filter1_text and 'none' in filter2_text and self.source_out.isChecked() is True and \
                self.dmm_out.isChecked() is True and self.dcm_out.isChecked() is True:
            self.Graph.clear()
            text = pg.TextItem(text='Nothing to plot... chose your settings!', color=(200, 0, 0), anchor=(0, 0))
            self.Graph.addItem(text)
            self.Graph.plotItem.enableAutoRange(enable=False)
            return

        transm_f1 = 1
        transm_f2 = 1
        if 'none' not in filter1_text or 'none' not in filter2_text:

            if 'Al' in filter1_text:
                filter1 = rm.Material('Al', rho=self.rho_al.value())
            elif 'Be' in filter1_text:
                filter1 = rm.Material('Be', rho=self.rho_be.value())
            elif 'Cu' in filter1_text:
                filter1 = rm.Material('Cu', rho=self.rho_cu.value())
            elif 'W' in filter1_text:
                filter1 = rm.Material('W', rho=self.rho_w_filter.value())
            else:
                filter1 = None

            if 'Al' in filter2_text:
                filter2 = rm.Material('Al', rho=self.rho_al.value())
            elif 'Be' in filter2_text:
                filter2 = rm.Material('Be', rho=self.rho_be.value())
            elif 'Cu' in filter2_text:
                filter2 = rm.Material('Cu', rho=self.rho_cu.value())
            elif 'W' in filter2_text:
                filter2 = rm.Material('W', rho=self.rho_w_filter.value())
            else:
                filter2 = None

            if filter1:
                absorp_koeff_f1 = filter1.get_absorption_coefficient(self.energy)  # in 1 / cm
                filter1_thickness = self.d_filter1.value() * 0.0001  # in cm
                transm_f1 = np.exp(-absorp_koeff_f1 * filter1_thickness)

            if filter2:
                absorp_koeff_f2 = filter2.get_absorption_coefficient(self.energy)  # in 1 / cm
                filter2_thickness = self.d_filter2.value() * 0.0001  # in cm
                transm_f2 = np.exp(-absorp_koeff_f2 * filter2_thickness)

        # der DMM
        spektrum_dmm = 1
        if self.dmm_out.isChecked() is False:
            ml_system = self.dmm_stripe.currentText()
            # Für unseren ML gilt z.B.: d(Mo) / d(Mo + B4C) = 0.4
            # d_Mo = (5.736 / 2) * 0.4 = 2.868 * 0.4 = 1.1472 nm
            # d_B4C = 2.868 - 1.1472 = 1.7208 nm
            # 1 nm = 10 Angstrom

            if ml_system == 'Mo / B4C':  # Mo/B4C Multilayer
                mt = rm.Material(['B', 'C'], [4, 1], rho=self.rho_b4c.value())  # top_layer
                mb = rm.Material('Mo', rho=self.rho_mo.value())  # bottom_layer
                ms = rm.Material('Si', rho=self.rho_si.value())  # Substrat
                # topLayer, Dicke topLayer in Angstrom, bLayer, Dicke bLayer in Angstrom, number of layer pairs,
                # Substrat
                ml = rm.Multilayer(mt, self.d_top_layer.value(), mb, self.d_bottom_layer.value(),
                                   self.layer_pairs.value(), ms)
            elif ml_system == 'W / Si':  # W/Si Multilayer
                mt = rm.Material('Si', rho=self.rho_si.value())  # top_layer
                mb = rm.Material('W', rho=self.rho_w_ml.value())  # bottom_layer
                ms = mt  # Substrat
                # topLayer, Dicke topLayer in Angstrom, bLayer, Dicke bLayer in Angstrom, number of layer pairs,
                # Substrat
                ml = rm.Multilayer(mt, self.d_top_layer.value(), mb, self.d_bottom_layer.value(),
                                   self.layer_pairs.value(), ms)
            else:
                ml = rm.Material('Pd', rho=self.rho_pd.value())

            # Reflektion
            theta = self.dmm_theta.value()
            dmm_spol, dmm_ppol = ml.get_amplitude(self.energy, math.sin(math.radians(theta)))[0:2]

            if self.dmm_one_ml.isChecked() is True:
                spektrum_dmm = abs(dmm_spol) ** 2
            else:
                spektrum_dmm = abs(dmm_spol) ** 4

        # der DCM
        spektrum_dcm = 1
        if self.dcm_out.isChecked() is False:
            if self.dcm_orientation.currentText() == '111':
                hkl_orientation = (1, 1, 1)
            else:
                hkl_orientation = (3, 1, 1)

            # gehe die Harmonischen durch
            spektrum_dcm = 0
            for i in range(self.dcm_harmonics.value()):
                hkl_ebene = tuple(j * (i + 1) for j in hkl_orientation)
                crystal = rm.CrystalSi(hkl=hkl_ebene)
                dcm_spol, dcm_ppol = crystal.get_amplitude(self.energy, math.sin(math.radians(self.dcm_theta.value())))

                if self.dcm_one_crystal.isChecked() is True:
                    spektrum_dcm = spektrum_dcm + abs(dcm_spol) ** 2
                else:
                    spektrum_dcm = spektrum_dcm + abs(dcm_spol) ** 4

        # welche Konstellation?
        if self.source_out.isChecked() is True:
            if self.dmm_out.isChecked() is True and self.dcm_out.isChecked() is True:
                self.Graph.setLabel('left', text='Transmittance / a.u.')  # Y-Achsenname
            else:
                self.Graph.setLabel('left', text='Reflectivity / a.u.')  # Y-Achsenname
            spektrum_bl = spektrum_dmm * spektrum_dcm * transm_f1 * transm_f2
        else:
            x_prime_max = self.hor_acceptance.value()
            z_prime_max = self.ver_acceptance.value()
            if self.dist_e.currentText() == '0.1% bandwidth':  # Y-Achsenname
                self.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/0.1%bw)'
                                    .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))
            else:
                self.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/eV)'
                                    .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))
            spektrum_bl = self.flux_xrt_wls * spektrum_dmm * spektrum_dcm * transm_f1 * transm_f2

        # Plot
        self.Graph.setLabel('bottom', text='Energy / keV')  # X-Achsenname
        self.Graph.plot(self.energy / 1e3, spektrum_bl, pen='k', clear=True, name='s-pol')
        self.spektrum_auswertung(energy_range=self.energy, spektrum=spektrum_bl)

    def xrt_filter_compare(self):

        """Plot aller 3 Filter-Tables."""

        e_ch_total = np.linspace(self.e_min.value() * 1000, self.e_max.value() * 1000, self.e_step.value())
        e_henke = np.linspace(10, 30000, 1000)
        e_chantler = np.linspace(11, 100000, 10000)
        e_brco = np.linspace(30, 100000, 10000)
        filter1_text = self.filter1.currentText()
        dicke = self.d_filter1.value() * 0.0001  # in cm

        if 'Al' in filter1_text:
            filter1_text = 'Al'
            density = self.rho_al.value()
        elif 'Be' in filter1_text:
            filter1_text = 'Be'
            density = self.rho_be.value()
        elif 'Cu' in filter1_text:
            filter1_text = 'Cu'
            density = self.rho_cu.value()
        elif 'W' in filter1_text:
            filter1_text = 'W'
            density = self.rho_w_filter.value()
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
        self.Graph.setLogMode(True, True)
        self.Graph.setLabel('bottom', text='Energy / keV')  # X-Achsenname
        self.Graph.setLabel('left', text='linear absorption coefficient / cm-1')  # Y-Achsenname

        # self.Graph.plot(e_ch_total, transm_ch_total, pen='k', clear=True, name='Chantler total')
        # self.Graph.plot(e_henke, transm_henke, pen='r', name='Henke')
        # self.Graph.plot(e_chantler, transm_chantler, pen='g', name='Chantler')
        # self.Graph.plot(e_brco, transm_brco, pen='b', name='BrCo')

        self.Graph.plot(e_ch_total / 1e3, mu_ch_total, pen='k', clear=True, name='Chantler total')
        self.Graph.plot(e_henke / 1e3, mu_henke, pen='r', name='Henke')
        self.Graph.plot(e_chantler / 1e3, mu_chantler, pen='g', name='Chantler')
        self.Graph.plot(e_brco / 1e3, mu_brco, pen='b', name='BrCo')

    # ab hier Spezialfunktionen, Rechnungen für Paper z.B.
    def xrt_dmm_envelope(self):

        # Energiebereich
        energy = np.linspace(self.e_min.value() * 1000, self.e_max.value() * 1000, self.e_step.value())
        self.Graph.setLabel('bottom', text='Energy / keV')  # X-Achsenname

        # die Quelle
        x_prime_max = self.hor_acceptance.value()
        z_prime_max = self.ver_acceptance.value()

        theta_wls = np.linspace(-1., 1., 51) * x_prime_max * 1e-3
        psi = np.linspace(-1., 1., 51) * z_prime_max * 1e-3
        dtheta, dpsi = theta_wls[1] - theta_wls[0], psi[1] - psi[0]

        if self.dist_e.currentText() == '0.1% bandwidth':
            dist_e = 'BW'
            # Y-Achsenname
            self.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/0.1%bw)'
                                .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))
        else:
            dist_e = 'eV'
            self.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/eV)'
                                .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))

        # self.Graph.setLogMode(False, True)
        # self.Graph.setYRange(7, 10, padding=0)

        kwargs_wls = dict(eE=self.electron_e.value(), eI=self.beam_current.value() / 1000,
                          B0=self.magnetic_field.value(),
                          distE=dist_e, xPrimeMax=x_prime_max, zPrimeMax=z_prime_max)
        source_wls = rs.BendingMagnet(**kwargs_wls)
        i0_xrt_wls = source_wls.intensities_on_mesh(energy, theta_wls, psi)[0]
        flux_xrt_wls = i0_xrt_wls.sum(axis=(1, 2)) * dtheta * dpsi

        # die Filter
        filter1_text = self.filter1.currentText()
        filter2_text = self.filter2.currentText()
        transm_f1 = 1
        transm_f2 = 1

        if 'none' not in filter1_text or 'none' not in filter2_text:

            if 'Al' in filter1_text:
                filter1 = rm.Material('Al', rho=self.rho_al.value())
            elif 'Be' in filter1_text:
                filter1 = rm.Material('Be', rho=self.rho_be.value())
            elif 'Cu' in filter1_text:
                filter1 = rm.Material('Cu', rho=self.rho_cu.value())
            elif 'W' in filter1_text:
                filter1 = rm.Material('W', rho=self.rho_w_filter.value())
            else:
                filter1 = None

            if 'Al' in filter2_text:
                filter2 = rm.Material('Al', rho=self.rho_al.value())
            elif 'Be' in filter2_text:
                filter2 = rm.Material('Be', rho=self.rho_be.value())
            elif 'Cu' in filter2_text:
                filter2 = rm.Material('Cu', rho=self.rho_cu.value())
            elif 'W' in filter2_text:
                filter2 = rm.Material('W', rho=self.rho_w_filter.value())
            else:
                filter2 = None

            if filter1:
                absorp_koeff_f1 = filter1.get_absorption_coefficient(energy)  # in 1 / cm
                filter1_thickness = self.d_filter1.value() * 0.0001  # in cm
                transm_f1 = np.exp(-absorp_koeff_f1 * filter1_thickness)

            if filter2:
                absorp_koeff_f2 = filter2.get_absorption_coefficient(energy)  # in 1 / cm
                filter2_thickness = self.d_filter2.value() * 0.0001  # in cm
                transm_f2 = np.exp(-absorp_koeff_f2 * filter2_thickness)

        # der DMM
        ml_system = self.dmm_stripe.currentText()

        if ml_system == 'Mo / B4C':  # Mo/B4C Multilayer
            theta_list = np.linspace(0.127, 2.2, 10000)  # Winkelbereich passend Mo_B4C
            mt = rm.Material(['B', 'C'], [4, 1], rho=self.rho_b4c.value())  # top_layer
            mb = rm.Material('Mo', rho=self.rho_mo.value())  # bottom_layer
        else:  # W/Si Multilayer
            theta_list = np.linspace(0.111, 2.2, 10000)  # Winkelbereich passend für W/Si
            mt = rm.Material('Si', rho=self.rho_si.value())  # top_layer
            mb = rm.Material('W', rho=self.rho_w_ml.value())  # bottom_layer

        ms = rm.Material('Si', rho=self.rho_si.value())  # Substrat
        # topLayer, Dicke topLayer in Angstrom, bLayer, Dicke bLayer in Angstrom, number of layer pairs, Substrat
        ml = rm.Multilayer(mt, self.d_top_layer.value(), mb, self.d_bottom_layer.value(), self.layer_pairs.value(), ms)

        # Reflektion, iteriere durch die Winkelliste und schreibe "max. Reflec at Energy" weg

        reflec_list = np.array([])
        energy_list = np.array([])

        for theta in theta_list:
            print('caculation running for theta %s / 2.2' % theta)
            rspol, rppol = ml.get_amplitude(energy, math.sin(math.radians(round(theta, 4))))[0:2]
            spektrum = flux_xrt_wls * abs(rspol) ** 4 * transm_f1 * transm_f2

            if 0.2 < theta < 0.38:  # guck nur auf die erste Ordnung
                reflec_list = np.append(reflec_list, spektrum[4000:].max())
                energy_list = np.append(energy_list, energy[4000:][spektrum[4000:].argmax()])
            elif theta <= 0.2:
                reflec_list = np.append(reflec_list, spektrum[10000:].max())
                energy_list = np.append(energy_list, energy[10000:][spektrum[10000:].argmax()])
            else:
                reflec_list = np.append(reflec_list, spektrum.max())
                energy_list = np.append(energy_list, energy[spektrum.argmax()])

        # Plot
        self.Graph.plot(energy_list / 1e3, reflec_list, pen='k', clear=True, name='s-pol')

    def xrt_dcm_envelope(self):
        # Energiebereich
        energy = np.linspace(self.e_min.value() * 1000, self.e_max.value() * 1000, self.e_step.value())
        self.Graph.setLabel('bottom', text='Energy / keV')  # X-Achsenname

        # die Quelle
        x_prime_max = self.hor_acceptance.value()
        z_prime_max = self.ver_acceptance.value()
        print('Calculation of Source Spectrum is running.')

        theta_wls = np.linspace(-1., 1., 51) * x_prime_max * 1e-3
        psi = np.linspace(-1., 1., 51) * z_prime_max * 1e-3
        dtheta, dpsi = theta_wls[1] - theta_wls[0], psi[1] - psi[0]

        dist_e = 'BW'
        # Y-Achsenname
        self.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/0.1%bw)'
                            .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))

        kwargs_wls = dict(eE=self.electron_e.value(), eI=self.beam_current.value() / 1000,
                          B0=self.magnetic_field.value(),
                          distE=dist_e, xPrimeMax=x_prime_max, zPrimeMax=z_prime_max)
        source_wls = rs.BendingMagnet(**kwargs_wls)
        i0_xrt_wls = source_wls.intensities_on_mesh(energy, theta_wls, psi)[0]
        flux_xrt_wls = i0_xrt_wls.sum(axis=(1, 2)) * dtheta * dpsi

        print('Calculation of Source Spectrum finished.')

        # die Filter
        filter1_text = self.filter1.currentText()
        filter2_text = self.filter2.currentText()
        transm_f1 = 1
        transm_f2 = 1

        filter1 = rm.Material('Be', rho=self.rho_be.value())
        filter2 = rm.Material('Be', rho=self.rho_be.value())

        absorp_koeff_f1 = filter1.get_absorption_coefficient(energy)  # in 1 / cm
        filter1_thickness = self.d_filter1.value() * 0.0001  # in cm
        transm_f1 = np.exp(-absorp_koeff_f1 * filter1_thickness)

        absorp_koeff_f2 = filter2.get_absorption_coefficient(energy)  # in 1 / cm
        filter2_thickness = self.d_filter2.value() * 0.0001  # in cm
        transm_f2 = np.exp(-absorp_koeff_f2 * filter2_thickness)

        # der DCM

        x_prime_max = self.hor_acceptance.value()
        z_prime_max = self.ver_acceptance.value()

        self.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/0.1%bw)'
                            .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))
        self.Graph.setLabel('left', text='total flux through {0}×{1} (mrad)² (ph/s/eV)'
                            .format(round(2 * x_prime_max, 3), round(2 * z_prime_max, 3)))

        # Reflektion, iteriere durch die Winkelliste und schreibe "max. Reflec at Energy" weg

        reflec_list = np.array([])
        energy_list = np.array([])
        theta_list = np.linspace(0.4, 24.2, 1000)  # Winkelbereich
        for theta in theta_list:
            print('caculation running for theta %s ' % theta)

            if self.dcm_orientation.currentText() == '111':
                hkl_orientation = (1, 1, 1)
            else:
                hkl_orientation = (3, 1, 1)
            hkl_ebene = tuple(j * (2) for j in hkl_orientation)
            crystal = rm.CrystalSi(hkl=hkl_ebene)
            dcm_spol, dcm_ppol = crystal.get_amplitude(energy, math.sin(math.radians(round(theta, 4))))[0:2]
            spektrum = flux_xrt_wls * abs(dcm_spol) ** 4 * transm_f1 * transm_f2
            # print(flux_xrt_wls.max(), spektrum.max(), transm_f1.max(), transm_f2.max())
            reflec_list = np.append(reflec_list, spektrum.max())
            energy_list = np.append(energy_list, energy[spektrum.argmax()])

        # Plot
        self.Graph.plot(energy_list / 1e3, reflec_list, pen='k', clear=True, name='s-pol')

    def xrt_dmm_1m_fwhm(self):

        # Energiebereich
        energy = np.linspace(self.e_min.value() * 1000, self.e_max.value() * 1000, self.e_step.value())
        self.Graph.setLabel('bottom', text='Energy / keV')  # X-Achsenname

        # der DMM
        ml_system = self.dmm_stripe.currentText()

        if ml_system == 'Mo / B4C':  # Mo/B4C Multilayer
            theta_list = np.linspace(0.127, 2.2, 10000)  # Winkelbereich passend Mo_B4C
            mt = rm.Material(['B', 'C'], [4, 1], rho=self.rho_b4c.value())  # top_layer
            mb = rm.Material('Mo', rho=self.rho_mo.value())  # bottom_layer
        else:  # W/Si Multilayer
            # theta_list = np.linspace(0.115, 2.17, 100)  # Winkelbereich passend für W/Si
            theta_list = np.linspace(0.205, 2.17, 10000)  # Winkelbereich passend für W/Si
            mt = rm.Material('Si', rho=self.rho_si.value())  # top_layer
            mb = rm.Material('W', rho=self.rho_w_ml.value())  # bottom_layer

        ms = rm.Material('Si', rho=self.rho_si.value())  # Substrat
        # topLayer, Dicke topLayer in Angstrom, bLayer, Dicke bLayer in Angstrom, number of layer pairs, Substrat
        ml = rm.Multilayer(mt, self.d_top_layer.value(), mb, self.d_bottom_layer.value(), self.layer_pairs.value(), ms,
                           substRoughness=self.roughness.value())

        # Reflektion, iteriere durch die Winkelliste und schreibe "max. Reflec at Energy" weg

        reflec_list = np.array([])
        energy_list = np.array([])
        fwhm_list = np.array([])

        for theta in theta_list:
            print('caculation running for theta %s / 2.2' % theta)
            rspol, rppol = ml.get_amplitude(energy, math.sin(math.radians(round(theta, 4))))[0:2]
            spektrum = abs(rspol) ** 2

            if theta <= 0.2:
                reflec_list = np.append(reflec_list, spektrum[10000:].max())
                energy_list = np.append(energy_list, energy[10000:][spektrum[10000:].argmax()])

                roi_energy, roi_reflec = calc.search_for_curve(energy[10000:], spektrum[10000:])
                links, rechts, sw = calc.peak_pos(roi_energy, roi_reflec, schwelle=self.fwhm.value())
                width = abs(rechts - links)
                fwhm_prozent = width / energy[spektrum.argmax()] * 100
                fwhm_list = np.append(fwhm_list, fwhm_prozent)

            else:
                reflec_list = np.append(reflec_list, spektrum.max())
                energy_list = np.append(energy_list, energy[spektrum.argmax()])

                roi_energy, roi_reflec = calc.search_for_curve(energy, spektrum)
                links, rechts, sw = calc.peak_pos(roi_energy, roi_reflec, schwelle=self.fwhm.value())
                width = abs(rechts - links)
                fwhm_prozent = width / energy[spektrum.argmax()] * 100
                fwhm_list = np.append(fwhm_list, fwhm_prozent)

        # Plot
        self.Graph.setLabel('left', text='FWHM / %')
        self.Graph.plot(energy_list / 1e3, fwhm_list, pen='k', clear=True, name='s-pol')


# Das Mapfenster


class Map(Ui_MapWindow, QMapWindow):

    def __init__(self):
        super(Map, self).__init__()
        self.setupUi(self)

        # Achsenbeschriftung
        self.map.setLabel('bottom', text='distance from source / mm')  # X-Achsenname
        self.map.setLabel('left', text='vertical beamoffset / mm')  # Y-Achsenname

        # die optische Achse
        self.map.plot((17000, 34500), (0, 0), pen=pg.mkPen('k', style=QtCore.Qt.DashLine))

        # BL-Komponenten
        # Blenden 1 (Position auf +- ~100mm geschätzt)
        self.map.plot((17800, 17800), (1, 11), pen=pg.mkPen('b'))  # obere Backe
        self.map.plot((17800, 17800), (-1, -11), pen=pg.mkPen('b'))  # untere Backe

        # Filter 1 (Position auf +- ~100mm geschätzt)
        self.map.plot((18060, 18060), (-5, 5), pen=pg.mkPen('r'))

        # Filter 2 (Position auf +- ~100mm geschätzt)
        self.map.plot((18160, 18160), (-5, 5), pen=pg.mkPen('r'))

        # Drahtmonitor M1
        self.map.plot((18487, 18487), (-15.5, -14.5), pen=pg.mkPen('r'))

        # DMM mirror 1
        self.map.plot((19239, 19559), (0, 0), pen=pg.mkPen('r'))

        # DMM mirror 2
        self.map.plot((19809, 20189), (10, 10), pen=pg.mkPen('r'))

        # Drahtmonitor M2
        self.map.plot((25542, 25542), (-15.5, -14.5), pen=pg.mkPen('r'))

        # DCM crystal 1
        self.map.plot((26750, 26850), (0, 0), pen=pg.mkPen('r'))

        # DCM crystal 2
        self.map.plot((26950, 27050), (10, 10), pen=pg.mkPen('r'))

        # Beamstop
        self.map.plot((27738, 27738), (-10, 0), pen=pg.mkPen('r', width=4.5))

        # Fluoreszenzschirm M4
        self.map.plot((28091, 28091), (-10, 0), pen=pg.mkPen('r'))

        # Blenden 2 (Position auf +- ~100mm geschätzt)
        self.map.plot((29950, 29950), (2, 12), pen=pg.mkPen('b'))  # obere Backe
        self.map.plot((29950, 29950), (-2, -12), pen=pg.mkPen('b'))  # untere Backe

        # Drahtmonitor M5 (vertical)
        self.map.plot((30330, 30330), (-15.5, -14.5), pen=pg.mkPen('r'))

        # Window
        self.map.plot((34000, 34000), (-12.5, 12.5), pen=pg.mkPen('r'))

        # Blenden 3 (Position auf +- ~100mm geschätzt)
        self.map.plot((34050, 34050), (3, 13), pen=pg.mkPen('b'))  # obere Backe
        self.map.plot((34050, 34050), (-3, -13), pen=pg.mkPen('b'))  # untere Backe


# XAFS-CSV-Listengenerator für Ana


class Xafscsv(Ui_XafscsvWindow, QXafscsvWindow):

    def __init__(self):
        super(Xafscsv, self).__init__()
        self.setupUi(self)

        self.generateList.clicked.connect(self.xafs_csv)
        self.getValues.clicked.connect(self.get_positions)

    def get_positions(self):

        y2 = caget("OMS58:25002003")
        theta = caget("OMS58:25002000")

        self.dcm_y2.setValue(y2)
        self.dcm_theta.setValue(theta)

        offset = y2 * 2 * math.cos(math.radians(theta))
        self.offset.setValue(offset)

    def xafs_csv(self):

        offset = self.dcm_y2.value() * 2 * math.cos(math.radians(self.dcm_theta.value()))
        self.offset.setValue(offset)
        e0 = self.e0.value()  # EDGE ENERGY IN eV

        prestart = self.preStart.value()  # PREEDGE START BEFORE EDGE IN eV
        prestop = self.preStop.value()  # PRE EDGE STOP BEFORE EDGE IN eV
        prestep = self.preStep.value()  # PREEDGE STEP

        xa1start = self.xa1Start.value()  # XANES 1 START BEFORE EDGE IN eV
        xa1stop = self.xa1Stop.value()  # XANES 1 STOP BEHIND EDGE IN eV
        xa1step = self.xa1Step.value()  # XANES 1 STEP

        xa2start = self.xa2Start.value()  # XANES 2 START BEHIND EDGE IN eV
        xa2stop = self.xa2Stop.value()  # XANES 2 STOP BEHIND EDGE IN eV
        xa2step = self.xa2Step.value()  # XANES 2 STEP

        # EXAFS START IN k - SPACE(XANES STOP = 50 --> k = 3.67
        #                                  =100 --> k = 5.13
        #                                 =200 --> k = 7.26
        exstart = self.exStart.value()
        exstop = self.exStop.value()  # EXAFS STOP IN k - SPACE
        # EXAFS STEP IN
        # k - SPACE (EXAFS: deltak = 0.04... 0.06, XANES deltak = 2.5)
        exstep = self.exStep.value()

        # Pre - edge
        energytable = np.arange(e0 - prestart, e0 - prestop, prestep)

        # XANES_1
        xanes1_table = np.arange(e0 - xa1start, e0 + xa1stop, xa1step)
        energytable = np.append(energytable, xanes1_table)

        # XANES_2
        xanes2_table = np.arange(e0 + xa2start, e0 + xa2stop, xa2step)
        energytable = np.append(energytable, xanes2_table)

        # EXAFS
        exafs_k_table = np.arange(exstart, exstop, exstep)
        for k in exafs_k_table:
            e = round((k ** 2 / 0.263) + e0, 4)
            energytable = np.append(energytable, e)

        dcm_y2_table = np.array([])

        for energy in energytable:
            neudcm_y2 = offset / (2. * math.cos(math.asin(1.239842 / (2 * 0.31356) / energy * 1000.0)))
            dcm_y2_table = np.append(dcm_y2_table, neudcm_y2)

        energytable = energytable / 1000
        combined_data = np.vstack((energytable, dcm_y2_table)).T

        # write it to CSV
        directory = '/messung/rfa/daten/.csv'
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save File', directory, 'CSV(*.csv)')

        path = path[0]
        header = "DCM_Energy;DCM_Y_2"
        if path == '':
            return

        np.savetxt(path, combined_data, fmt='%.4f', delimiter=';', header=header, comments='')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Helper()
    main.show()
    sys.exit(app.exec_())
