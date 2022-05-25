# -*- coding: utf-8 -*-

from PySide2.QtWidgets import QWidget

import window_loader


class DMMParam(QWidget):
    def __init__(self):
        super().__init__()
        self.dmm_window = window_loader.load_ui('dmm_parameter.ui')

        self.dmm_window.dmm_2d_wsi.valueChanged.connect(lambda: self.new_dmm_parameter(stripe='wsi'))
        self.dmm_window.dmm_2d_mob4c.valueChanged.connect(lambda: self.new_dmm_parameter(stripe='mob4c'))
        self.dmm_window.dmm_gamma_wsi.valueChanged.connect(lambda: self.new_dmm_parameter(stripe='wsi'))
        self.dmm_window.dmm_gamma_mob4c.valueChanged.connect(lambda: self.new_dmm_parameter(stripe='mob4c'))

    def show(self):

        self.dmm_window.show()

    def new_dmm_parameter(self, stripe='wsi'):

        """Calculate top- and bottom-layer thickness when there was user input."""

        # The original W/Si-multilayer of the BAMline: d(W) / d(W + Si) = 0.4
        # d_W = (6.619 / 2) * 0.4 = 3.3095 * 0.4 = 1.3238 nm
        # d_Si = 3.3095 - 1.3238 = 1.9857 nm
        # 1 nm = 10 angstrom

        if stripe == 'wsi':
            d = self.dmm_window.dmm_2d_wsi.value() * 10 / 2
            d_bottom = d * self.dmm_window.dmm_gamma_wsi.value()
            d_top = d - d_bottom

            self.dmm_window.d_top_layer_wsi.setValue(d_top)
            self.dmm_window.d_bottom_layer_wsi.setValue(d_bottom)
        else:
            d = self.dmm_window.dmm_2d_mob4c.value() * 10 / 2
            d_bottom = d * self.dmm_window.dmm_gamma_mob4c.value()
            d_top = d - d_bottom

            self.dmm_window.d_top_layer_mob4c.setValue(d_top)
            self.dmm_window.d_bottom_layer_mob4c.setValue(d_bottom)
