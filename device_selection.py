# -*- coding: utf-8 -*-
# import sys
import time

from PySide6 import QtWidgets, QtCore
from epics import caget, caput, cainfo


class DeviceDialog(QtWidgets.QDialog):
    def __init__(self, device_list):
        super().__init__()
        self.setWindowTitle('Select devices to move')

        self.device_list = device_list

        formLayout = QtWidgets.QFormLayout()
        groupBox = QtWidgets.QGroupBox()

        for name in self.device_list:
            # if there isn't a destination position for the device, skip it
            if 'destination' not in self.device_list[name].keys():
                continue

            # get the position of the PV; if the positions is a switch, get the position string
            if 'switch' in self.device_list[name].keys():
                pos = caget(self.device_list[name]['PV'], as_string=True, timeout=0.1)
            else:
                pos = caget(self.device_list[name]['PV'], timeout=0.1)

            # if the PV wasn't found, tell it to the user
            online = True
            # we have to compare with None because a zero position is not recognized as true
            if pos == None:
                self.device_list[name]['position'] = 'PV OFFLINE!'
                online = False

            # else, get the current device position
            if online:
                if 'switch' in self.device_list[name].keys():
                    self.device_list[name]['position'] = pos
                else:
                    self.device_list[name]['position'] = round(pos, 4)

            # if the device is already at the destination, skip it
            if self.device_list[name]['position'] == self.device_list[name]['destination']:
                continue

            layout_h = QtWidgets.QHBoxLayout()

            check_box = QtWidgets.QCheckBox()
            # do not auto-check offline and user-experiment devices
            if online and 'exp' not in self.device_list[name].keys():
                check_box.setChecked(True)
            else:
                check_box.setChecked(False)
                if not online:
                    check_box.setEnabled(False)
            check_box.setObjectName(name)

            name_label = QtWidgets.QLabel(name)
            name_label.setFixedWidth(160)

            pos_label = QtWidgets.QLabel(str(self.device_list[name]['position']))
            pos_label.setFixedWidth(100)

            arrow_label = QtWidgets.QLabel('>>>')
            arrow_label.setFixedWidth(40)

            # if it's a switch-PV, get all possible fields and put them into a QComboBox, else it's a QLineEdit
            if 'switch' in self.device_list[name].keys():
                info = cainfo(self.device_list[name]['PV'], print_out=False)
                info_as_list = info.splitlines()
                # get rid ot the whitespaces
                info_as_list = [i.strip() for i in info_as_list]
                # get all the enum strings except for the 'undefined' one
                enums = []
                if 'enum strings:' in info_as_list:
                    list_slice = info_as_list[info_as_list.index('enum strings:'):]
                    for i, n in enumerate(list_slice):
                        if list_slice[i][0].isdigit():
                            if 'Undefined' in list_slice[i]:
                                continue
                            else:
                                enums.append(list_slice[i].split('= ')[1])
                dest = QtWidgets.QComboBox()
                dest.addItems(enums)
                dest.setCurrentText(str(self.device_list[name]['destination']))
            else:
                dest = QtWidgets.QLineEdit(str(self.device_list[name]['destination']))
                dest.setAlignment(QtCore.Qt.AlignRight)

            dest.setMaximumWidth(80)
            dest.setFixedWidth(100)
            dest.setObjectName(name + 'dest')

            layout_h.addWidget(check_box)
            layout_h.addWidget(name_label)
            layout_h.addWidget(pos_label)
            layout_h.addWidget(arrow_label)
            layout_h.addWidget(dest)

            layout_h.setObjectName(name)

            formLayout.addRow(layout_h)

        groupBox.setLayout(formLayout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(groupBox)
        scroll.setWidgetResizable(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
                                                QtCore.Qt.Horizontal, self)
        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    def move_selected_devices(self):

        """Find all the checked devices and caput them to the desired position"""

        pos = 0
        for box in self.findChildren(QtWidgets.QCheckBox):
            if box.checkState():
                if 'switch' in self.device_list[box.objectName()].keys():
                    for i in self.findChildren(QtWidgets.QComboBox, box.objectName() + 'dest'):
                        pos = i.currentText()
                else:
                    for i in self.findChildren(QtWidgets.QLineEdit, box.objectName() + 'dest'):
                        pos = i.text()

                caput(self.device_list[box.objectName()]['PV'], pos)
                # print("caput(%s, %s)" % (self.device_list[box.objectName()]['PV'], pos))
                # wait a bit because it is not good to send requests in such high frequency to the VME-IOC
                time.sleep(0.1)
