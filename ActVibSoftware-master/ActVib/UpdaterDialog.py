# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'UpdaterDialog.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_UpdaterDialog(object):
    def setupUi(self, UpdaterDialog):
        if not UpdaterDialog.objectName():
            UpdaterDialog.setObjectName(u"UpdaterDialog")
        UpdaterDialog.resize(612, 435)
        self.gridLayout = QGridLayout(UpdaterDialog)
        self.gridLayout.setSpacing(3)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(3, 3, 3, 3)
        self.startFirmware = QPushButton(UpdaterDialog)
        self.startFirmware.setObjectName(u"startFirmware")

        self.gridLayout.addWidget(self.startFirmware, 1, 0, 1, 1)

        self.startSoftware = QPushButton(UpdaterDialog)
        self.startSoftware.setObjectName(u"startSoftware")
        self.startSoftware.setEnabled(True)

        self.gridLayout.addWidget(self.startSoftware, 1, 1, 1, 1)

        self.messageArea = QTextEdit(UpdaterDialog)
        self.messageArea.setObjectName(u"messageArea")

        self.gridLayout.addWidget(self.messageArea, 0, 0, 1, 2)


        self.retranslateUi(UpdaterDialog)

        QMetaObject.connectSlotsByName(UpdaterDialog)
    # setupUi

    def retranslateUi(self, UpdaterDialog):
        UpdaterDialog.setWindowTitle(QCoreApplication.translate("UpdaterDialog", u"Firmware/Software Updater", None))
        self.startFirmware.setText(QCoreApplication.translate("UpdaterDialog", u"Start Firmware Update...", None))
        self.startSoftware.setText(QCoreApplication.translate("UpdaterDialog", u"Start Software Update...", None))
    # retranslateUi

