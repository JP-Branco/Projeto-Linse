# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'AutomatorDialog.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_AutomatorDialog(object):
    def setupUi(self, AutomatorDialog):
        if not AutomatorDialog.objectName():
            AutomatorDialog.setObjectName(u"AutomatorDialog")
        AutomatorDialog.resize(584, 520)
        self.gridLayout = QGridLayout(AutomatorDialog)
        self.gridLayout.setSpacing(3)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(3, 3, 3, 3)
        self.bStop = QPushButton(AutomatorDialog)
        self.bStop.setObjectName(u"bStop")

        self.gridLayout.addWidget(self.bStop, 2, 3, 1, 1)

        self.bLoad = QPushButton(AutomatorDialog)
        self.bLoad.setObjectName(u"bLoad")

        self.gridLayout.addWidget(self.bLoad, 2, 1, 1, 1)

        self.bClear = QPushButton(AutomatorDialog)
        self.bClear.setObjectName(u"bClear")

        self.gridLayout.addWidget(self.bClear, 2, 4, 1, 1)

        self.bStart = QPushButton(AutomatorDialog)
        self.bStart.setObjectName(u"bStart")

        self.gridLayout.addWidget(self.bStart, 2, 2, 1, 1)

        self.messageArea = QTextEdit(AutomatorDialog)
        self.messageArea.setObjectName(u"messageArea")

        self.gridLayout.addWidget(self.messageArea, 1, 1, 1, 4)

        self.tableView = QTableView(AutomatorDialog)
        self.tableView.setObjectName(u"tableView")

        self.gridLayout.addWidget(self.tableView, 0, 1, 1, 4)


        self.retranslateUi(AutomatorDialog)

        QMetaObject.connectSlotsByName(AutomatorDialog)
    # setupUi

    def retranslateUi(self, AutomatorDialog):
        AutomatorDialog.setWindowTitle(QCoreApplication.translate("AutomatorDialog", u"Automator", None))
        self.bStop.setText(QCoreApplication.translate("AutomatorDialog", u"Stop", None))
        self.bLoad.setText(QCoreApplication.translate("AutomatorDialog", u"Load...", None))
        self.bClear.setText(QCoreApplication.translate("AutomatorDialog", u"Clear", None))
        self.bStart.setText(QCoreApplication.translate("AutomatorDialog", u"Start", None))
    # retranslateUi

