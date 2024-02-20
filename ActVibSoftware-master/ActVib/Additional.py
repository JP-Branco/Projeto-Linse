# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Additional.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_AdditionalConfigDialog(object):
    def setupUi(self, AdditionalConfigDialog):
        if not AdditionalConfigDialog.objectName():
            AdditionalConfigDialog.setObjectName(u"AdditionalConfigDialog")
        AdditionalConfigDialog.resize(600, 390)
        self.gridLayout_2 = QGridLayout(AdditionalConfigDialog)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(3, 3, 3, 3)
        self.groupBox = QGroupBox(AdditionalConfigDialog)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setAutoFillBackground(False)
        self.label.setFrameShape(QFrame.Box)
        self.label.setFrameShadow(QFrame.Raised)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)

        self.textPoly1 = QLineEdit(self.groupBox)
        self.textPoly1.setObjectName(u"textPoly1")

        self.gridLayout.addWidget(self.textPoly1, 1, 1, 1, 1)

        self.checkEnable3 = QCheckBox(self.groupBox)
        self.checkEnable3.setObjectName(u"checkEnable3")

        self.gridLayout.addWidget(self.checkEnable3, 3, 0, 1, 1)

        self.checkEnable1 = QCheckBox(self.groupBox)
        self.checkEnable1.setObjectName(u"checkEnable1")

        self.gridLayout.addWidget(self.checkEnable1, 1, 0, 1, 1)

        self.checkEnable2 = QCheckBox(self.groupBox)
        self.checkEnable2.setObjectName(u"checkEnable2")

        self.gridLayout.addWidget(self.checkEnable2, 2, 0, 1, 1)

        self.textPoly3 = QLineEdit(self.groupBox)
        self.textPoly3.setObjectName(u"textPoly3")

        self.gridLayout.addWidget(self.textPoly3, 3, 1, 1, 1)

        self.textPoly2 = QLineEdit(self.groupBox)
        self.textPoly2.setObjectName(u"textPoly2")

        self.gridLayout.addWidget(self.textPoly2, 2, 1, 1, 1)

        self.checkEnable4 = QCheckBox(self.groupBox)
        self.checkEnable4.setObjectName(u"checkEnable4")

        self.gridLayout.addWidget(self.checkEnable4, 4, 0, 1, 1)

        self.textPoly4 = QLineEdit(self.groupBox)
        self.textPoly4.setObjectName(u"textPoly4")

        self.gridLayout.addWidget(self.textPoly4, 4, 1, 1, 1)


        self.gridLayout_2.addWidget(self.groupBox, 0, 0, 1, 2)

        self.statusLabel = QLabel(AdditionalConfigDialog)
        self.statusLabel.setObjectName(u"statusLabel")

        self.gridLayout_2.addWidget(self.statusLabel, 7, 0, 1, 2)

        self.bSave = QPushButton(AdditionalConfigDialog)
        self.bSave.setObjectName(u"bSave")

        self.gridLayout_2.addWidget(self.bSave, 6, 0, 1, 2)

        self.groupBox_2 = QGroupBox(AdditionalConfigDialog)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 6, -1, 6)
        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)

        self.horizontalLayout.addWidget(self.label_2)

        self.spinFusionW1 = QDoubleSpinBox(self.groupBox_2)
        self.spinFusionW1.setObjectName(u"spinFusionW1")
        self.spinFusionW1.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.spinFusionW1.setDecimals(3)
        self.spinFusionW1.setMaximum(1000.000000000000000)
        self.spinFusionW1.setValue(0.500000000000000)

        self.horizontalLayout.addWidget(self.spinFusionW1)

        self.line = QFrame(self.groupBox_2)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line)

        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)

        self.horizontalLayout.addWidget(self.label_3)

        self.spinFusionW2 = QDoubleSpinBox(self.groupBox_2)
        self.spinFusionW2.setObjectName(u"spinFusionW2")
        self.spinFusionW2.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.spinFusionW2.setDecimals(3)
        self.spinFusionW2.setMaximum(1000.000000000000000)
        self.spinFusionW2.setValue(0.500000000000000)

        self.horizontalLayout.addWidget(self.spinFusionW2)


        self.gridLayout_2.addWidget(self.groupBox_2, 5, 0, 1, 2)


        self.retranslateUi(AdditionalConfigDialog)

        QMetaObject.connectSlotsByName(AdditionalConfigDialog)
    # setupUi

    def retranslateUi(self, AdditionalConfigDialog):
        AdditionalConfigDialog.setWindowTitle(QCoreApplication.translate("AdditionalConfigDialog", u"Additional System Configuration", None))
        self.groupBox.setTitle(QCoreApplication.translate("AdditionalConfigDialog", u"Output Predistortion", None))
        self.label.setText(QCoreApplication.translate("AdditionalConfigDialog", u"<html><head/><body><p align=\"center\"><span style=\" font-size:9pt;\"><br/>Input must be an array containing the polynomial coefficients </span><span style=\" font-size:9pt; font-weight:600;\">[a(k) a(k-1) ... a(0)]</span><span style=\" font-size:9pt;\">,</span></p><p align=\"center\"><span style=\" font-size:9pt;\">where </span><span style=\" font-size:9pt; font-weight:600;\">y = a(k)*x</span><span style=\" font-size:9pt; font-weight:600; vertical-align:super;\">k</span><span style=\" font-size:9pt; font-weight:600;\"> + a(k-1)*x</span><span style=\" font-size:9pt; font-weight:600; vertical-align:super;\">k-1</span><span style=\" font-size:9pt; font-weight:600;\"> + ... + a(1)*x + a(0)</span><span style=\" font-size:9pt;\">. Thus, for a </span><span style=\" font-size:9pt; font-weight:600;\">K-th</span><span style=\" font-size:9pt;\"> order </span></p><p align=\"center\"><span style=\" font-size:9pt;\">polynomial, </span><span style=\" font-size:9pt; font-weight:600;\">K+1</span><span style=\" font-size:9pt;"
                        "\"> coefficients are required. Maximum order is </span><span style=\" font-size:9pt; font-weight:600;\">9</span><span style=\" font-size:9pt;\">.<br/></span></p></body></html>", None))
        self.textPoly1.setText(QCoreApplication.translate("AdditionalConfigDialog", u"[0.0  0.0  0.0  0.0  0.0  0.0  0.0]", None))
        self.checkEnable3.setText(QCoreApplication.translate("AdditionalConfigDialog", u"Enable for Output 3:", None))
        self.checkEnable1.setText(QCoreApplication.translate("AdditionalConfigDialog", u"Enable for Output 1:", None))
        self.checkEnable2.setText(QCoreApplication.translate("AdditionalConfigDialog", u"Enable for Output 2:", None))
        self.textPoly3.setText(QCoreApplication.translate("AdditionalConfigDialog", u"[0.0  0.0  0.0  0.0  0.0  0.0  0.0]", None))
        self.textPoly2.setText(QCoreApplication.translate("AdditionalConfigDialog", u"[0.0  0.0  0.0  0.0  0.0  0.0  0.0]", None))
        self.checkEnable4.setText(QCoreApplication.translate("AdditionalConfigDialog", u"Enable for Output 4:", None))
        self.textPoly4.setText(QCoreApplication.translate("AdditionalConfigDialog", u"[0.0  0.0  0.0  0.0  0.0  0.0  0.0]", None))
        self.statusLabel.setText("")
        self.bSave.setText(QCoreApplication.translate("AdditionalConfigDialog", u"Check and Save", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("AdditionalConfigDialog", u"Sensor Fusion", None))
        self.label_2.setText(QCoreApplication.translate("AdditionalConfigDialog", u"Weight for Sensor 1:", None))
        self.label_3.setText(QCoreApplication.translate("AdditionalConfigDialog", u"Weight for Sensor 2:", None))
    # retranslateUi

