# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ADCPanel.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_ADCForm(object):
    def setupUi(self, ADCForm):
        if not ADCForm.objectName():
            ADCForm.setObjectName(u"ADCForm")
        ADCForm.resize(609, 56)
        self.verticalLayout = QVBoxLayout(ADCForm)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(2, 0, 2, 0)
        self.adcframe = QFrame(ADCForm)
        self.adcframe.setObjectName(u"adcframe")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.adcframe.sizePolicy().hasHeightForWidth())
        self.adcframe.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(self.adcframe)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setHorizontalSpacing(5)
        self.gridLayout.setVerticalSpacing(2)
        self.gridLayout.setContentsMargins(10, 2, 8, 4)
        self.comboADCRange = QComboBox(self.adcframe)
        self.comboADCRange.addItem("")
        self.comboADCRange.addItem("")
        self.comboADCRange.addItem("")
        self.comboADCRange.addItem("")
        self.comboADCRange.addItem("")
        self.comboADCRange.addItem("")
        self.comboADCRange.setObjectName(u"comboADCRange")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.comboADCRange.sizePolicy().hasHeightForWidth())
        self.comboADCRange.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.comboADCRange, 0, 7, 1, 2)

        self.comboRate1115 = QComboBox(self.adcframe)
        self.comboRate1115.addItem("")
        self.comboRate1115.addItem("")
        self.comboRate1115.addItem("")
        self.comboRate1115.addItem("")
        self.comboRate1115.addItem("")
        self.comboRate1115.addItem("")
        self.comboRate1115.addItem("")
        self.comboRate1115.addItem("")
        self.comboRate1115.setObjectName(u"comboRate1115")
        self.comboRate1115.setEnabled(True)
        sizePolicy.setHeightForWidth(self.comboRate1115.sizePolicy().hasHeightForWidth())
        self.comboRate1115.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.comboRate1115, 2, 7, 1, 1)

        self.comboRate1015 = QComboBox(self.adcframe)
        self.comboRate1015.addItem("")
        self.comboRate1015.addItem("")
        self.comboRate1015.addItem("")
        self.comboRate1015.addItem("")
        self.comboRate1015.addItem("")
        self.comboRate1015.addItem("")
        self.comboRate1015.addItem("")
        self.comboRate1015.setObjectName(u"comboRate1015")
        self.comboRate1015.setEnabled(False)
        sizePolicy.setHeightForWidth(self.comboRate1015.sizePolicy().hasHeightForWidth())
        self.comboRate1015.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.comboRate1015, 2, 8, 1, 1)

        self.radioADC4 = QRadioButton(self.adcframe)
        self.radioADC4.setObjectName(u"radioADC4")

        self.gridLayout.addWidget(self.radioADC4, 2, 4, 1, 1)

        self.label = QLabel(self.adcframe)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.comboADCModel = QComboBox(self.adcframe)
        self.comboADCModel.addItem("")
        self.comboADCModel.addItem("")
        self.comboADCModel.setObjectName(u"comboADCModel")
        sizePolicy1.setHeightForWidth(self.comboADCModel.sizePolicy().hasHeightForWidth())
        self.comboADCModel.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.comboADCModel, 2, 0, 1, 1)

        self.label_30 = QLabel(self.adcframe)
        self.label_30.setObjectName(u"label_30")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_30.sizePolicy().hasHeightForWidth())
        self.label_30.setSizePolicy(sizePolicy2)
        self.label_30.setLayoutDirection(Qt.LeftToRight)
        self.label_30.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_30, 0, 6, 1, 1)

        self.line_2 = QFrame(self.adcframe)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_2, 0, 5, 3, 1)

        self.radioADC1 = QRadioButton(self.adcframe)
        self.radioADC1.setObjectName(u"radioADC1")
        self.radioADC1.setAutoRepeat(False)
        self.radioADC1.setAutoExclusive(True)

        self.gridLayout.addWidget(self.radioADC1, 0, 3, 1, 1)

        self.label_33 = QLabel(self.adcframe)
        self.label_33.setObjectName(u"label_33")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_33.sizePolicy().hasHeightForWidth())
        self.label_33.setSizePolicy(sizePolicy3)
        self.label_33.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_33, 2, 6, 1, 1)

        self.line = QFrame(self.adcframe)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 0, 1, 3, 1)

        self.radioOff = QRadioButton(self.adcframe)
        self.radioOff.setObjectName(u"radioOff")
        self.radioOff.setChecked(True)

        self.gridLayout.addWidget(self.radioOff, 0, 2, 1, 1)

        self.radioADC3 = QRadioButton(self.adcframe)
        self.radioADC3.setObjectName(u"radioADC3")
        self.radioADC3.setAutoExclusive(True)

        self.gridLayout.addWidget(self.radioADC3, 0, 4, 1, 1)

        self.radioADC2 = QRadioButton(self.adcframe)
        self.radioADC2.setObjectName(u"radioADC2")

        self.gridLayout.addWidget(self.radioADC2, 2, 3, 1, 1)


        self.verticalLayout.addWidget(self.adcframe)


        self.retranslateUi(ADCForm)

        QMetaObject.connectSlotsByName(ADCForm)
    # setupUi

    def retranslateUi(self, ADCForm):
        ADCForm.setWindowTitle(QCoreApplication.translate("ADCForm", u"Form", None))
        self.comboADCRange.setItemText(0, QCoreApplication.translate("ADCForm", u"\u00b16.144 V", None))
        self.comboADCRange.setItemText(1, QCoreApplication.translate("ADCForm", u"\u00b14.096 V", None))
        self.comboADCRange.setItemText(2, QCoreApplication.translate("ADCForm", u"\u00b12.048 V", None))
        self.comboADCRange.setItemText(3, QCoreApplication.translate("ADCForm", u"\u00b11.024 V", None))
        self.comboADCRange.setItemText(4, QCoreApplication.translate("ADCForm", u"\u00b10.512 V", None))
        self.comboADCRange.setItemText(5, QCoreApplication.translate("ADCForm", u"\u00b10.256 V", None))

        self.comboRate1115.setItemText(0, QCoreApplication.translate("ADCForm", u"8 SPS", None))
        self.comboRate1115.setItemText(1, QCoreApplication.translate("ADCForm", u"16 SPS", None))
        self.comboRate1115.setItemText(2, QCoreApplication.translate("ADCForm", u"32 SPS", None))
        self.comboRate1115.setItemText(3, QCoreApplication.translate("ADCForm", u"64 SPS", None))
        self.comboRate1115.setItemText(4, QCoreApplication.translate("ADCForm", u"128 SPS", None))
        self.comboRate1115.setItemText(5, QCoreApplication.translate("ADCForm", u"250 SPS", None))
        self.comboRate1115.setItemText(6, QCoreApplication.translate("ADCForm", u"475 SPS", None))
        self.comboRate1115.setItemText(7, QCoreApplication.translate("ADCForm", u"860 SPS", None))

        self.comboRate1115.setCurrentText(QCoreApplication.translate("ADCForm", u"860 SPS", None))
        self.comboRate1015.setItemText(0, QCoreApplication.translate("ADCForm", u"128 SPS", None))
        self.comboRate1015.setItemText(1, QCoreApplication.translate("ADCForm", u"250 SPS", None))
        self.comboRate1015.setItemText(2, QCoreApplication.translate("ADCForm", u"490 SPS", None))
        self.comboRate1015.setItemText(3, QCoreApplication.translate("ADCForm", u"920 SPS", None))
        self.comboRate1015.setItemText(4, QCoreApplication.translate("ADCForm", u"1600 SPS", None))
        self.comboRate1015.setItemText(5, QCoreApplication.translate("ADCForm", u"2400 SPS", None))
        self.comboRate1015.setItemText(6, QCoreApplication.translate("ADCForm", u"3300 SPS", None))

        self.comboRate1015.setCurrentText(QCoreApplication.translate("ADCForm", u"1600 SPS", None))
        self.radioADC4.setText(QCoreApplication.translate("ADCForm", u"Channel 4", None))
        self.label.setText(QCoreApplication.translate("ADCForm", u"Model:", None))
        self.comboADCModel.setItemText(0, QCoreApplication.translate("ADCForm", u"ADS1115", None))
        self.comboADCModel.setItemText(1, QCoreApplication.translate("ADCForm", u"ADS1015", None))

        self.label_30.setText(QCoreApplication.translate("ADCForm", u"Range:", None))
        self.radioADC1.setText(QCoreApplication.translate("ADCForm", u"Channel 1", None))
        self.label_33.setText(QCoreApplication.translate("ADCForm", u"Rate:", None))
        self.radioOff.setText(QCoreApplication.translate("ADCForm", u"Off", None))
        self.radioADC3.setText(QCoreApplication.translate("ADCForm", u"Channel 3", None))
        self.radioADC2.setText(QCoreApplication.translate("ADCForm", u"Channel 2", None))
    # retranslateUi

