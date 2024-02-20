# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'IMUPanel.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_IMUPanel(object):
    def setupUi(self, IMUPanel):
        if not IMUPanel.objectName():
            IMUPanel.setObjectName(u"IMUPanel")
        IMUPanel.resize(566, 53)
        self.verticalLayout = QVBoxLayout(IMUPanel)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.imuframe = QFrame(IMUPanel)
        self.imuframe.setObjectName(u"imuframe")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imuframe.sizePolicy().hasHeightForWidth())
        self.imuframe.setSizePolicy(sizePolicy)
        self.imuframe.setFrameShape(QFrame.NoFrame)
        self.imuframe.setFrameShadow(QFrame.Plain)
        self.imuframe.setLineWidth(0)
        self.gridLayout = QGridLayout(self.imuframe)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setHorizontalSpacing(6)
        self.gridLayout.setVerticalSpacing(1)
        self.gridLayout.setContentsMargins(3, 2, 3, 2)
        self.comboAddress = QComboBox(self.imuframe)
        self.comboAddress.addItem("")
        self.comboAddress.addItem("")
        self.comboAddress.addItem("")
        self.comboAddress.addItem("")
        self.comboAddress.addItem("")
        self.comboAddress.addItem("")
        self.comboAddress.setObjectName(u"comboAddress")
        sizePolicy.setHeightForWidth(self.comboAddress.sizePolicy().hasHeightForWidth())
        self.comboAddress.setSizePolicy(sizePolicy)
        self.comboAddress.setMinimumSize(QSize(0, 0))
        self.comboAddress.setBaseSize(QSize(0, 0))

        self.gridLayout.addWidget(self.comboAddress, 3, 1, 1, 1)

        self.label = QLabel(self.imuframe)
        self.label.setObjectName(u"label")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.label, 3, 0, 1, 1)

        self.label_14 = QLabel(self.imuframe)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_14, 1, 6, 1, 1)

        self.comboType = QComboBox(self.imuframe)
        self.comboType.addItem("")
        self.comboType.addItem("")
        self.comboType.addItem("")
        self.comboType.setObjectName(u"comboType")
        sizePolicy.setHeightForWidth(self.comboType.sizePolicy().hasHeightForWidth())
        self.comboType.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.comboType, 1, 0, 1, 2)

        self.label_15 = QLabel(self.imuframe)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_15, 3, 6, 1, 1)

        self.label_13 = QLabel(self.imuframe)
        self.label_13.setObjectName(u"label_13")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy2)
        self.label_13.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_13, 3, 3, 1, 1)

        self.line_2 = QFrame(self.imuframe)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_2, 1, 2, 4, 1)

        self.line = QFrame(self.imuframe)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 1, 5, 4, 1)

        self.comboAccRange = QComboBox(self.imuframe)
        self.comboAccRange.addItem("")
        self.comboAccRange.addItem("")
        self.comboAccRange.addItem("")
        self.comboAccRange.addItem("")
        self.comboAccRange.setObjectName(u"comboAccRange")
        sizePolicy.setHeightForWidth(self.comboAccRange.sizePolicy().hasHeightForWidth())
        self.comboAccRange.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.comboAccRange, 3, 4, 1, 1)

        self.comboBus = QComboBox(self.imuframe)
        self.comboBus.addItem("")
        self.comboBus.addItem("")
        self.comboBus.addItem("")
        self.comboBus.setObjectName(u"comboBus")
        sizePolicy.setHeightForWidth(self.comboBus.sizePolicy().hasHeightForWidth())
        self.comboBus.setSizePolicy(sizePolicy)
        self.comboBus.setBaseSize(QSize(30, 0))

        self.gridLayout.addWidget(self.comboBus, 1, 4, 1, 1)

        self.label_2 = QLabel(self.imuframe)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 3, 1, 1)

        self.comboGyroRange = QComboBox(self.imuframe)
        self.comboGyroRange.addItem("")
        self.comboGyroRange.addItem("")
        self.comboGyroRange.addItem("")
        self.comboGyroRange.addItem("")
        self.comboGyroRange.addItem("")
        self.comboGyroRange.setObjectName(u"comboGyroRange")
        sizePolicy.setHeightForWidth(self.comboGyroRange.sizePolicy().hasHeightForWidth())
        self.comboGyroRange.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.comboGyroRange, 1, 7, 1, 1)

        self.comboFilter2 = QComboBox(self.imuframe)
        self.comboFilter2.addItem("")
        self.comboFilter2.addItem("")
        self.comboFilter2.addItem("")
        self.comboFilter2.addItem("")
        self.comboFilter2.addItem("")
        self.comboFilter2.setObjectName(u"comboFilter2")
        sizePolicy.setHeightForWidth(self.comboFilter2.sizePolicy().hasHeightForWidth())
        self.comboFilter2.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.comboFilter2, 3, 7, 1, 1)


        self.verticalLayout.addWidget(self.imuframe)

        QWidget.setTabOrder(self.comboType, self.comboAddress)
        QWidget.setTabOrder(self.comboAddress, self.comboBus)
        QWidget.setTabOrder(self.comboBus, self.comboAccRange)
        QWidget.setTabOrder(self.comboAccRange, self.comboGyroRange)

        self.retranslateUi(IMUPanel)

        QMetaObject.connectSlotsByName(IMUPanel)
    # setupUi

    def retranslateUi(self, IMUPanel):
        IMUPanel.setWindowTitle(QCoreApplication.translate("IMUPanel", u"Form", None))
        self.comboAddress.setItemText(0, QCoreApplication.translate("IMUPanel", u"0x68", None))
        self.comboAddress.setItemText(1, QCoreApplication.translate("IMUPanel", u"0x69", None))
        self.comboAddress.setItemText(2, QCoreApplication.translate("IMUPanel", u"0x6A", None))
        self.comboAddress.setItemText(3, QCoreApplication.translate("IMUPanel", u"0x6B", None))
        self.comboAddress.setItemText(4, QCoreApplication.translate("IMUPanel", u"CS5", None))
        self.comboAddress.setItemText(5, QCoreApplication.translate("IMUPanel", u"CS33", None))

        self.label.setText(QCoreApplication.translate("IMUPanel", u"Addr:", None))
        self.label_14.setText(QCoreApplication.translate("IMUPanel", u"GyroR:", None))
        self.comboType.setItemText(0, QCoreApplication.translate("IMUPanel", u"Disabled", None))
        self.comboType.setItemText(1, QCoreApplication.translate("IMUPanel", u"MPU6050", None))
        self.comboType.setItemText(2, QCoreApplication.translate("IMUPanel", u"LSM6DS3", None))

        self.label_15.setText(QCoreApplication.translate("IMUPanel", u"Filter:", None))
        self.label_13.setText(QCoreApplication.translate("IMUPanel", u"AccR:", None))
        self.comboAccRange.setItemText(0, QCoreApplication.translate("IMUPanel", u"\u00b12g (\u00b119,6m/s\u00b2)", None))
        self.comboAccRange.setItemText(1, QCoreApplication.translate("IMUPanel", u"\u00b14g (\u00b139,2m/s\u00b2)", None))
        self.comboAccRange.setItemText(2, QCoreApplication.translate("IMUPanel", u"\u00b18g (\u00b178,4m/s\u00b2)", None))
        self.comboAccRange.setItemText(3, QCoreApplication.translate("IMUPanel", u"\u00b116g (\u00b1156,9m/s\u00b2)", None))

        self.comboBus.setItemText(0, QCoreApplication.translate("IMUPanel", u"I2C-1", None))
        self.comboBus.setItemText(1, QCoreApplication.translate("IMUPanel", u"I2C-2", None))
        self.comboBus.setItemText(2, QCoreApplication.translate("IMUPanel", u"VSPI", None))

        self.label_2.setText(QCoreApplication.translate("IMUPanel", u"Bus:", None))
        self.comboGyroRange.setItemText(0, QCoreApplication.translate("IMUPanel", u"\u00b1125\u00ba/s", None))
        self.comboGyroRange.setItemText(1, QCoreApplication.translate("IMUPanel", u"\u00b1250\u00ba/s", None))
        self.comboGyroRange.setItemText(2, QCoreApplication.translate("IMUPanel", u"\u00b1500\u00ba/s", None))
        self.comboGyroRange.setItemText(3, QCoreApplication.translate("IMUPanel", u"\u00b11000\u00ba/s", None))
        self.comboGyroRange.setItemText(4, QCoreApplication.translate("IMUPanel", u"\u00b12000\u00ba/s", None))

        self.comboFilter2.setItemText(0, QCoreApplication.translate("IMUPanel", u"Auto", None))
        self.comboFilter2.setItemText(1, QCoreApplication.translate("IMUPanel", u"400Hz", None))
        self.comboFilter2.setItemText(2, QCoreApplication.translate("IMUPanel", u"200Hz", None))
        self.comboFilter2.setItemText(3, QCoreApplication.translate("IMUPanel", u"100Hz", None))
        self.comboFilter2.setItemText(4, QCoreApplication.translate("IMUPanel", u"50Hz", None))

    # retranslateUi

