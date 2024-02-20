# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainFigContextMenu.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainForm(object):
    def setupUi(self, MainForm):
        if not MainForm.objectName():
            MainForm.setObjectName(u"MainForm")
        MainForm.resize(369, 170)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainForm.sizePolicy().hasHeightForWidth())
        MainForm.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(MainForm)
        self.gridLayout.setSpacing(3)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(3, 3, 3, 3)
        self.label_3 = QLabel(MainForm)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 7, 1, 1, 1)

        self.label_2 = QLabel(MainForm)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 3, 1, 1, 1)

        self.frame = QFrame(MainForm)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.NoFrame)
        self.frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setSpacing(1)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(1, 0, 1, 0)
        self.ylim1 = QLineEdit(self.frame)
        self.ylim1.setObjectName(u"ylim1")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.ylim1.sizePolicy().hasHeightForWidth())
        self.ylim1.setSizePolicy(sizePolicy1)
        self.ylim1.setMinimumSize(QSize(50, 0))
        self.ylim1.setMaximumSize(QSize(1677215, 16777215))

        self.horizontalLayout.addWidget(self.ylim1)

        self.label_4 = QLabel(self.frame)
        self.label_4.setObjectName(u"label_4")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy2)
        self.label_4.setMinimumSize(QSize(0, 0))
        self.label_4.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.label_4)

        self.ylim2 = QLineEdit(self.frame)
        self.ylim2.setObjectName(u"ylim2")
        sizePolicy3 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.ylim2.sizePolicy().hasHeightForWidth())
        self.ylim2.setSizePolicy(sizePolicy3)
        self.ylim2.setMinimumSize(QSize(50, 0))
        self.ylim2.setMaximumSize(QSize(1677215, 16777215))

        self.horizontalLayout.addWidget(self.ylim2)


        self.gridLayout.addWidget(self.frame, 7, 2, 1, 4)

        self.comboSensor = QComboBox(MainForm)
        self.comboSensor.addItem("")
        self.comboSensor.addItem("")
        self.comboSensor.addItem("")
        self.comboSensor.addItem("")
        self.comboSensor.addItem("")
        self.comboSensor.addItem("")
        self.comboSensor.addItem("")
        self.comboSensor.setObjectName(u"comboSensor")

        self.gridLayout.addWidget(self.comboSensor, 2, 2, 1, 1)

        self.checkZ = QCheckBox(MainForm)
        self.checkZ.setObjectName(u"checkZ")

        self.gridLayout.addWidget(self.checkZ, 2, 5, 1, 1)

        self.spinTWindow = QSpinBox(MainForm)
        self.spinTWindow.setObjectName(u"spinTWindow")
        self.spinTWindow.setMinimum(1)
        self.spinTWindow.setMaximum(10)

        self.gridLayout.addWidget(self.spinTWindow, 3, 2, 1, 1)

        self.checkX = QCheckBox(MainForm)
        self.checkX.setObjectName(u"checkX")

        self.gridLayout.addWidget(self.checkX, 2, 3, 1, 1)

        self.labelSensor = QLabel(MainForm)
        self.labelSensor.setObjectName(u"labelSensor")

        self.gridLayout.addWidget(self.labelSensor, 2, 1, 1, 1)

        self.checkY = QCheckBox(MainForm)
        self.checkY.setObjectName(u"checkY")

        self.gridLayout.addWidget(self.checkY, 2, 4, 1, 1)

        self.bviewall = QPushButton(MainForm)
        self.bviewall.setObjectName(u"bviewall")
        self.bviewall.setFlat(False)

        self.gridLayout.addWidget(self.bviewall, 1, 1, 1, 5)


        self.retranslateUi(MainForm)

        QMetaObject.connectSlotsByName(MainForm)
    # setupUi

    def retranslateUi(self, MainForm):
        MainForm.setWindowTitle(QCoreApplication.translate("MainForm", u"Form", None))
        self.label_3.setText(QCoreApplication.translate("MainForm", u"Y Axis:", None))
        self.label_2.setText(QCoreApplication.translate("MainForm", u"Window:", None))
        self.label_4.setText(QCoreApplication.translate("MainForm", u"to", None))
        self.comboSensor.setItemText(0, QCoreApplication.translate("MainForm", u"Disabled", None))
        self.comboSensor.setItemText(1, QCoreApplication.translate("MainForm", u"IMU1 Accel", None))
        self.comboSensor.setItemText(2, QCoreApplication.translate("MainForm", u"IMU2 Accel", None))
        self.comboSensor.setItemText(3, QCoreApplication.translate("MainForm", u"IMU3 Accel", None))
        self.comboSensor.setItemText(4, QCoreApplication.translate("MainForm", u"IMU1 Gyro", None))
        self.comboSensor.setItemText(5, QCoreApplication.translate("MainForm", u"IMU2 Gyro", None))
        self.comboSensor.setItemText(6, QCoreApplication.translate("MainForm", u"IMU3 Gyro", None))

        self.checkZ.setText(QCoreApplication.translate("MainForm", u"Z", None))
        self.spinTWindow.setSuffix(QCoreApplication.translate("MainForm", u" s", None))
        self.checkX.setText(QCoreApplication.translate("MainForm", u"X", None))
        self.labelSensor.setText(QCoreApplication.translate("MainForm", u"Sensor:", None))
        self.checkY.setText(QCoreApplication.translate("MainForm", u"Y", None))
        self.bviewall.setText(QCoreApplication.translate("MainForm", u"View All", None))
    # retranslateUi

