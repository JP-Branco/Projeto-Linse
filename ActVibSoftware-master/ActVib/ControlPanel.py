# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ControlPanel.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_ControlForm(object):
    def setupUi(self, ControlForm):
        if not ControlForm.objectName():
            ControlForm.setObjectName(u"ControlForm")
        ControlForm.resize(499, 115)
        self.horizontalLayout = QHBoxLayout(ControlForm)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.Controle = QFrame(ControlForm)
        self.Controle.setObjectName(u"Controle")
        self.Controle.setAutoFillBackground(False)
        self.Controle.setStyleSheet(u"")
        self.Controle.setFrameShape(QFrame.NoFrame)
        self.Controle.setFrameShadow(QFrame.Plain)
        self.Controle.setLineWidth(0)
        self.verticalLayout = QVBoxLayout(self.Controle)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 5, 0, 5)
        self.frame = QFrame(self.Controle)
        self.frame.setObjectName(u"frame")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QSize(0, 0))
        self.frame.setFrameShape(QFrame.NoFrame)
        self.frame.setFrameShadow(QFrame.Raised)
        self.gridLayout = QGridLayout(self.frame)
        self.gridLayout.setSpacing(3)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(3, 0, 5, 0)
        self.line_2 = QFrame(self.frame)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_2, 0, 6, 2, 1)

        self.comboIMURef = QComboBox(self.frame)
        self.comboIMURef.addItem("")
        self.comboIMURef.addItem("")
        self.comboIMURef.addItem("")
        self.comboIMURef.setObjectName(u"comboIMURef")
        sizePolicy.setHeightForWidth(self.comboIMURef.sizePolicy().hasHeightForWidth())
        self.comboIMURef.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.comboIMURef, 0, 8, 1, 1)

        self.label_17 = QLabel(self.frame)
        self.label_17.setObjectName(u"label_17")
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)
        self.label_17.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_17, 1, 7, 1, 1)

        self.label_16 = QLabel(self.frame)
        self.label_16.setObjectName(u"label_16")
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        self.label_16.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_16, 0, 7, 1, 1)

        self.comboRef = QComboBox(self.frame)
        self.comboRef.addItem("")
        self.comboRef.addItem("")
        self.comboRef.addItem("")
        self.comboRef.addItem("")
        self.comboRef.addItem("")
        self.comboRef.addItem("")
        self.comboRef.addItem("")
        self.comboRef.setObjectName(u"comboRef")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.comboRef.sizePolicy().hasHeightForWidth())
        self.comboRef.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.comboRef, 0, 9, 1, 1)

        self.comboControlChannel = QComboBox(self.frame)
        self.comboControlChannel.addItem("")
        self.comboControlChannel.addItem("")
        self.comboControlChannel.addItem("")
        self.comboControlChannel.addItem("")
        self.comboControlChannel.setObjectName(u"comboControlChannel")
        sizePolicy1.setHeightForWidth(self.comboControlChannel.sizePolicy().hasHeightForWidth())
        self.comboControlChannel.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.comboControlChannel, 1, 5, 1, 1)

        self.comboIMUError = QComboBox(self.frame)
        self.comboIMUError.addItem("")
        self.comboIMUError.addItem("")
        self.comboIMUError.addItem("")
        self.comboIMUError.setObjectName(u"comboIMUError")
        sizePolicy.setHeightForWidth(self.comboIMUError.sizePolicy().hasHeightForWidth())
        self.comboIMUError.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.comboIMUError, 1, 8, 1, 1)

        self.comboErro = QComboBox(self.frame)
        self.comboErro.addItem("")
        self.comboErro.addItem("")
        self.comboErro.addItem("")
        self.comboErro.addItem("")
        self.comboErro.addItem("")
        self.comboErro.addItem("")
        self.comboErro.addItem("")
        self.comboErro.setObjectName(u"comboErro")
        sizePolicy1.setHeightForWidth(self.comboErro.sizePolicy().hasHeightForWidth())
        self.comboErro.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.comboErro, 1, 9, 1, 1)

        self.label_48 = QLabel(self.frame)
        self.label_48.setObjectName(u"label_48")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_48.sizePolicy().hasHeightForWidth())
        self.label_48.setSizePolicy(sizePolicy2)
        self.label_48.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_48, 1, 4, 1, 1)

        self.label_24 = QLabel(self.frame)
        self.label_24.setObjectName(u"label_24")
        sizePolicy2.setHeightForWidth(self.label_24.sizePolicy().hasHeightForWidth())
        self.label_24.setSizePolicy(sizePolicy2)
        self.label_24.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_24, 0, 4, 1, 1)

        self.comboPerturbChannel = QComboBox(self.frame)
        self.comboPerturbChannel.addItem("")
        self.comboPerturbChannel.addItem("")
        self.comboPerturbChannel.addItem("")
        self.comboPerturbChannel.addItem("")
        self.comboPerturbChannel.setObjectName(u"comboPerturbChannel")
        sizePolicy1.setHeightForWidth(self.comboPerturbChannel.sizePolicy().hasHeightForWidth())
        self.comboPerturbChannel.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.comboPerturbChannel, 0, 5, 1, 1)

        self.line_4 = QFrame(self.frame)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.VLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_4, 0, 3, 2, 1)

        self.comboCtrlTask = QComboBox(self.frame)
        self.comboCtrlTask.addItem("")
        self.comboCtrlTask.addItem("")
        self.comboCtrlTask.setObjectName(u"comboCtrlTask")

        self.gridLayout.addWidget(self.comboCtrlTask, 1, 1, 1, 1)

        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setBold(False)
        self.label.setFont(font)

        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)


        self.verticalLayout.addWidget(self.frame)

        self.frame_7 = QFrame(self.Controle)
        self.frame_7.setObjectName(u"frame_7")
        sizePolicy.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy)
        self.frame_7.setMinimumSize(QSize(0, 0))
        self.frame_7.setFrameShadow(QFrame.Sunken)
        self.frame_7.setLineWidth(0)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_7)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(3, 1, 5, 1)
        self.label_8 = QLabel(self.frame_7)
        self.label_8.setObjectName(u"label_8")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy3)
        self.label_8.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.label_8.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.label_8)

        self.checkAlgOn = QCheckBox(self.frame_7)
        self.checkAlgOn.setObjectName(u"checkAlgOn")

        self.horizontalLayout_3.addWidget(self.checkAlgOn)

        self.spinTAlgOn = QSpinBox(self.frame_7)
        self.spinTAlgOn.setObjectName(u"spinTAlgOn")
        self.spinTAlgOn.setMinimumSize(QSize(50, 0))

        self.horizontalLayout_3.addWidget(self.spinTAlgOn)

        self.comboAlgoritmo = QComboBox(self.frame_7)
        self.comboAlgoritmo.addItem("")
        self.comboAlgoritmo.addItem("")
        self.comboAlgoritmo.addItem("")
        self.comboAlgoritmo.addItem("")
        self.comboAlgoritmo.setObjectName(u"comboAlgoritmo")
        sizePolicy4 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.comboAlgoritmo.sizePolicy().hasHeightForWidth())
        self.comboAlgoritmo.setSizePolicy(sizePolicy4)

        self.horizontalLayout_3.addWidget(self.comboAlgoritmo)


        self.verticalLayout.addWidget(self.frame_7)

        self.frame_2 = QFrame(self.Controle)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(3, 0, 5, 0)
        self.label_12 = QLabel(self.frame_2)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout_2.addWidget(self.label_12)

        self.spinMemCtrl = QSpinBox(self.frame_2)
        self.spinMemCtrl.setObjectName(u"spinMemCtrl")
        self.spinMemCtrl.setMinimum(1)
        self.spinMemCtrl.setMaximum(1000)
        self.spinMemCtrl.setValue(100)

        self.horizontalLayout_2.addWidget(self.spinMemCtrl)

        self.label_9 = QLabel(self.frame_2)
        self.label_9.setObjectName(u"label_9")
        sizePolicy5 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy5)

        self.horizontalLayout_2.addWidget(self.label_9)

        self.passoCtrl = QLineEdit(self.frame_2)
        self.passoCtrl.setObjectName(u"passoCtrl")

        self.horizontalLayout_2.addWidget(self.passoCtrl)

        self.label_11 = QLabel(self.frame_2)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout_2.addWidget(self.label_11)

        self.normCtrl = QLineEdit(self.frame_2)
        self.normCtrl.setObjectName(u"normCtrl")

        self.horizontalLayout_2.addWidget(self.normCtrl)


        self.verticalLayout.addWidget(self.frame_2)


        self.horizontalLayout.addWidget(self.Controle)


        self.retranslateUi(ControlForm)

        QMetaObject.connectSlotsByName(ControlForm)
    # setupUi

    def retranslateUi(self, ControlForm):
        ControlForm.setWindowTitle(QCoreApplication.translate("ControlForm", u"Form", None))
        self.comboIMURef.setItemText(0, QCoreApplication.translate("ControlForm", u"IMU 1", None))
        self.comboIMURef.setItemText(1, QCoreApplication.translate("ControlForm", u"IMU 2", None))
        self.comboIMURef.setItemText(2, QCoreApplication.translate("ControlForm", u"IMU 3", None))

        self.label_17.setText(QCoreApplication.translate("ControlForm", u"Error:", None))
        self.label_16.setText(QCoreApplication.translate("ControlForm", u"Ref.:", None))
        self.comboRef.setItemText(0, QCoreApplication.translate("ControlForm", u"X Accel", None))
        self.comboRef.setItemText(1, QCoreApplication.translate("ControlForm", u"Y Accel", None))
        self.comboRef.setItemText(2, QCoreApplication.translate("ControlForm", u"Z Accel", None))
        self.comboRef.setItemText(3, QCoreApplication.translate("ControlForm", u"X Gyro", None))
        self.comboRef.setItemText(4, QCoreApplication.translate("ControlForm", u"Y Gyro", None))
        self.comboRef.setItemText(5, QCoreApplication.translate("ControlForm", u"Z Gyro", None))
        self.comboRef.setItemText(6, QCoreApplication.translate("ControlForm", u"Z Accel + X Gyro", None))

        self.comboControlChannel.setItemText(0, QCoreApplication.translate("ControlForm", u"Output 1", None))
        self.comboControlChannel.setItemText(1, QCoreApplication.translate("ControlForm", u"Output 2", None))
        self.comboControlChannel.setItemText(2, QCoreApplication.translate("ControlForm", u"Output 3", None))
        self.comboControlChannel.setItemText(3, QCoreApplication.translate("ControlForm", u"Output 4", None))

        self.comboControlChannel.setCurrentText(QCoreApplication.translate("ControlForm", u"Output 1", None))
        self.comboIMUError.setItemText(0, QCoreApplication.translate("ControlForm", u"IMU 1", None))
        self.comboIMUError.setItemText(1, QCoreApplication.translate("ControlForm", u"IMU 2", None))
        self.comboIMUError.setItemText(2, QCoreApplication.translate("ControlForm", u"IMU 3", None))

        self.comboErro.setItemText(0, QCoreApplication.translate("ControlForm", u"X Accel", None))
        self.comboErro.setItemText(1, QCoreApplication.translate("ControlForm", u"Y Accel", None))
        self.comboErro.setItemText(2, QCoreApplication.translate("ControlForm", u"Z Accel", None))
        self.comboErro.setItemText(3, QCoreApplication.translate("ControlForm", u"X Gyro", None))
        self.comboErro.setItemText(4, QCoreApplication.translate("ControlForm", u"Y Gyro", None))
        self.comboErro.setItemText(5, QCoreApplication.translate("ControlForm", u"Z Gyro", None))
        self.comboErro.setItemText(6, QCoreApplication.translate("ControlForm", u"Z Accel + X Gyro", None))

        self.label_48.setText(QCoreApplication.translate("ControlForm", u"Control:", None))
        self.label_24.setText(QCoreApplication.translate("ControlForm", u"Perturb.:", None))
        self.comboPerturbChannel.setItemText(0, QCoreApplication.translate("ControlForm", u"Output 1", None))
        self.comboPerturbChannel.setItemText(1, QCoreApplication.translate("ControlForm", u"Output 2", None))
        self.comboPerturbChannel.setItemText(2, QCoreApplication.translate("ControlForm", u"Output 3", None))
        self.comboPerturbChannel.setItemText(3, QCoreApplication.translate("ControlForm", u"Output 4", None))

        self.comboCtrlTask.setItemText(0, QCoreApplication.translate("ControlForm", u"Control", None))
        self.comboCtrlTask.setItemText(1, QCoreApplication.translate("ControlForm", u"Path Modelling", None))

        self.label.setText(QCoreApplication.translate("ControlForm", u" Task:", None))
        self.label_8.setText(QCoreApplication.translate("ControlForm", u"Algorithm:", None))
        self.checkAlgOn.setText(QCoreApplication.translate("ControlForm", u"On \u2265", None))
        self.comboAlgoritmo.setItemText(0, QCoreApplication.translate("ControlForm", u"FxNLMS", None))
        self.comboAlgoritmo.setItemText(1, QCoreApplication.translate("ControlForm", u"FxNLMS with full buffers", None))
        self.comboAlgoritmo.setItemText(2, QCoreApplication.translate("ControlForm", u"CVA-FxNLMS", None))
        self.comboAlgoritmo.setItemText(3, QCoreApplication.translate("ControlForm", u"CVA-FxNLMS with full buffers", None))

        self.label_12.setText(QCoreApplication.translate("ControlForm", u"Memory size:", None))
        self.label_9.setText(QCoreApplication.translate("ControlForm", u"Step size:", None))
        self.passoCtrl.setText(QCoreApplication.translate("ControlForm", u"0.01", None))
        self.label_11.setText(QCoreApplication.translate("ControlForm", u"Penalization:", None))
        self.normCtrl.setText(QCoreApplication.translate("ControlForm", u"1e-4", None))
    # retranslateUi

