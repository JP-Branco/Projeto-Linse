# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'GeneratorPanel.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_GeneratorPanel(object):
    def setupUi(self, GeneratorPanel):
        if not GeneratorPanel.objectName():
            GeneratorPanel.setObjectName(u"GeneratorPanel")
        GeneratorPanel.resize(548, 55)
        self.verticalLayout = QVBoxLayout(GeneratorPanel)
        self.verticalLayout.setSpacing(3)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(1, 1, 1, 1)
        self.frame_4 = QFrame(GeneratorPanel)
        self.frame_4.setObjectName(u"frame_4")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setFrameShape(QFrame.NoFrame)
        self.frame_4.setFrameShadow(QFrame.Plain)
        self.frame_4.setLineWidth(0)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_6.setSpacing(5)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(2, 2, 2, 2)
        self.checkEnable = QCheckBox(self.frame_4)
        self.checkEnable.setObjectName(u"checkEnable")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.checkEnable.sizePolicy().hasHeightForWidth())
        self.checkEnable.setSizePolicy(sizePolicy1)
        self.checkEnable.setCheckable(True)
        self.checkEnable.setChecked(True)

        self.horizontalLayout_6.addWidget(self.checkEnable)

        self.comboType = QComboBox(self.frame_4)
        self.comboType.addItem("")
        self.comboType.addItem("")
        self.comboType.addItem("")
        self.comboType.addItem("")
        self.comboType.addItem("")
        self.comboType.setObjectName(u"comboType")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.comboType.sizePolicy().hasHeightForWidth())
        self.comboType.setSizePolicy(sizePolicy2)

        self.horizontalLayout_6.addWidget(self.comboType)

        self.label_2 = QLabel(self.frame_4)
        self.label_2.setObjectName(u"label_2")
        sizePolicy3 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy3)

        self.horizontalLayout_6.addWidget(self.label_2)

        self.spinAmpl = QDoubleSpinBox(self.frame_4)
        self.spinAmpl.setObjectName(u"spinAmpl")
        sizePolicy4 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.spinAmpl.sizePolicy().hasHeightForWidth())
        self.spinAmpl.setSizePolicy(sizePolicy4)
        self.spinAmpl.setDecimals(3)
        self.spinAmpl.setMaximum(1.000000000000000)
        self.spinAmpl.setSingleStep(0.100000000000000)
        self.spinAmpl.setValue(0.000000000000000)

        self.horizontalLayout_6.addWidget(self.spinAmpl)

        self.label_3 = QLabel(self.frame_4)
        self.label_3.setObjectName(u"label_3")
        sizePolicy3.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy3)

        self.horizontalLayout_6.addWidget(self.label_3)

        self.spinFreq = QDoubleSpinBox(self.frame_4)
        self.spinFreq.setObjectName(u"spinFreq")
        sizePolicy4.setHeightForWidth(self.spinFreq.sizePolicy().hasHeightForWidth())
        self.spinFreq.setSizePolicy(sizePolicy4)
        self.spinFreq.setMaximum(125.000000000000000)
        self.spinFreq.setSingleStep(0.100000000000000)
        self.spinFreq.setValue(10.000000000000000)

        self.horizontalLayout_6.addWidget(self.spinFreq)

        self.label = QLabel(self.frame_4)
        self.label.setObjectName(u"label")

        self.horizontalLayout_6.addWidget(self.label)

        self.spinDCLevel = QSpinBox(self.frame_4)
        self.spinDCLevel.setObjectName(u"spinDCLevel")
        self.spinDCLevel.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.spinDCLevel.setMaximum(2048)
        self.spinDCLevel.setValue(1668)

        self.horizontalLayout_6.addWidget(self.spinDCLevel)


        self.verticalLayout.addWidget(self.frame_4)

        self.extrasCanal1 = QFrame(GeneratorPanel)
        self.extrasCanal1.setObjectName(u"extrasCanal1")
        sizePolicy.setHeightForWidth(self.extrasCanal1.sizePolicy().hasHeightForWidth())
        self.extrasCanal1.setSizePolicy(sizePolicy)
        self.extrasCanal1.setMinimumSize(QSize(0, 0))
        self.extrasCanal1.setAutoFillBackground(False)
        self.extrasCanal1.setStyleSheet(u"")
        self.extrasCanal1.setFrameShape(QFrame.NoFrame)
        self.extrasCanal1.setFrameShadow(QFrame.Plain)
        self.extrasCanal1.setLineWidth(0)
        self.horizontalLayout_2 = QHBoxLayout(self.extrasCanal1)
        self.horizontalLayout_2.setSpacing(3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(2, 1, 0, 1)
        self.label_26 = QLabel(self.extrasCanal1)
        self.label_26.setObjectName(u"label_26")
        sizePolicy5 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.label_26.sizePolicy().hasHeightForWidth())
        self.label_26.setSizePolicy(sizePolicy5)

        self.horizontalLayout_2.addWidget(self.label_26)

        self.chirpA2 = QDoubleSpinBox(self.extrasCanal1)
        self.chirpA2.setObjectName(u"chirpA2")
        self.chirpA2.setDecimals(3)
        self.chirpA2.setMaximum(1.000000000000000)
        self.chirpA2.setSingleStep(0.100000000000000)

        self.horizontalLayout_2.addWidget(self.chirpA2)

        self.label_19 = QLabel(self.extrasCanal1)
        self.label_19.setObjectName(u"label_19")
        sizePolicy3.setHeightForWidth(self.label_19.sizePolicy().hasHeightForWidth())
        self.label_19.setSizePolicy(sizePolicy3)

        self.horizontalLayout_2.addWidget(self.label_19)

        self.chirpTinicio = QSpinBox(self.extrasCanal1)
        self.chirpTinicio.setObjectName(u"chirpTinicio")
        self.chirpTinicio.setMaximum(255)
        self.chirpTinicio.setValue(10)

        self.horizontalLayout_2.addWidget(self.chirpTinicio)

        self.label_18 = QLabel(self.extrasCanal1)
        self.label_18.setObjectName(u"label_18")
        sizePolicy3.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy3)

        self.horizontalLayout_2.addWidget(self.label_18)

        self.chirpDeltaI = QDoubleSpinBox(self.extrasCanal1)
        self.chirpDeltaI.setObjectName(u"chirpDeltaI")
        self.chirpDeltaI.setDecimals(1)
        self.chirpDeltaI.setMinimum(0.100000000000000)
        self.chirpDeltaI.setMaximum(25.500000000000000)
        self.chirpDeltaI.setSingleStep(1.000000000000000)
        self.chirpDeltaI.setValue(10.000000000000000)

        self.horizontalLayout_2.addWidget(self.chirpDeltaI)

        self.label_20 = QLabel(self.extrasCanal1)
        self.label_20.setObjectName(u"label_20")
        sizePolicy3.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy3)

        self.horizontalLayout_2.addWidget(self.label_20)

        self.chirpTfim = QSpinBox(self.extrasCanal1)
        self.chirpTfim.setObjectName(u"chirpTfim")
        self.chirpTfim.setMaximum(255)
        self.chirpTfim.setValue(100)

        self.horizontalLayout_2.addWidget(self.chirpTfim)

        self.label_25 = QLabel(self.extrasCanal1)
        self.label_25.setObjectName(u"label_25")
        sizePolicy3.setHeightForWidth(self.label_25.sizePolicy().hasHeightForWidth())
        self.label_25.setSizePolicy(sizePolicy3)

        self.horizontalLayout_2.addWidget(self.label_25)

        self.chirpDeltaF = QDoubleSpinBox(self.extrasCanal1)
        self.chirpDeltaF.setObjectName(u"chirpDeltaF")
        self.chirpDeltaF.setDecimals(1)
        self.chirpDeltaF.setMinimum(0.100000000000000)
        self.chirpDeltaF.setMaximum(25.500000000000000)
        self.chirpDeltaF.setValue(10.000000000000000)

        self.horizontalLayout_2.addWidget(self.chirpDeltaF)


        self.verticalLayout.addWidget(self.extrasCanal1)


        self.retranslateUi(GeneratorPanel)

        self.comboType.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(GeneratorPanel)
    # setupUi

    def retranslateUi(self, GeneratorPanel):
        GeneratorPanel.setWindowTitle(QCoreApplication.translate("GeneratorPanel", u"Form", None))
        self.checkEnable.setText(QCoreApplication.translate("GeneratorPanel", u"Enable", None))
        self.comboType.setItemText(0, QCoreApplication.translate("GeneratorPanel", u"Noise", None))
        self.comboType.setItemText(1, QCoreApplication.translate("GeneratorPanel", u"Harmonic", None))
        self.comboType.setItemText(2, QCoreApplication.translate("GeneratorPanel", u"Chirp", None))
        self.comboType.setItemText(3, QCoreApplication.translate("GeneratorPanel", u"Step", None))
        self.comboType.setItemText(4, QCoreApplication.translate("GeneratorPanel", u"Square", None))

        self.label_2.setText(QCoreApplication.translate("GeneratorPanel", u"Ampl.:", None))
        self.label_3.setText(QCoreApplication.translate("GeneratorPanel", u"Freq.:", None))
        self.label.setText(QCoreApplication.translate("GeneratorPanel", u"DCLevel:", None))
#if QT_CONFIG(tooltip)
        self.spinDCLevel.setToolTip(QCoreApplication.translate("GeneratorPanel", u"Use to avoid overfitting. Regular values are 2048 for outputs 1 and 2 and 128 for outputs 3 and 4.", None))
#endif // QT_CONFIG(tooltip)
        self.label_26.setText(QCoreApplication.translate("GeneratorPanel", u"A2:", None))
        self.label_19.setText(QCoreApplication.translate("GeneratorPanel", u"ST:", None))
        self.label_18.setText(QCoreApplication.translate("GeneratorPanel", u"DI:", None))
        self.label_20.setText(QCoreApplication.translate("GeneratorPanel", u"ET:", None))
        self.label_25.setText(QCoreApplication.translate("GeneratorPanel", u"DF:", None))
    # retranslateUi

