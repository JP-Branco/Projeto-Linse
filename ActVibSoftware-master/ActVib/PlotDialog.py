# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'PlotDialog.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(200, 160)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(Dialog)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(3, 3, 3, 3)
        self.groupBox_2 = QGroupBox(Dialog)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy1)
        self.horizontalLayout = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(9, 3, 9, 3)
        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.janelaX = QSpinBox(self.groupBox_2)
        self.janelaX.setObjectName(u"janelaX")
        self.janelaX.setMinimum(1)
        self.janelaX.setMaximum(600)
        self.janelaX.setSingleStep(10)
        self.janelaX.setValue(30)

        self.horizontalLayout.addWidget(self.janelaX)


        self.verticalLayout.addWidget(self.groupBox_2)

        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy1.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy1)
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setSpacing(3)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(9, 3, -1, 3)
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_2.addWidget(self.label_2, 2, 0, 1, 1)

        self.minY = QDoubleSpinBox(self.groupBox)
        self.minY.setObjectName(u"minY")
        self.minY.setMinimum(-2000.000000000000000)
        self.minY.setMaximum(2000.000000000000000)
        self.minY.setValue(-100.000000000000000)

        self.gridLayout_2.addWidget(self.minY, 3, 1, 1, 1)

        self.maxY = QDoubleSpinBox(self.groupBox)
        self.maxY.setObjectName(u"maxY")
        self.maxY.setMinimum(-2000.000000000000000)
        self.maxY.setMaximum(2000.000000000000000)
        self.maxY.setValue(100.000000000000000)

        self.gridLayout_2.addWidget(self.maxY, 2, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_2.addWidget(self.label_3, 3, 0, 1, 1)

        self.checkAutoY = QCheckBox(self.groupBox)
        self.checkAutoY.setObjectName(u"checkAutoY")

        self.gridLayout_2.addWidget(self.checkAutoY, 1, 0, 1, 2)


        self.verticalLayout.addWidget(self.groupBox)

        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(True)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Plot Config", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Dialog", u" Eixo X ", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"Janela de Tempo:", None))
        self.janelaX.setSuffix(QCoreApplication.translate("Dialog", u" s", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u" Eixo Y ", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"M\u00e1ximo em Y:", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"M\u00ednimo em Y:", None))
        self.checkAutoY.setText(QCoreApplication.translate("Dialog", u"Auto", None))
    # retranslateUi

