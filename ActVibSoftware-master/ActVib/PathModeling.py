# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'PathModeling.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_PathModelingDialog(object):
    def setupUi(self, PathModelingDialog):
        if not PathModelingDialog.objectName():
            PathModelingDialog.setObjectName(u"PathModelingDialog")
        PathModelingDialog.resize(715, 495)
        self.verticalLayout = QVBoxLayout(PathModelingDialog)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(3, 3, 3, 3)
        self.widget_3 = QWidget(PathModelingDialog)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout = QHBoxLayout(self.widget_3)
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(3, 3, 3, 3)
        self.label_7 = QLabel(self.widget_3)
        self.label_7.setObjectName(u"label_7")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)

        self.horizontalLayout.addWidget(self.label_7)

        self.comboSource = QComboBox(self.widget_3)
        self.comboSource.addItem("")
        self.comboSource.addItem("")
        self.comboSource.setObjectName(u"comboSource")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.comboSource.sizePolicy().hasHeightForWidth())
        self.comboSource.setSizePolicy(sizePolicy1)

        self.horizontalLayout.addWidget(self.comboSource)

        self.line_3 = QFrame(self.widget_3)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.VLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line_3)

        self.label_8 = QLabel(self.widget_3)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout.addWidget(self.label_8)

        self.fileName = QLineEdit(self.widget_3)
        self.fileName.setObjectName(u"fileName")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(2)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.fileName.sizePolicy().hasHeightForWidth())
        self.fileName.setSizePolicy(sizePolicy2)

        self.horizontalLayout.addWidget(self.fileName)

        self.bOpenFile = QPushButton(self.widget_3)
        self.bOpenFile.setObjectName(u"bOpenFile")

        self.horizontalLayout.addWidget(self.bOpenFile)


        self.verticalLayout.addWidget(self.widget_3)

        self.widget = QWidget(PathModelingDialog)
        self.widget.setObjectName(u"widget")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy3)
        self.plotLayout = QVBoxLayout(self.widget)
        self.plotLayout.setSpacing(0)
        self.plotLayout.setObjectName(u"plotLayout")
        self.plotLayout.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout.addWidget(self.widget)

        self.widget_2 = QWidget(PathModelingDialog)
        self.widget_2.setObjectName(u"widget_2")
        sizePolicy4 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy4)
        self.gridLayout = QGridLayout(self.widget_2)
        self.gridLayout.setSpacing(3)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(3, 3, 3, 3)
        self.label = QLabel(self.widget_2)
        self.label.setObjectName(u"label")
        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy5)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.spinMemSize = QSpinBox(self.widget_2)
        self.spinMemSize.setObjectName(u"spinMemSize")
        self.spinMemSize.setMinimum(1)
        self.spinMemSize.setMaximum(3000)
        self.spinMemSize.setValue(1000)

        self.gridLayout.addWidget(self.spinMemSize, 0, 8, 1, 1)

        self.line = QFrame(self.widget_2)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 0, 2, 2, 1)

        self.spinEndTime = QDoubleSpinBox(self.widget_2)
        self.spinEndTime.setObjectName(u"spinEndTime")

        self.gridLayout.addWidget(self.spinEndTime, 1, 1, 1, 1)

        self.textPenalization = QLineEdit(self.widget_2)
        self.textPenalization.setObjectName(u"textPenalization")

        self.gridLayout.addWidget(self.textPenalization, 1, 5, 1, 1)

        self.spinStartTime = QDoubleSpinBox(self.widget_2)
        self.spinStartTime.setObjectName(u"spinStartTime")

        self.gridLayout.addWidget(self.spinStartTime, 0, 1, 1, 1)

        self.label_3 = QLabel(self.widget_2)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 4, 1, 1)

        self.spinAveraging = QSpinBox(self.widget_2)
        self.spinAveraging.setObjectName(u"spinAveraging")
        self.spinAveraging.setMinimum(1)
        self.spinAveraging.setMaximum(1000)
        self.spinAveraging.setValue(250)

        self.gridLayout.addWidget(self.spinAveraging, 1, 8, 1, 1)

        self.label_4 = QLabel(self.widget_2)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 1, 4, 1, 1)

        self.label_2 = QLabel(self.widget_2)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.textStepSize = QLineEdit(self.widget_2)
        self.textStepSize.setObjectName(u"textStepSize")

        self.gridLayout.addWidget(self.textStepSize, 0, 5, 1, 1)

        self.label_6 = QLabel(self.widget_2)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 1, 7, 1, 1)

        self.label_5 = QLabel(self.widget_2)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 0, 7, 1, 1)

        self.line_2 = QFrame(self.widget_2)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_2, 0, 6, 2, 1)


        self.verticalLayout.addWidget(self.widget_2)

        self.bRunModeling = QPushButton(PathModelingDialog)
        self.bRunModeling.setObjectName(u"bRunModeling")

        self.verticalLayout.addWidget(self.bRunModeling)

        self.bSaveToFile = QPushButton(PathModelingDialog)
        self.bSaveToFile.setObjectName(u"bSaveToFile")

        self.verticalLayout.addWidget(self.bSaveToFile)

        self.widget_4 = QWidget(PathModelingDialog)
        self.widget_4.setObjectName(u"widget_4")
        self.horizontalLayout_2 = QHBoxLayout(self.widget_4)
        self.horizontalLayout_2.setSpacing(3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.bUploadAndRec = QPushButton(self.widget_4)
        self.bUploadAndRec.setObjectName(u"bUploadAndRec")
        sizePolicy1.setHeightForWidth(self.bUploadAndRec.sizePolicy().hasHeightForWidth())
        self.bUploadAndRec.setSizePolicy(sizePolicy1)

        self.horizontalLayout_2.addWidget(self.bUploadAndRec)

        self.progressBar = QProgressBar(self.widget_4)
        self.progressBar.setObjectName(u"progressBar")
        sizePolicy6 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy6.setHorizontalStretch(1)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy6)
        self.progressBar.setValue(24)

        self.horizontalLayout_2.addWidget(self.progressBar)


        self.verticalLayout.addWidget(self.widget_4)

        self.bCheck = QPushButton(PathModelingDialog)
        self.bCheck.setObjectName(u"bCheck")

        self.verticalLayout.addWidget(self.bCheck)

        self.statusLabel = QLabel(PathModelingDialog)
        self.statusLabel.setObjectName(u"statusLabel")

        self.verticalLayout.addWidget(self.statusLabel)


        self.retranslateUi(PathModelingDialog)

        QMetaObject.connectSlotsByName(PathModelingDialog)
    # setupUi

    def retranslateUi(self, PathModelingDialog):
        PathModelingDialog.setWindowTitle(QCoreApplication.translate("PathModelingDialog", u"Path Modeling", None))
        self.label_7.setText(QCoreApplication.translate("PathModelingDialog", u"Source:", None))
        self.comboSource.setItemText(0, QCoreApplication.translate("PathModelingDialog", u"Present Recording", None))
        self.comboSource.setItemText(1, QCoreApplication.translate("PathModelingDialog", u"File", None))

        self.label_8.setText(QCoreApplication.translate("PathModelingDialog", u"File:", None))
        self.bOpenFile.setText(QCoreApplication.translate("PathModelingDialog", u"...", None))
        self.label.setText(QCoreApplication.translate("PathModelingDialog", u"Start time:", None))
        self.textPenalization.setText(QCoreApplication.translate("PathModelingDialog", u"1e-2", None))
        self.label_3.setText(QCoreApplication.translate("PathModelingDialog", u"NLMS step-size:", None))
        self.label_4.setText(QCoreApplication.translate("PathModelingDialog", u"NLMS penalization:", None))
        self.label_2.setText(QCoreApplication.translate("PathModelingDialog", u"End time:", None))
        self.textStepSize.setText(QCoreApplication.translate("PathModelingDialog", u"0.05", None))
        self.label_6.setText(QCoreApplication.translate("PathModelingDialog", u"Weight averaging:", None))
        self.label_5.setText(QCoreApplication.translate("PathModelingDialog", u"Memory size:", None))
        self.bRunModeling.setText(QCoreApplication.translate("PathModelingDialog", u"Run...", None))
        self.bSaveToFile.setText(QCoreApplication.translate("PathModelingDialog", u"Save paths to file...", None))
        self.bUploadAndRec.setText(QCoreApplication.translate("PathModelingDialog", u"Upload and Record in Flash...", None))
        self.bCheck.setText(QCoreApplication.translate("PathModelingDialog", u"Check Recorded Paths", None))
        self.statusLabel.setText("")
    # retranslateUi

