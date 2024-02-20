# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'DataViewer.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_DataViewerDialog(object):
    def setupUi(self, DataViewerDialog):
        if not DataViewerDialog.objectName():
            DataViewerDialog.setObjectName(u"DataViewerDialog")
        DataViewerDialog.resize(718, 613)
        DataViewerDialog.setAcceptDrops(True)
        self.gridLayout = QGridLayout(DataViewerDialog)
        self.gridLayout.setSpacing(3)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(6, 6, 6, 6)
        self.bOpenDataInMemory = QPushButton(DataViewerDialog)
        self.bOpenDataInMemory.setObjectName(u"bOpenDataInMemory")

        self.gridLayout.addWidget(self.bOpenDataInMemory, 0, 2, 1, 1)

        self.statusLabel = QLabel(DataViewerDialog)
        self.statusLabel.setObjectName(u"statusLabel")

        self.gridLayout.addWidget(self.statusLabel, 3, 0, 1, 3)

        self.bOpen = QPushButton(DataViewerDialog)
        self.bOpen.setObjectName(u"bOpen")

        self.gridLayout.addWidget(self.bOpen, 0, 1, 1, 1)

        self.fileName = QLineEdit(DataViewerDialog)
        self.fileName.setObjectName(u"fileName")

        self.gridLayout.addWidget(self.fileName, 0, 0, 1, 1)

        self.widget = QWidget(DataViewerDialog)
        self.widget.setObjectName(u"widget")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setAcceptDrops(True)
        self.plotLayout = QVBoxLayout(self.widget)
        self.plotLayout.setObjectName(u"plotLayout")

        self.gridLayout.addWidget(self.widget, 1, 0, 1, 3)

        self.logText = QTextEdit(DataViewerDialog)
        self.logText.setObjectName(u"logText")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.logText.sizePolicy().hasHeightForWidth())
        self.logText.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.logText, 2, 0, 1, 3)


        self.retranslateUi(DataViewerDialog)

        QMetaObject.connectSlotsByName(DataViewerDialog)
    # setupUi

    def retranslateUi(self, DataViewerDialog):
        DataViewerDialog.setWindowTitle(QCoreApplication.translate("DataViewerDialog", u"Data Viewer", None))
        self.bOpenDataInMemory.setText(QCoreApplication.translate("DataViewerDialog", u"Data In Memory", None))
        self.statusLabel.setText("")
        self.bOpen.setText(QCoreApplication.translate("DataViewerDialog", u"...", None))
        self.logText.setHtml(QCoreApplication.translate("DataViewerDialog", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
    # retranslateUi

