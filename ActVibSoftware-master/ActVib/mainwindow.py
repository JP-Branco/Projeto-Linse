from pathlib import Path
from os import getenv
import time

import pandas as pd
import numpy as np

from PySide2 import QtWidgets, QtGui, QtCore
from PySide2.QtWidgets import QMessageBox, QFileDialog, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox

from .VibViewWindow import Ui_MainWindow

from .figures import (CtrlFigQtGraph, FigOutputQtGraph, MyFigQtGraph)

from .dialogs import (WorkdirManager, MyUploadDialog, MyPathModelingDialog, MyDataViewer, MyAdditionalDialog)

from .panels import (IMUPanel,GeneratorPanel,ControlPanel,ADCPanel)

from .automator import Automator

from .updater import Updater

class mainwindow(QtWidgets.QMainWindow):

    resumeAutomator = QtCore.Signal()

    def __init__(self, app, driver, dataman, mainfig : MyFigQtGraph):
        super(mainwindow, self).__init__()
        self.app = app
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.Port = "COM3"
        self.TSampling = 4000   # 4000 us is the default
        self.FSampling = 250  # 250 Hz is the default
        self.MaxTime = 10   # 10 min is the default
        self.BaudRate = 500000
        self.workdir = Path.home()
        self.dataman = dataman
        self.driver = driver
        self.saveconfigtypes = [QtWidgets.QCheckBox, QtWidgets.QComboBox, QtWidgets.QRadioButton,
                                QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox, QtWidgets.QLineEdit]
        self.imupanel = [IMUPanel(idd) for idd in range(3)]
        self.genpanel = [GeneratorPanel(idd) for idd in range(4)]
        self.adcpanel = ADCPanel()

        self.ctrlpanel = ControlPanel()
        self.ui.comboMode.currentIndexChanged.connect(self.mainConfigControl)

        self.mfig = mainfig #MyFigQtGraph(self.dataman, self)
        self.ctrlfig = CtrlFigQtGraph(self.dataman, self)        
        self.ofig = FigOutputQtGraph(self.dataman, self)
        self.addPlots()

        self.readConfig()        
        self.ui.bInit.clicked.connect(lambda: self.bInit(3600.0))
        self.ui.bLimpar.clicked.connect(lambda: self.bReset(ignoreunsaved=False))

        self.mapper = QtCore.QSignalMapper(self)
        self.mapper.mapped['QString'].connect(self.setPort)

        self.mapper2 = QtCore.QSignalMapper(self)
        for sra in self.ui.menuSampling_Rate.actions():
            self.mapper2.setMapping(sra, sra.text())
            sra.triggered.connect(self.mapper2.map)
        self.mapper2.mapped['QString'].connect(self.setSampling)

        self.mapper3 = QtCore.QSignalMapper(self)
        for sra in self.ui.menuMax_Time.actions():
            self.mapper3.setMapping(sra, sra.text())
            sra.triggered.connect(self.mapper3.map)
        self.mapper3.mapped['QString'].connect(self.setMaxTime)

        self.mapper4 = QtCore.QSignalMapper(self)
        for sra in self.ui.menuSerial_Baud_Rate.actions():
            self.mapper4.setMapping(sra, sra.text())
            sra.triggered.connect(self.mapper4.map)
        self.mapper4.mapped['QString'].connect(self.setBaudRate)
        
        self.ui.menuSelecionar_Porta.aboutToShow.connect(self.populatePorts)

        self.ui.actionSair.triggered.connect(self.closeEvent)
        self.ui.actionSalvar_dados.triggered.connect(self.saveFile)
        self.ui.actionUpload.triggered.connect(self.openUploadDialog)
        self.ui.actionWorkdirManager.triggered.connect(self.openWorkdirManager)
        self.ui.actionPathModeling.triggered.connect(self.openPathModelingDialog)
        self.ui.actionDataViewer.triggered.connect(self.openDataViewer)
        self.ui.actionAdditional.triggered.connect(self.openAdditionalConfig)

        self.ui.actionFirmware_Update.triggered.connect(self.SFUpdate)
        
        # IMU Connections:        
        self.ui.imutab1.layout().addWidget(self.imupanel[0])
        self.ui.imutab2.layout().addWidget(self.imupanel[1])
        self.ui.imutab3.layout().addWidget(self.imupanel[2])
        for imupanel in self.imupanel:
            # imupanel = self.imupanel[0]
            for cc in imupanel.findChildren(QComboBox):
                cc.activated.connect(self.changeMPUConfig)

        # ADC Connections
        self.ui.adctab.layout().addWidget(self.adcpanel)

        # Signal Generator Connections:
        self.ui.canal1.layout().addWidget(self.genpanel[0])
        self.ui.canal2.layout().addWidget(self.genpanel[1])
        self.ui.canal3.layout().addWidget(self.genpanel[2])
        self.ui.canal4.layout().addWidget(self.genpanel[3])
        for cc in self.ui.tabWidget.findChildren(QComboBox, QtCore.QRegExp("comboType.*")):
            cc.activated.connect(self.changeGeneratorConfig)
        for cc in self.ui.tabWidget.findChildren(QDoubleSpinBox, QtCore.QRegExp("spinAmpl.*|spinFreq.*")):
            cc.editingFinished.connect(self.changeGeneratorConfig)
        for cc in self.ui.tabWidget.findChildren(QCheckBox, QtCore.QRegExp("checkEnable.*")):
            cc.toggled.connect(self.plotOutConfig)
            cc.toggled.connect(self.changeGeneratorConfig)
        
        # ControlPanel:
        self.ui.controlFrame.layout().addWidget(self.ctrlpanel)  
        self.ctrlpanel.connectControlChanged(self.configControl)
        self.ctrlpanel.ui.checkAlgOn.toggled.connect(self.configAlgOn)
        self.ctrlpanel.setActive(self.ui.comboMode.currentIndex() == 1)

        self.ui.notes.textChanged.connect(self.txtInputChanged)
        self.maxNoteLength = 1000
        
        self.dataman.updateFigures.connect(self.updateFigs)
        self.dataman.reset.connect(self.dataReset)
        self.dataman.statusMessage.connect(self.statusMessage)
        self.dataman.stopped.connect(self.readingsStopped)
        self.dataman.started.connect(self.readingsStarted)
        self.dataman.logMessage.connect(self.processLogMsg)
        self.loglist = []

        self.changeGeneratorConfig()
        self.changeMPUConfig()
        self.changeADCConfig()
        self.configControl()

        saveplussc = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self.saveFile)
        saveplus2sc = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+S"), self, self.savePlus2)
        initstop = QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, self.kSpace)
        stepfocus = QtWidgets.QShortcut(QtGui.QKeySequence("Alt+P"), self, self.pFocus)
        self.ui.actionSavePlus.triggered.connect(self.savePlus)
        self.ui.actionDefineWorkdir.triggered.connect(self.defWorkdir)
        self.flagsaveplus = False
        
        self.ui.actionAutomator.triggered.connect(self.showAutomator)

        self.lastSavedExtension = None

        self.disabledwhenrunning = []

        self.automator = Automator(self.app)
        self.automatorWaiting = False
        self.automatorWaitingStart = False
        self.automator.actionMessage.connect(self.processActions)
        self.resumeAutomator.connect(self.automator.resume)

        self.updater = Updater()
        self.updater.actionMessage.connect(self.updater.printMessage) 


    def SFUpdate(self):
        self.updater.showUpdaterDialog(self.Port)
        # fupd.runUpdate(self.porta)
    

    def showAutomator(self):
        self.automatorWaiting = False        
        self.automator.showAutomatorDialog() 

    def processActions(self,msg,opts):
        dialogmsg = f"{self.automator.elapsedTime():.0f}: {msg}"
        if msg == "Reset":
            self.bReset(ignoreunsaved=True)
            self.resumeAutomator.emit()
        elif msg == "Delay":
            dialogmsg += f" - {opts}"            
        elif msg == "ConfigOutput":
            dialogmsg += f" - {opts}"
            idxoutput = int(opts[0]) - 1
            if (idxoutput >= 0) and (idxoutput < 4):
                genenabled = True if (opts[1] in ["Enabled","True"]) else False
                # self.genpanel[idxoutput].setEnabled(genenabled)
                self.genpanel[idxoutput].ui.checkEnable.setChecked(genenabled)
                if genenabled:
                    self.genpanel[idxoutput].setGeneratorConfig(opts[2], float(opts[3]), float(opts[4]))
                self.changeGeneratorConfig(idxoutput)
            else: 
                dialogmsg += f" Error: {str(ex)}"
                self.automator.stop()
            self.resumeAutomator.emit()
        elif msg == "ConfigIMU":
            dialogmsg += f" - {opts}"
            idximu = int(opts[0]) - 1
            if (idximu >= 0) and (idximu < 3):
                self.imupanel[idximu].ui.comboType.setCurrentText(opts[1])
                if opts[1] != "Disabled":
                    self.imupanel[idximu].ui.comboAddress.setCurrentText(opts[2])
                    self.imupanel[idximu].ui.comboBus.setCurrentText(opts[3])
                    self.imupanel[idximu].ui.comboAccRange.setCurrentIndex(int(opts[4]))
                    self.imupanel[idximu].ui.comboGyroRange.setCurrentIndex(int(opts[5]))
                    self.imupanel[idximu].ui.comboFilter.setCurrentIndex(int(opts[6]))
                    self.imupanel[idximu].ui.comboFilter2.setCurrentIndex(int(opts[7]))
            else: 
                dialogmsg += f" Error: {str(ex)}"
                self.automator.stop()
            self.resumeAutomator.emit()
        elif msg == "ConfigADC":
            dialogmsg += f" - {opts}"
            self.adcpanel.ui.comboADCModel.setCurrentText(opts[0])
            auxdict = { "Off":self.adcpanel.ui.radioOff, 
                "1":self.adcpanel.ui.radioADC1, 
                "2":self.adcpanel.ui.radioADC2, 
                "3":self.adcpanel.ui.radioADC3, 
                "4":self.adcpanel.ui.radioADC4}
            auxdict[opts[1]].setChecked(True)            
            self.adcpanel.ui.comboADCRange.setCurrentText(f"±{opts[2]} V")
            if opts[0] == "ADS1115":
                self.adcpanel.ui.comboRate1115.setCurrentText(f"{opts[3]} SPS")
            elif opts[0] == "ADS1015":
                self.adcpanel.ui.comboRate1015.setCurrentText(f"{opts[3]} SPS")
            else: 
                dialogmsg += f" Error: Invalid ADC Model"
                self.automator.stop()
            self.resumeAutomator.emit()
        elif msg == "ControlChannels":
            dialogmsg += f" - {opts}"
            self.ctrlpanel.ui.comboPerturbChannel.setCurrentText(opts[0])
            self.ctrlpanel.ui.comboControlChannel.setCurrentText(opts[1])
            self.ctrlpanel.ui.comboIMURef.setCurrentText(opts[2])
            self.ctrlpanel.ui.comboRef.setCurrentText(opts[3])
            self.ctrlpanel.ui.comboIMUError.setCurrentText(opts[4])
            self.ctrlpanel.ui.comboErro.setCurrentText(opts[5])
            self.resumeAutomator.emit()
        elif msg == "SetControlMode": 
            dialogmsg += f" - {opts}"
            # self.ctrlpanel.ui.checkControle.setChecked(True)
            self.ui.comboMode.setCurrentIndex(1)
            self.ctrlpanel.ui.comboCtrlTask.setCurrentIndex(0)
            self.ctrlpanel.ui.comboAlgoritmo.setCurrentText(opts[0])
            self.ctrlpanel.ui.spinTAlgOn.setValue(int(opts[1]))
            self.ctrlpanel.ui.spinMemCtrl.setValue(int(opts[2]))
            self.ctrlpanel.ui.passoCtrl.setText(opts[3])
            self.ctrlpanel.ui.normCtrl.setText(opts[4]) 
            self.resumeAutomator.emit()
        elif msg == "SetPathModeling":            
            # self.ctrlpanel.ui.checkControle.setChecked(True)
            self.ui.comboMode.setCurrentIndex(1)
            self.ctrlpanel.ui.comboCtrlTask.setCurrentIndex(1)            
            self.resumeAutomator.emit() 
        elif msg == "SetReadingMode":            
            # self.ctrlpanel.ui.checkControle.setChecked(False)
            self.ui.comboMode.setCurrentIndex(0) 
            self.resumeAutomator.emit() 
        elif msg == "Wait":            
            if not self.dataman.flagrodando:
                self.resumeAutomator.emit()
                self.automatorWaiting = False
            else:
                self.automatorWaiting = True
        elif msg == "Start":
            dialogmsg += f" with stop time {opts[0]}"
            self.automatorWaiting = False
            self.bInit(stoptime=int(opts[0])) 
            self.resumeAutomator.emit()
        elif msg == "Stop":
            # dialogmsg += f" with stop time {opts[0]}"
            self.automatorWaiting = True
            self.bInit()
        elif msg == "StartWait":
            dialogmsg += f" with stop time {opts[0]} - waiting end of experiment."
            self.automatorWaiting = False
            self.bInit(stoptime=int(opts[0])) 
            self.automatorWaiting = True
        elif msg == "StartNotify":
            dialogmsg += f" - waiting notification of start of experiment."
            self.automatorWaitingStart = False
            self.bInit(stoptime=int(opts[0])) 
            self.automatorWaitingStart = True
        elif msg == "AlgOn":
            self.ctrlpanel.ui.checkAlgOn.setChecked(True if (opts[0] == "True") else False)  
            self.resumeAutomator.emit() 
        elif msg == "Stopping":
            if self.dataman.flagrodando:
                self.bInit()       
        elif msg == "Print":
            print(opts[0])
            self.resumeAutomator.emit()
        elif msg == "SaveFile":
            try:
                self.saveFile(fname=Path(self.workdir,opts[0]))
                self.resumeAutomator.emit()
            except BaseException as ex:
                dialogmsg += f" Error: {str(ex)}"
                self.automator.stop()
        elif msg == "SetWorkDir":
            try:
                aux = Path.home() / opts[0]
                aux.mkdir(parents=True,exist_ok=True)
                print(aux)
                self.workdir = aux
                self.resumeAutomator.emit()
            except BaseException as ex:
                dialogmsg += f" Error: {str(ex)}"
                self.automator.stop()
        self.automator.printMessage(dialogmsg + "\n") 


    def processLogMsg(self,tstamp,msg):
        if tstamp == 0:
            self.loglist = []
        if msg == "Started":
            self.loglist.append((tstamp,msg))
            for k,imu in enumerate(self.imupanel):
                self.loglist.append((tstamp,f"IMU{k+1}|{imu.getLogString()}"))
            self.loglist.append((tstamp,f"ADC|{self.adcpanel.getLogString()}"))
            for k,gen in enumerate(self.genpanel):                
                self.loglist.append((tstamp,f"Gen{k+1}|{gen.getLogString()}"))   
            self.loglist.append((tstamp,f"Ctrl|{self.ctrlpanel.getLogString()}"))
            # print( self.loglist )        
        elif msg == "Stopped":
            self.loglist.append((tstamp,msg))
            # print(self.loglist[-1])
        elif msg.startswith("Gen"):
            idx = int(msg[3:])
            self.loglist.append((tstamp,f"Gen{idx+1}|{self.genpanel[idx].getLogString()}"))
            # print(self.loglist[-1])
        elif msg == "Ctrl":
            self.loglist.append((tstamp,f"Ctrl|{self.ctrlpanel.getLogString()}"))
            # print(self.loglist[-1])
        elif msg == "Alg":
            self.loglist.append((tstamp,self.ctrlpanel.getAlgOnLogString()))
            # print(self.loglist[-1])

    
    def txtInputChanged(self):
        txt = self.ui.notes.toPlainText()
        if len(txt) > self.maxNoteLength:
            self.ui.notes.setPlainText(txt[:self.maxNoteLength])
            cursor = self.ui.notes.textCursor()
            cursor.setPosition(self.maxNoteLength)
            self.ui.notes.setTextCursor(cursor)

    def pFocus(self):
        self.ui.passoCtrl.selectAll()
        self.ui.passoCtrl.setFocus()


    def kSpace(self):
        if not self.dataman.flagrodando:
            self.bInit()
        elif self.driver.controlMode and (not self.ctrlpanel.isAlgOn()):
            self.ctrlpanel.ui.checkAlgOn.setChecked(True)
        else:
            self.bInit()


    def savePlus2(self):
        self.savePlus()
        self.bReset()


    def savePlus(self):
        try:
            if self.flagsaveplus:
                raise Exception("Data already saved...")
            fn = str(int(time.time() * 1000)) + '.feather'
            fname = Path(self.workdir) / fn
            ct = 0
            while fname.exists():
                fn = str(int(time.time() * 1000)) + f'{ct}.feather'
                fname = Path(self.workdir) / fn
                ct = ct + 1
                if ct > 10:
                    raise Exception("10 file creation attempts and file still does not exist.")
            if self.dataman.globalctreadings > 0:
                if self.driver.controlMode:
                    mdatafile = Path(self.workdir) / "metacontrol.feather"
                    if mdatafile.exists():
                        mddf = pd.read_feather(mdatafile)
                    else:
                        mddf = pd.DataFrame(
                            columns=['filename', 'alg', 'mu', 'psi', 'memctrl',
                                     'tinicio', 'tipoperturb', 'amplperturb',
                                     'freqperturb', 'sensorref', 'sensorerro']
                        )
                    fdata = {
                        'filename': str(fn),
                        'alg': self.ui.comboAlgoritmo.currentText(),
                        'mu': float(self.ui.passoCtrl.text()),
                        'psi': float(self.ui.normCtrl.text()),
                        'memctrl': self.ui.spinMemCtrl.value(),
                        'tinicio': self.ui.spinTAlgOn.value(),
                        'tipoperturb': (self.ui.comboCanal1.currentText()
                                        if (self.ui.comboCanaisControle.currentIndex() == 0)
                                        else self.ui.comboCanal2.currentText()),
                        'amplperturb': (self.ui.spinAmpl1.value()    # TODO!!!!
                                        if (self.ui.comboCanaisControle.currentIndex() == 0)
                                        else self.ui.spinAmpl2.value()),
                        'freqperturb': (self.ui.spinFreq1.value()
                                        if (self.ui.comboCanaisControle.currentIndex() == 0)
                                        else self.ui.spinFreq2.value()),
                        'sensorref': self.ui.comboRef.currentText(),
                        'sensorerro': self.ui.comboErro.currentText()
                    }
                    mddf = mddf.append(fdata, ignore_index=True)
                    mddf.reset_index()
                    mddf.to_feather(mdatafile)
                    try:
                        self.dataman.salvaArquivo(fname, setsaved=True)
                    except Exception as err:
                        QMessageBox.question(self.app, "Erro!", str(err), QMessageBox.Ok)
                    self.flagsaveplus = True
                    self.ui.statusbar.showMessage("Data successfully saved.")
                else:
                    pass
            else:
                raise Exception("No data to save.")
        except Exception as err:
            self.ui.statusbar.showMessage(str(err))


    def defWorkdir(self):
        fdg = QFileDialog(self, "Choose the working directory (workdir)", Path.home())
        fdg.setFileMode(QFileDialog.DirectoryOnly)
        if fdg.exec():
            dirnames = fdg.selectedFiles()
            if dirnames is not None:
                if dirnames[0] != '':
                    self.workdir = dirnames[0]


    def readConfig(self):        
        settings = QtCore.QSettings("VibSoftware", "VibView")
        for w in self.app.allWidgets():
            if ((type(w) in self.saveconfigtypes) and (settings.value(w.objectName()) is not None)):
                if (type(w) in [QtWidgets.QCheckBox,QtWidgets.QRadioButton]):
                    w.setChecked(settings.value(w.objectName()) == 'True')                    
                elif (type(w) == QtWidgets.QComboBox):
                    w.setCurrentIndex(int(settings.value(w.objectName())))
                elif (type(w) == QtWidgets.QDoubleSpinBox):
                    w.setValue(float(settings.value(w.objectName())))
                elif (type(w) == QtWidgets.QSpinBox):
                    w.setValue(int(settings.value(w.objectName())))
                elif (type(w) == QtWidgets.QLineEdit):
                    w.setText(settings.value(w.objectName()))
        if (settings.value("Porta") is not None):
            self.Port = settings.value("Porta")
            self.setPort(self.Port)
        if (settings.value("FSampling") is not None):
            self.setSampling(settings.value("FSampling"))  
        if (settings.value("MaxTime") is not None):
            self.setMaxTime(settings.value("MaxTime"))
        if (settings.value("BaudRate") is not None):
            self.setBaudRate(settings.value("BaudRate"))  
        if (settings.value("WorkDir") is not None):
            self.workdir = settings.value("WorkDir")
        if (settings.value("LastDataFolder") is not None):
            self.dataman.lastdatafolder = Path(settings.value("LastDataFolder"))
        if (settings.value("PreDistEnMap") is not None):
            tpredistmap = settings.value("PreDistEnMap")
            for k in range(4):
                self.driver.predistenablemap[k] = True if tpredistmap[k] == 'true' else False
        if settings.value("PreDistCoefs"):
            self.driver.predistcoefs = settings.value("PreDistCoefs")
            if len(self.driver.predistcoefs) != 4:
                self.driver.predistcoefs = [np.array([1.0,0.0]),np.array([1.0,0.0]),np.array([1.0,0.0]),np.array([1.0,0.0])]
        if settings.value("FusionWeights"):            
            self.driver.fusionweights = [float(x) for x in settings.value("FusionWeights")]
        for imp in self.imupanel:
            imp.restoreState(settings)
            imp.typechanged()
        for gp in self.genpanel:
            gp.restoreState(settings)
        if self.mfig:
            for k in range(2):
                auxval = settings.value(f"SensorChoice{k}")
                self.mfig.setSensorChoice(k,int(auxval) if auxval is not None else k*3+1)
                auxval2 = settings.value(f"XYZmap{k}")                
                self.mfig.setXYZCheckMap(k,auxval2 if auxval2 is not None else [True,True,True])
            self.mfig.loadSensorChoices()
        

    def writeConfig(self):
        settings = QtCore.QSettings("VibSoftware", "VibView")
        for w in self.app.allWidgets():            
            if (type(w) in [QtWidgets.QCheckBox,QtWidgets.QRadioButton]):
                settings.setValue(w.objectName(), str(w.isChecked()))
            elif (type(w) == QtWidgets.QComboBox):
                settings.setValue(w.objectName(), w.currentIndex())
            elif (type(w) == QtWidgets.QDoubleSpinBox):
                settings.setValue(w.objectName(), w.value())
            elif (type(w) == QtWidgets.QSpinBox):
                settings.setValue(w.objectName(), w.value())
            elif (type(w) == QtWidgets.QLineEdit):
                settings.setValue(w.objectName(), w.text())
        settings.setValue("Porta", self.Port)
        settings.setValue("FSampling", str(self.FSampling) + " Hz")
        settings.setValue("MaxTime", str(self.MaxTime) + " min")
        settings.setValue("BaudRate", str(self.BaudRate))
        settings.setValue("MFig", self.mfig.getConfigString())
        settings.setValue("OFig", self.ofig.getConfigString())
        settings.setValue("CtrlFig", self.ctrlfig.getConfigString())
        settings.setValue('WorkDir', self.workdir)
        settings.setValue("LastDataFolder", str(self.dataman.lastdatafolder))
        settings.setValue("PreDistEnMap", self.driver.predistenablemap)
        settings.setValue("PreDistCoefs", self.driver.predistcoefs)
        settings.setValue("FusionWeights", self.driver.fusionweights)
        for imp in self.imupanel:
            imp.saveState(settings)
        for gp in self.genpanel:
            gp.saveState(settings)
        for k in range(2):
            settings.setValue(f"SensorChoice{k}",self.mfig.plotchoice[k])
            settings.setValue(f"XYZmap{k}",self.mfig.getXYZCheckMap(k))

    def statusMessage(self, msg):
        if msg is None:
            self.ui.statusbar.clearMessage()
        else:
            self.ui.statusbar.showMessage(msg) 


    def updateFigs(self):
        self.ui.elapsedTime.setText(f'{int(self.dataman.readtime)} s / {self.dataman.maxtime} s')
        self.ui.controlTime.setText(f'{self.dataman.realtime:.3f} s')
        self.ui.sampleTime.setText(f'{self.driver.calctime[0]:.0f}/{self.driver.calctime[1]:.0f}/{self.driver.calctime[2]:.0f} us')
        aux = self.driver.errorflag
        self.ui.errorLabel.setText(f'{aux:x}')
        if aux > 0:
            self.ui.errorLabel.setStyleSheet("background-color:rgb(255,150,150)")
        else:
            self.ui.errorLabel.setStyleSheet("")
        if self.driver.controlMode:
            self.ctrlfig.updateFig()
        else:
            self.mfig.updateFig()
        self.ofig.updateFig()

    
    def dataReset(self):
        self.ui.elapsedTime.setText('0 s')
        self.ui.controlTime.setText('0 s')
        self.ui.sampleTime.setText('0 us')
        self.mfig.updateFig()
        self.ofig.updateFig()
        self.ctrlfig.updateFig()


    def configAlgOn(self):
        algon = self.ctrlpanel.ui.checkAlgOn.isChecked()
        algontime = float(self.ctrlpanel.ui.spinTAlgOn.value())
        if algon:
            self.driver.ctrlmu = float(self.ctrlpanel.ui.passoCtrl.text())
            self.driver.ctrlfi = float(self.ctrlpanel.ui.normCtrl.text())
        self.driver.setAlgOn(algon, algontime)
        self.ctrlpanel.ui.normCtrl.setEnabled(not algon)
        self.ctrlpanel.ui.passoCtrl.setEnabled(not algon)
        self.ctrlpanel.ui.spinTAlgOn.setEnabled(not algon)

    def mainConfigControl(self):
        self.ctrlpanel.setActive(self.ui.comboMode.currentIndex() == 1)
        self.configControl()

    def configControl(self):   
        self.driver.setControlMode(self.ctrlpanel.isControlOn(),self.ctrlpanel.getControlTask())
        if self.driver.controlMode:
            self.driver.controlChannel = self.ctrlpanel.getControlChannel()
            self.driver.perturbChannel = self.ctrlpanel.getPerturbChannel()
            for k,gg in enumerate(self.genpanel):
                if k == self.driver.controlChannel:
                    gg.setEnabled(not self.driver.taskIsControl)
                    self.ui.tabWidget.setTabText(k, f"Output {k+1} (Control)")
                elif k == self.driver.perturbChannel:
                    gg.setEnabled(self.driver.taskIsControl)
                    self.ui.tabWidget.setTabText(k, f"Output {k+1} (Perturb.)")
                else:
                    gg.setEnabled(False)
                    self.ui.tabWidget.setTabText(k, f"Output {k+1}")
            self.driver.setControlConfig( **self.ctrlpanel.getControlConfiguration() )
            self.driver.setDebugMode(False)
            if (self.mfig is not None):
                self.ctrlfig.show()
                self.mfig.hide()
        else:
            for k,gg in enumerate(self.genpanel):
                self.ui.tabWidget.setTabText(k,f"Output {k+1}")
                gg.setEnabled(True)
            self.changeGeneratorConfig()
            if (self.ctrlfig is not None):
                self.mfig.show()
                self.ctrlfig.hide()


    def plotOutConfig(self):
        if self.ctrlpanel.isControlOn() and self.ctrlpanel.isTaskControl():
            self.ofig.dacenable = [True,True,False,False]
        else: 
            self.ofig.dacenable = [aa.isGeneratorOn() for aa in self.genpanel]
        # print(self.ofig.dacenable)
        self.ofig.resetFigure()


    def plotConfig(self):
        self.mfig.loadSensorChoices()
        # self.mfig.accEnable = self.plotcfgpanel.getAccEnableList()
        # self.mfig.gyroEnable = self.plotcfgpanel.getGyroEnableList()        
        self.mfig.resetFigure()
        self.ctrlfig.plotSetup([self.ctrlpanel.ui.comboRef.currentIndex(), self.ctrlpanel.ui.comboErro.currentIndex()])
        self.ctrlfig.resetFigure()


    def changeGeneratorConfig(self,idx=None):
        if idx:
            generatorconfig = self.genpanel[idx].getGeneratorConfig()
            self.driver.setGeneratorConfig(idx, **generatorconfig)
        else:
            for k in range(4):
                generatorconfig = self.genpanel[k].getGeneratorConfig()
                self.driver.setGeneratorConfig(k, **generatorconfig)


    def changeMPUConfig(self):
        if self.driver is not None:
            for id in range(3):
                self.driver.setIMUConfig(id,self.imupanel[id].getIMUConfig())


    def changeADCConfig(self):        
        if self.driver is not None:
            self.driver.setADCConfig(self.adcpanel.getADCConfig())
        if self.mfig is not None:
            self.mfig.adcenable = (True in self.driver.adcenablemap)
            # self.mfig.adcEnableMap = self.driver.adcenablemap


    def addPlots(self):
        settings = QtCore.QSettings("VibSoftware", "VibView")
        self.ui.plotOutLayout.addWidget(self.mfig)
        self.ui.plotOutLayout.addWidget(self.ctrlfig)
        if self.ctrlpanel.isControlOn():
            self.mfig.hide()
        else:
            self.ctrlfig.hide()
        aux = settings.value('MFig')
        if aux is not None:
            # print(aux)
            self.mfig.parseConfigString(aux)
        aux = settings.value('CtrlFig')
        if aux is not None:
            self.ctrlfig.parseConfigString(aux)
        self.plotConfig()

        settings = QtCore.QSettings("VibSoftware", "VibView")
        aux = settings.value('OFig')
        if aux is not None:
            self.ofig.parseConfigString(aux)
        self.ui.plotLayout2.addWidget(self.ofig)


    def initChecks(self):
        if self.ctrlpanel.isControlOn():
            errimuidx = self.ctrlpanel.getErrorIMU()           
            if not self.imupanel[errimuidx].isIMUEnabled():
                raise BaseException("Error IMU is disabled.")
            refimuidx = self.ctrlpanel.getRefIMU()           
            if not self.imupanel[refimuidx].isIMUEnabled():
                raise BaseException("Reference IMU is disabled.")

    
    def bInit(self,stoptime=3600.0):
        self.ui.bInit.setEnabled(False)        
        if self.dataman.flagrodando:            
            self.dataman.ParaLeituras()
        else:
            try: 
                self.initChecks()
            except BaseException as ex:
                self.statusMessage(f"Error: {ex}")                
                self.ui.bInit.setEnabled(True)
                return
            self.expLog = []
            self.flagsaveplus = False
            self.configControl()
            self.changeGeneratorConfig()                   
            self.changeMPUConfig()
            self.changeADCConfig()
            self.plotConfig()
            self.plotOutConfig()
            self.ui.bLimpar.setEnabled(False)

            self.disabledwhenrunning = self.imupanel + [self.ctrlpanel,self.adcpanel]
            for cc in self.disabledwhenrunning: 
                cc.setEnabled(False)
            
            if self.ctrlpanel.isControlOn() and self.ctrlpanel.isTaskControl():
                self.ctrlpanel.ui.checkAlgOn.setEnabled(True)

            if (self.driver.adcconfig[0] & 0x0F) == 0:
                self.mfig.removeADCPlot()
            else:
                self.mfig.addADCPlot()

            for k in range(3):
                self.driver.setIMUConfig(k,self.imupanel[k].getIMUConfig())
            self.changeADCConfig()

            self.ui.comboMode.setEnabled(False)

            self.dataman.IniciaLeituras(stoptime=stoptime)

    def readingsStarted(self):
        if self.automator.running:
            self.ui.bInit.setEnabled(False)
        else:
            self.ui.bInit.setEnabled(True)
        if self.automatorWaitingStart:
            self.resumeAutomator.emit()
            self.automatorWaitingStart = False

    def readingsStopped(self):
        for cc in self.disabledwhenrunning:
            cc.setEnabled(True)
        self.ctrlpanel.ui.checkAlgOn.setEnabled(False)
        self.ctrlpanel.ui.checkAlgOn.setChecked(False)
        if (self.dataman.errorlevel > 0) and self.automator.running:
            self.automator.stop()
        elif self.automatorWaiting:
            self.resumeAutomator.emit()
        self.ui.bInit.setEnabled(True)
        self.ui.bLimpar.setEnabled(True)


    def bReset(self,ignoreunsaved=False):
        if self.dataman.flagrodando:
            self.ui.statusbar.showMessage("Reading must be stopped before erasing data.")
            return
        if (not self.dataman.flagsaved) and (not ignoreunsaved):
            buttonReply = QMessageBox.question(self, 'Warning!', "Data is not saved and will be erased. Confirm?",
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                self.dataman.resetData(self.TSampling/1e6,self.MaxTime)
                self.ui.statusbar.clearMessage()
            else:
                self.ui.statusbar.showMessage("Cancelado...")
        else:
            self.dataman.resetData(self.TSampling/1e6,self.MaxTime)
            self.ui.statusbar.clearMessage()
        self.ctrlpanel.setEnabled(True,isreset=True)
        self.ui.comboMode.setEnabled(True)

    def setMaxTime(self, selmaxtime):
        if self.dataman.flagrodando or (not self.dataman.flagsaved):
            QMessageBox.critical(self, 'Error', "Operation not allowed when running or with unsaved data.")
            # self.ui.statusbar.showMessage("Not allowed when running or with unsaved data.")
        else:
            if selmaxtime.startswith(">"):
                selmaxtime = selmaxtime[1:]
            self.MaxTime = int(selmaxtime[0:2])
            for sra in self.ui.menuMax_Time.actions():
                if sra.text().endswith(selmaxtime):
                    sra.setText(f">{selmaxtime}")
                else:
                    if sra.text().startswith(">"):
                        sra.setText(sra.text()[1:])
            self.dataman.resetData(self.TSampling/1e6,self.MaxTime)
            self.ui.statusbar.clearMessage()   

    def setBaudRate(self, selbaudrate):
        if self.dataman.flagrodando or (not self.dataman.flagsaved):
            QMessageBox.critical(self, 'Error', "Operation not allowed when running or with unsaved data.")
            # self.ui.statusbar.showMessage("Not allowed when running or with unsaved data.")
        else:
            if selbaudrate.startswith(">"):
                selbaudrate = selbaudrate[1:]
            self.BaudRate = int(selbaudrate)
            for sra in self.ui.menuSerial_Baud_Rate.actions():
                if sra.text().endswith(selbaudrate):
                    sra.setText(f">{selbaudrate}")
                else:
                    if sra.text().startswith(">"):
                        sra.setText(sra.text()[1:])
            # self.dataman.resetData(self.TSampling/1e6,self.MaxTime)
            self.driver.setBaudRate(self.BaudRate)
            self.ui.statusbar.clearMessage()

    def setSampling(self, selsampling):
        if self.dataman.flagrodando or (not self.dataman.flagsaved):
            QMessageBox.critical(self, 'Error', "Operation not allowed when running or with unsaved data.")
            # self.ui.statusbar.showMessage("Not allowed when running or with unsaved data.")
        else:
            if selsampling.endswith("ms"):
                selsampling = f"{(1 / (np.round(float(selsampling[:-2]) * 1e-3))):.0f} Hz" 
            if selsampling.startswith(">"):
                selsampling = selsampling[1:]
            # self.TSampling = int(selsampling[0])
            self.FSampling = int(selsampling[0:-2])
            for sra in self.ui.menuSampling_Rate.actions():
                if sra.text().endswith(selsampling):
                    sra.setText(f">{selsampling}")
                else:
                    if sra.text().startswith(">"):
                        sra.setText(sra.text()[1:])
            self.TSampling = int(np.round(1e6/self.FSampling))
            self.dataman.resetData(self.TSampling/1e6,self.MaxTime)
            self.mfig.setSamplingPeriod(self.dataman.samplingperiod)
            self.ofig.setSamplingPeriod(self.dataman.samplingperiod)
            self.ctrlfig.setSamplingPeriod(self.dataman.samplingperiod)
            self.ui.statusbar.clearMessage()   


    def populatePorts(self):
        self.ui.menuSelecionar_Porta.clear()
        ports = self.driver.listPorts()
        for port, desc, hwid in sorted(ports):
            # print("{}: {} [{}]".format(port, desc, hwid))
            if port == self.Port:
                port = f">{port}"
            self.ui.menuSelecionar_Porta.addAction(port)
        for acc in self.ui.menuSelecionar_Porta.actions():
            self.mapper.setMapping(acc, acc.text())
            acc.triggered.connect(self.mapper.map)
        

    def setPort(self, portasel):
        # print(portasel)
        if not self.dataman.flagrodando:
            if portasel.startswith(">"):
                portasel = portasel[1:]
            self.Port = portasel
            for acc in self.ui.menuSelecionar_Porta.actions():
                if acc.text().endswith(portasel):
                    acc.setText(f">{portasel}")
                else:
                    if acc.text().startswith(">"):
                        acc.setText(acc.text()[1:])
            self.ui.statusbar.clearMessage()
            self.driver.setPort(self.Port)
        else: 
            self.ui.statusbar.showMessage("Not allowed when running.")


    def openWorkdirManager(self):
        if not self.dataman.flagrodando:
            self.wdman = WorkdirManager(self.workdir)
            self.wdman.showWorkdirManager()


    def openUploadDialog(self):
        if not self.dataman.flagrodando:
            self.driver.setPort(self.Port)
            self.upd = MyUploadDialog(self.driver,self.dataman)
            self.upd.showUploadDialog()


    def openPathModelingDialog(self):
        if not self.dataman.flagrodando:    
            self.driver.setPort(self.Port)        
            self.pmd = MyPathModelingDialog(self.dataman,self.driver)
            self.pmd.showPathModelingdDialog()
    

    def openDataViewer(self):
        if not self.dataman.flagrodando:
            self.dvd = MyDataViewer(self.dataman)
            self.dvd.showDataViewerDialog()


    def openAdditionalConfig(self):
        if not self.dataman.flagrodando:
            self.pdd = MyAdditionalDialog(self.driver)
            self.pdd.showAdditionalDialog()


    def saveFile(self,fname=None):
        if self.dataman.flagrodando:
            self.ui.statusbar.showMessage("Readings must be stopped before saving file.")
            return
        if fname:
            filename = [fname]
        else:
            txtfilter = 'Feather file (*.feather);; CSV file (*.csv)'
            if self.lastSavedExtension == ".csv":
                txtfilter = 'CSV file (*.csv);; Feather file (*.feather)'
            filename = QFileDialog.getSaveFileName(self, "Salvar Arquivo", getenv('HOME'), txtfilter)
            # print(filename)
            if filename[0].endswith(".csv"):
                self.lastSavedExtension = ".csv"
            elif filename[0].endswith(".feather"):
                self.lastSavedExtension = ".feather"
            else:
                if filename[1].startswith("Feather"):
                    filename = (filename[0]+".feather",filename[1])
                    self.lastSavedExtension = ".feather"
                elif filename[1].startswith("CSV"):
                    filename = (filename[0]+".csv",filename[1])
                    self.lastSavedExtension = ".csv"
        if (filename[0] != ''):
            try:
                notes = self.ui.notes.toPlainText()
                if len(notes) > 0:
                    self.loglist = [(0,notes)] + self.loglist
                self.dataman.salvaArquivo(str(filename[0]), True, loglist=self.loglist)
            except Exception as err:
                # QMessageBox.question(self.app, "Erro!", str(err), QMessageBox.Ok)
                print(err)
                self.ui.statusbar.showMessage(err)


    def closeEvent(self, event):
        if not event:
            self.writeConfig()
            event = QtGui.QCloseEvent()
            event.accept = self.close
            event.ignore = (lambda *args: None)
        if self.dataman.flagrodando:
            self.ui.statusbar.showMessage("Pare as leituras antes de fechar...")
            event.ignore()
        elif not self.dataman.flagsaved:
            reply = QMessageBox.question(self, 'Saindo', 'Existem dados não salvos, deseja mesmo sair?',
                                         QMessageBox.No | QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.writeConfig()
                event.accept()
            else:
                event.ignore()
        else:
            self.writeConfig()
            event.accept()
