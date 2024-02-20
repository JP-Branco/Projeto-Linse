from PySide2 import (QtCore, QtWidgets, QtGui)
from .IMUPanel import Ui_IMUPanel
from .GeneratorPanel import Ui_GeneratorPanel
from .ControlPanel import Ui_ControlForm
from .ADCPanel import Ui_ADCForm


class StateSaver:

    def saveState(self, settings: QtCore.QSettings):
        if self.saveprefix is None:
            self.saveprefix = ""
        wdgs = self.findChildren(QtWidgets.QComboBox) 
        for ww in wdgs:
            keyval = f'{self.saveprefix}{ww.objectName()}'
            settings.setValue(keyval, ww.currentIndex())
            settings.setValue(f'EN{keyval}', str(ww.isEnabled()))
        wdgs = self.findChildren(QtWidgets.QCheckBox)
        for ww in wdgs:
            keyval = f'{self.saveprefix}{ww.objectName()}'
            settings.setValue(keyval, str(ww.isChecked()))
            settings.setValue(f'EN{keyval}', str(ww.isEnabled()))
        wdgs = self.findChildren(QtWidgets.QDoubleSpinBox) + self.findChildren(QtWidgets.QSpinBox)
        for ww in wdgs:
            keyval = f'{self.saveprefix}{ww.objectName()}'
            # print(f'{keyval} - {ww.isEnabled()}')
            settings.setValue(keyval, ww.value())
            settings.setValue(f'EN{keyval}', str(ww.isEnabled()))

    
    def restoreState(self, settings: QtCore.QSettings):
        if self.saveprefix is None:
            self.saveprefix = ""

        wdgs = self.findChildren(QtWidgets.QComboBox,QtCore.QRegExp("combo.*")) 
        for ww in wdgs:
            keyval = f'{self.saveprefix}{ww.objectName()}'
            if settings.value(keyval) is not None:
                ww.setCurrentIndex(int(settings.value(keyval)))
            if settings.value(f'EN{keyval}') is not None:
                ww.setEnabled(settings.value(f'EN{keyval}') == 'True')

        wdgs = self.findChildren(QtWidgets.QCheckBox,QtCore.QRegExp("check.*"))
        for ww in wdgs:
            keyval = f'{self.saveprefix}{ww.objectName()}'
            if settings.value(keyval) is not None:
                ww.setChecked(settings.value(keyval) == 'True')
            if settings.value(f'EN{keyval}') is not None:
                ww.setEnabled(settings.value(f'EN{keyval}') == 'True')
        
        wdgs = self.findChildren(QtWidgets.QDoubleSpinBox)
        for ww in wdgs:
            keyval = f'{self.saveprefix}{ww.objectName()}'
            # aux = settings.value(f'EN{keyval}')
            # print(f'{keyval} - {aux}')
            if settings.value(keyval) is not None:
                ww.setValue(float(settings.value(keyval)))
            if settings.value(f'EN{keyval}') is not None:
                ww.setEnabled(settings.value(f'EN{keyval}') == 'True')
        
        wdgs = self.findChildren(QtWidgets.QSpinBox)
        for ww in wdgs:
            keyval = f'{self.saveprefix}{ww.objectName()}'
            if settings.value(keyval) is not None:
                ww.setValue(int(settings.value(keyval)))
            if settings.value(f'EN{keyval}') is not None:
                ww.setEnabled(settings.value(f'EN{keyval}') == 'True')


class ControlPanel(QtWidgets.QWidget,StateSaver):

    def __init__(self):
        super().__init__()
        self.id = id
        self.ui = Ui_ControlForm()
        self.ui.setupUi(self)
        self.saveprefix = 'CTRL'
        # self.ui.checkControle.toggled.connect(self.controlChanged)
        self.controlenabled = False
        self.ui.checkAlgOn.setChecked(False)
        # self.ui.checkAlgOn.setEnabled(False)
        # self.ui.checkAlgOn.toggled.connect(self.controlChanged)
        self.ui.passoCtrl.setValidator(QtGui.QDoubleValidator(0.0, 1000.0, 2, self))
        self.ui.passoCtrl.editingFinished.connect(self.validateStep)
        self.ui.normCtrl.setValidator(QtGui.QDoubleValidator(0.0, 1000.0, 2, self))
        self.ui.normCtrl.editingFinished.connect(self.validateRegularization)
        self.controlChangeFunc = None
        self.oldctrlmu = float(self.ui.passoCtrl.text())
        self.oldctrlpsi = float(self.ui.normCtrl.text())
        self.ui.comboControlChannel.activated.connect(self.controlChannelChanged)
        self.ui.comboPerturbChannel.activated.connect(self.perturbChannelChanged)
        self.ui.comboCtrlTask.activated.connect(self.taskChanged)
        self.cmps = [self.ui.comboRef,self.ui.comboErro,
                self.ui.comboIMUError,self.ui.comboIMURef,
                self.ui.comboPerturbChannel,self.ui.comboControlChannel]
        self.cmpscontrolon = [self.ui.comboAlgoritmo,self.ui.spinMemCtrl]
        self.cmpscontrolalwayson = [self.ui.checkAlgOn,self.ui.spinTAlgOn,self.ui.passoCtrl,self.ui.normCtrl]
    
    def controlChannelChanged(self):
        if self.ui.comboControlChannel.currentIndex() == self.ui.comboPerturbChannel.currentIndex():
            self.ui.comboPerturbChannel.setCurrentIndex( (self.ui.comboControlChannel.currentIndex() + 1) % 4 )
        self.controlChanged()
    
    def perturbChannelChanged(self):
        if self.ui.comboControlChannel.currentIndex() == self.ui.comboPerturbChannel.currentIndex():
            self.ui.comboControlChannel.setCurrentIndex( (self.ui.comboPerturbChannel.currentIndex() + 1) % 4 )
        self.controlChanged()
    
    def connectControlChanged(self,controlchangefunc):
        self.controlChangeFunc = controlchangefunc
    
    def setActive(self, active : bool):
        self.controlenabled = active
        self.setEnabled(active,True)

    def isControlOn(self):
        # return self.ui.checkControle.isChecked()
        return self.controlenabled

    def taskChanged(self):
        self.setEnabled(True,True)

    def isTaskControl(self):
        return True if (self.ui.comboCtrlTask.currentIndex() == 0) else False

    def getControlTask(self):
        '''
           Control Task: 0 = Active Control / 1 = Path Modelling 
        '''
        return self.ui.comboCtrlTask.currentIndex()
    
    def isAlgOn(self):
        return self.ui.checkAlgOn.isChecked()
    
    def setEnabled(self,en,isreset=False):
        if not self.controlenabled:
            for cc in (self.cmps + self.cmpscontrolon + self.cmpscontrolalwayson):
                cc.setEnabled(False)  
            self.ui.comboCtrlTask.setEnabled(False)
        else:
            if self.isTaskControl():
                for cc in self.cmpscontrolon:
                    cc.setEnabled(en)
                for cc in self.cmpscontrolalwayson:
                    cc.setEnabled(True)                
            else: 
                for cc in (self.cmpscontrolon + self.cmpscontrolalwayson):
                    cc.setEnabled(False)        
            for cc in self.cmps:
                cc.setEnabled(en)
            if (not en) or (isreset and en):
                self.ui.comboCtrlTask.setEnabled(True)

    def controlChanged(self):
        if self.controlChangeFunc is not None:
            self.controlChangeFunc()

    def validateStep(self):
        try:
            self.oldctrlmu = float(self.ui.passoCtrl.text())
        except Exception as exc:
            self.ui.passoCtrl.setText(str(self.oldctrlmu))

    def validateRegularization(self):
        try:
            self.oldctrlpsi = float(self.ui.normCtrl.text())
        except Exception as exc:
            self.ui.normCtrl.setText(str(self.oldctrlpsi))

    def getControlChannel(self):
        return self.ui.comboControlChannel.currentIndex()
    
    def getPerturbChannel(self):
        return self.ui.comboPerturbChannel.currentIndex()

    def getErrorIMU(self):
        return self.ui.comboIMUError.currentIndex()
    
    def getRefIMU(self):
        return self.ui.comboIMURef.currentIndex()

    def getControlConfiguration(self):
        try:
            fifloat = float(self.ui.normCtrl.text())            
        except:
            self.ui.normCtrl.setText("0.01")
            fifloat = 0.01
        try:
            mufloat = float(self.ui.passoCtrl.text())
        except:
            self.ui.passoCtrl.setText("0.01")
            mufloat = 0.01
        data = {
            "alg": self.ui.comboAlgoritmo.currentIndex(),
            "mem": int(self.ui.spinMemCtrl.value()),
            "mu": mufloat,
            "fi": fifloat,
            "refimuid": self.ui.comboIMURef.currentIndex(),
            "errimuid": self.ui.comboIMUError.currentIndex(),
            "refid": self.ui.comboRef.currentIndex(), # from 0 to 5, from AccX to GyroZ
            "erroid": self.ui.comboErro.currentIndex(),
            "ctrltask": self.ui.comboCtrlTask.currentIndex() # 0 for control, 1 for path modeling
        }
        return data

    def getLogString(self):
        logstring = "Enabled" if self.controlenabled else "Disabled"
        if logstring == "Enabled":
            info = [self.ui.comboCtrlTask.currentText(), self.ui.comboPerturbChannel.currentText(), self.ui.comboControlChannel.currentText(),
                    self.ui.comboIMURef.currentText(),self.ui.comboRef.currentText(),self.ui.comboIMUError.currentText(),self.ui.comboErro.currentText(),
                    "AlgOn" if self.ui.checkAlgOn.isChecked() else "AlgOff",str(self.ui.spinTAlgOn.value()),self.ui.comboAlgoritmo.currentText(),
                    str(self.ui.spinMemCtrl.value()),self.ui.passoCtrl.text(),self.ui.normCtrl.text()]
            logstring = logstring + "|" + "|".join(info)
        return logstring
    
    def getAlgOnLogString(self):
        logstring = "AlgOn" if self.ui.checkAlgOn.isChecked() else "AlgOff"
        if logstring == "AlgOn":
            logstring = logstring + "|" + self.ui.passoCtrl.text() + "|" + self.ui.normCtrl.text() + "|" + self.ui.spinTAlgOn.text()
        return logstring


class GeneratorPanel(QtWidgets.QWidget,StateSaver):

    def __init__(self, id: int):
        super().__init__()
        self.id = id
        self.ui = Ui_GeneratorPanel()
        self.ui.setupUi(self)
        self.saveprefix = f'GEN{id}'
        if id < 2:
            self.ui.spinDCLevel.setMaximum(2048)
        else:
            self.ui.spinDCLevel.setMaximum(128)
        self.ui.comboType.activated.connect(self.typechanged)
    
    def isGeneratorOn(self):
        return self.ui.checkEnable.isEnabled() and self.ui.checkEnable.isChecked()

    def typechanged(self):
        idx = self.ui.comboType.currentIndex()
        if idx == 0:  # Noise
            for aa in [self.ui.spinFreq,self.ui.chirpDeltaI,self.ui.chirpTfim,self.ui.chirpDeltaF,self.ui.chirpA2,self.ui.chirpTinicio]:
                aa.setEnabled(False)
            self.ui.spinAmpl.setEnabled(True)
        elif idx == 1:  # Harmonic
            for aa in [self.ui.chirpDeltaI,self.ui.chirpTfim,self.ui.chirpDeltaF,self.ui.chirpA2,self.ui.chirpTinicio]:
                aa.setEnabled(False)
            self.ui.spinAmpl.setEnabled(True)
            self.ui.spinFreq.setEnabled(True)
        elif idx == 2:  # Chirp
            for aa in [self.ui.spinAmpl,self.ui.spinFreq,self.ui.chirpDeltaI,self.ui.chirpTfim,self.ui.chirpDeltaF,self.ui.chirpA2,self.ui.chirpTinicio]:
                aa.setEnabled(True)
        elif idx == 3:  # Step
            for aa in [self.ui.chirpDeltaI,self.ui.chirpDeltaF,self.ui.chirpA2,self.ui.spinFreq]:
                aa.setEnabled(False)
            for aa in [self.ui.chirpTinicio,self.ui.chirpTfim,self.ui.spinAmpl]:
                aa.setEnabled(True)
        elif idx == 4: # Square
            for aa in [self.ui.chirpDeltaI,self.ui.chirpTfim,self.ui.chirpDeltaF,self.ui.chirpA2,self.ui.chirpTinicio]:
                aa.setEnabled(False)
            self.ui.spinAmpl.setEnabled(True)
            self.ui.spinFreq.setEnabled(True)
    
    def setGeneratorConfig(self,signaltype,amp,freq,dclevel=None):
        if isinstance(signaltype, str):
            self.ui.comboType.setCurrentText(signaltype)
        elif isinstance(signaltype, int):
            self.ui.comboType.setCurrentIndex(signaltype)
        self.ui.spinAmpl.setValue(amp)
        self.ui.spinFreq.setValue(freq)
        if dclevel:
            self.ui.spinDCLevel.setValue(dclevel)

    
    def setEnabled(self,en):
        self.ui.checkEnable.setEnabled(en)
        self.ui.comboType.setEnabled(en)
        self.ui.spinAmpl.setEnabled(en)
        self.ui.spinFreq.setEnabled(en)
        self.ui.spinDCLevel.setEnabled(en)
    
    def getGeneratorConfig(self):
        if self.isGeneratorOn():
            generatorconfig = {'tipo': self.ui.comboType.currentIndex(),
                                'amp': self.ui.spinAmpl.value(),
                                'freq': self.ui.spinFreq.value(),
                                'dclevel': self.ui.spinDCLevel.value()}
            if self.ui.comboType.currentIndex() == 2:  # Chirp
                generatorconfig['chirpconf'] = [self.ui.chirpTinicio.value(),
                                                self.ui.chirpDeltaI.value(),
                                                self.ui.chirpTfim.value(),
                                                self.ui.chirpDeltaF.value(),
                                                self.ui.chirpA2.value()]
            return generatorconfig
        else:
            return {'tipo': 0, 'amp': 0, 'freq': 0, 'dclevel': self.ui.spinDCLevel.value()}
    
    def getLogString(self):
        logstring = "Enabled" if self.isGeneratorOn() else "Disabled"
        if logstring == "Enabled":
            info = [self.ui.comboType.currentText(),
                    str(self.ui.spinAmpl.value()), str(self.ui.spinFreq.value()), str(self.ui.spinDCLevel.value())]
            if self.ui.comboType.currentText() == "Chirp":
                info.append(f"{self.ui.chirpTinicio.value()},{self.ui.chirpDeltaI.value()},{self.ui.chirpTfim.value()},{self.ui.chirpDeltaF.value()},{self.ui.chirpA2.value()}")
            logstring = logstring + "|" + "|".join(info)
        return logstring


class IMUPanel(QtWidgets.QWidget,StateSaver):

    def __init__(self, id: int):
        super().__init__()
        self.ui = Ui_IMUPanel()
        self.ui.setupUi(self)
        self.ui.comboType.activated.connect(self.typechanged)
        self.ui.comboAddress.highlighted.connect(self.addresshighlight)
        self.ui.comboBus.activated.connect(self.addresshighlight)
        self.saveprefix = f'IMU{id}'

    '''
        IMU config data:

        Byte 0 -| Enabled (1 bit) (LSB)
                | Type: MPU, LSM (1 bit)
                | Bus selection (2 bits, for future expansion)

        Byte 1 - Address (8 bits)
                
        Byte 2 -| AccRange (2 bits) (LSBs)
                | GyroRange (3 bits) (LSM Gyro have 5 options)
                | Filter config (3 bits)
    '''
    def getIMUConfig(self):
        configbytes = []        
        # Byte 0:
        imutype = int(self.ui.comboType.currentIndex() == 2)
        imuenabled = int(self.ui.comboType.currentIndex() > 0)
        configbytes.append( (self.ui.comboBus.currentIndex()<<2) + (imutype << 1) +  imuenabled  )
        # Byte 1:
        if self.ui.comboBus.currentIndex() < 2:
            configbytes.append( int(str(self.ui.comboAddress.currentText())[2:],16) )
        else: 
            if self.ui.comboAddress.currentIndex() < 4:
                self.ui.comboAddress.setCurrentIndex(4)
            configbytes.append( int(str(self.ui.comboAddress.currentText())[2:],10) )
        # Byte 2:
        configbytes.append( (self.getFilter() << 5) + (self.getGyroRange()<<2) + self.getAccRange() )
        return configbytes

    def getMPUAddress(self):
        return self.ui.comboAddress.currentIndex()
    
    def getFilter(self):
        return self.ui.comboFilter2.currentIndex()

    def getAccRange(self):
        return self.ui.comboAccRange.currentIndex()

    def getGyroRange(self):
        aux = self.ui.comboGyroRange.currentIndex()
        if self.ui.comboType.currentIndex() == 1:            
            if aux < 1: 
                self.ui.comboGyroRange.setCurrentIndex(1)
                aux = 1
            return aux
        elif self.ui.comboType.currentIndex() == 2:
            return aux
        else:
            return 0
    
    def setEnabled(self,en):
        if self.ui.comboType.currentIndex() > 0:
            cmps = [self.ui.comboAddress,self.ui.comboAccRange,self.ui.comboGyroRange,#self.ui.comboFilter,
                    self.ui.comboBus,self.ui.comboType,self.ui.comboFilter2]
            for cc in cmps:
                cc.setEnabled(en)
        else:
            self.ui.comboType.setEnabled(en)
    
    def isIMUEnabled(self):
        return self.ui.comboType.currentIndex() > 0

    def addresshighlight(self):
        if self.ui.comboType.currentIndex() == 1:
            self.ui.comboAddress.model().item(0).setEnabled(True)
            self.ui.comboAddress.model().item(1).setEnabled(True)
            self.ui.comboAddress.model().item(2).setEnabled(False)
            self.ui.comboAddress.model().item(3).setEnabled(False)
            self.ui.comboAddress.model().item(4).setEnabled(False)
            self.ui.comboAddress.model().item(5).setEnabled(False)
            self.ui.comboGyroRange.model().item(0).setEnabled(False)
            self.ui.comboBus.model().item(2).setEnabled(False)
            if self.ui.comboGyroRange.currentIndex() == 0:
                self.ui.comboGyroRange.setCurrentIndex(1)
            if self.ui.comboAddress.currentIndex() > 1:
                self.ui.comboAddress.setCurrentIndex(0)
            # self.ui.comboFilter.setEnabled(True)
            # self.ui.comboFilter2.setEnabled(False)
            self.ui.comboFilter2.clear()
            self.ui.comboFilter2.addItems(["260 Hz","184 Hz","94 Hz","44 Hz","21 Hz","10 Hz","5 Hz"])
        elif self.ui.comboType.currentIndex() == 2:
            self.ui.comboAddress.model().item(0).setEnabled(False)
            self.ui.comboAddress.model().item(1).setEnabled(False)
            if self.ui.comboBus.currentIndex() == 2:
                self.ui.comboAddress.model().item(2).setEnabled(False)
                self.ui.comboAddress.model().item(3).setEnabled(False)
                self.ui.comboAddress.model().item(4).setEnabled(True)
                self.ui.comboAddress.model().item(5).setEnabled(True)
                if self.ui.comboAddress.currentIndex() < 4:
                    self.ui.comboAddress.setCurrentIndex(4)
            else:
                self.ui.comboAddress.model().item(2).setEnabled(True)
                self.ui.comboAddress.model().item(3).setEnabled(True)
                self.ui.comboAddress.model().item(4).setEnabled(False)
                self.ui.comboAddress.model().item(5).setEnabled(False)
                if (self.ui.comboAddress.currentIndex() < 2) or (self.ui.comboAddress.currentIndex() > 3):
                    self.ui.comboAddress.setCurrentIndex(2)
            self.ui.comboGyroRange.model().item(0).setEnabled(True)
            self.ui.comboBus.model().item(2).setEnabled(True)
            if self.ui.comboAddress.currentIndex() < 2:
                self.ui.comboAddress.setCurrentIndex(2)
            self.ui.comboFilter2.clear()
            self.ui.comboFilter2.addItems(["Auto","400 Hz","200 Hz","100 Hz","50 Hz"])
            # self.ui.comboFilter.setEnabled(False)
            # self.ui.comboFilter2.setEnabled(True)

    def typechanged(self):
        cmps = [self.ui.comboAddress,self.ui.comboAccRange,self.ui.comboGyroRange,self.ui.comboFilter2,self.ui.comboBus]
        if self.ui.comboType.currentIndex() == 0:
            enabled = False
        else:
            enabled = True
        for cc in cmps:
            cc.setEnabled(enabled)
        self.addresshighlight()

    def getLogString(self):
        logstring = self.ui.comboType.currentText()
        if logstring != "Disabled":
            info = [self.ui.comboAddress.currentText(),self.ui.comboBus.currentText(),self.ui.comboAccRange.currentText(),
                    self.ui.comboGyroRange.currentText(), 
                    self.ui.comboFilter2.currentText() ]
            logstring = logstring + "|" + "|".join(info)
        return logstring
        
        
class ADCPanel(QtWidgets.QWidget,StateSaver):

    def __init__(self):
        super().__init__()
        self.ui = Ui_ADCForm()
        self.ui.setupUi(self)
        self.saveprefix = f'ADC'
        self.ui.comboADCModel.currentIndexChanged.connect(lambda: self.modelChanged(False))

    #
    # adcconfig[0]: less significant bits = channel enablers
    # adcconfig[1]: ranges
    # adcconfig[2]: rates
    #
    def getADCConfig(self):
        isADC1115 = (self.ui.comboADCModel.currentIndex() == 0)
        flagon = (0 if isADC1115 else 16) + \
                 (1 if self.ui.radioADC1.isChecked() else 0) + \
                 (2 if self.ui.radioADC2.isChecked() else 0) + \
                 (4 if self.ui.radioADC3.isChecked() else 0) + \
                 (8 if self.ui.radioADC4.isChecked() else 0)
        adcconfig = [flagon,
                     self.ui.comboADCRange.currentIndex(),
                     self.ui.comboRate1115.currentIndex() if isADC1115 else self.ui.comboRate1015.currentIndex()]
        return adcconfig

    def setEnabled(self,en):
        for cc in [self.ui.radioADC1,self.ui.radioADC2,self.ui.radioOff,
                   self.ui.radioADC3,self.ui.radioADC4,
                   self.ui.comboADCModel,self.ui.comboADCRange,self.ui.comboRate1115,self.ui.comboRate1015]:
            cc.setEnabled(en)
        self.modelChanged(not en)
    
    def modelChanged(self,disableall=False):
        isADC1115 = (self.ui.comboADCModel.currentIndex() == 0)
        if disableall:
            self.ui.comboRate1115.setEnabled(False)
            self.ui.comboRate1015.setEnabled(False)
        else: 
            self.ui.comboRate1115.setEnabled(isADC1115)
            self.ui.comboRate1015.setEnabled(not isADC1115)
    
    def getLogString(self):
        logstring = self.ui.comboADCModel.currentText()
        info = [ "1" if self.ui.radioADC1.isChecked() else "0",
                 "1" if self.ui.radioADC2.isChecked() else "0",
                 "1" if self.ui.radioADC3.isChecked() else "0",
                 "1" if self.ui.radioADC4.isChecked() else "0",
                 self.ui.comboADCRange.currentText(),
                 self.ui.comboRate1015.currentText() if (logstring == "ADS1015") else self.ui.comboRate1115.currentText() ]
        logstring = logstring + "|" + "|".join(info)
        return logstring



