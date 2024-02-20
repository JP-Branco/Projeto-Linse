"""
Helper classes/methods
"""

import numpy as np
import pyqtgraph as pg

from .BaseFigQtGraph import BaseFigQtGraph

class CtrlFigQtGraph(BaseFigQtGraph):

    def __init__(self, dman, app):
        self.dman = dman
        self.app = app
        super().__init__(2, self.dman.samplingperiod)
        pens = [pg.mkPen('r', width=1), pg.mkPen('b', width=1)]
        self.lineref = self.pitens[0].plot(np.array([]), np.array([]), pen=pens[0])
        self.lineerr = self.pitens[1].plot(np.array([]), np.array([]), pen=pens[1])
        self.plotSetup()
        self.getMenu(0).disableSensor()
        self.getMenu(1).disableSensor()

    def plotSetup(self, sensorids=[0, 6]):
        for k in range(2):
            self.pitens[k].disableAutoRange()
            self.pitens[k].setLabel('bottom', 'Time (s)')
            if (sensorids[k] % 6) < 3:
                self.pitens[k].setLabel('left', f'{"Reference" if k == 0 else "Error"}<br>Acceleration (m/s²)')
                # self.miny[k] = -0.25
                # self.maxy[k] = 0.25
            else:
                self.pitens[k].setLabel('left', f'{"Reference" if k == 0 else "Error"}<br>Gyro (dg/s)')
                # self.miny[k] = -10
                # self.maxy[k] = 10
            self.pitens[k].setXRange(-self.janelax[k], 0, padding=0.01)
            self.pitens[k].setYRange(self.miny[k], self.maxy[k], padding=0.01)

    def updateFig(self):
        limi = [0, 0]
        limf = [0, 0]
        npontos = [0, 0]
        for k in range(2):
            limi[k] = self.dman.globalctreadings - self.npontosjanela[k]
            limf[k] = self.dman.globalctreadings
            if limi[k] < 0:
                limi[k] = 0
                npontos[k] = limf[k] - limi[k]
            else:
                npontos[k] = self.npontosjanela[k]
        if npontos[0] > 0:
            self.lineref.setData(self.vetoreixox[0][-npontos[0]:], self.dman.xrefdata[limi[0]:limf[0]])
            self.lineerr.setData(self.vetoreixox[1][-npontos[1]:], self.dman.xerrodata[limi[1]:limf[1]])
        else:
            self.lineref.setData([], [])
            self.lineerr.setData([], [])


class MyFigQtGraph(BaseFigQtGraph):

    def __init__(self, dman): # , app):
        self.dman = dman
        # self.app = app
        super().__init__(3, self.dman.samplingperiod)
        self.wsize = 60
        pens = [pg.mkPen('r', width=1), pg.mkPen('b', width=1), pg.mkPen('g', width=1), pg.mkPen('y', width=1)]
        
        self.plotlines = []
        # self.linesp1 = []
        # self.linesp2 = []        
        for k in range(3):
            self.plotlines.append(self.pitens[0].plot(np.array([]), np.array([]), pen=pens[k]))
            # self.linesp1.append(self.pitens[0].plot(np.array([]), np.array([]), pen=pens[k]))
            # self.linesp2.append(self.pitens[1].plot(np.array([]), np.array([]), pen=pens[k]))
        for k in range(3):
            self.plotlines.append(self.pitens[1].plot(np.array([]), np.array([]), pen=pens[k]))        

        # self.lineadc = []
        # for k in range(4):
        #     self.lineadc.append(self.pitens[2].plot(np.array([]), np.array([]), pen=pens[k]))
        self.lineadc = self.pitens[2].plot(np.array([]), np.array([]), pen=pens[0])
        for k, pitem in enumerate(self.pitens):
            pitem.disableAutoRange(axis=pg.ViewBox.YAxis)
            pitem.enableAutoRange(axis=pg.ViewBox.XAxis)
            pitem.setXRange(-self.janelax[k], 0, padding=0.01)
            pitem.setYRange(self.miny[k], self.maxy[k], padding=0.01)
            pitem.setLabel('bottom', 'Time (s)')
            pitem.setMouseEnabled(x=False,y=True)
            # for act in pitem.ctrlMenu.actions():
            #     act.setVisible(False)
            #     print(act)
        self.pitens[0].setLabel('left', 'Accel. (m/s²)')
        self.pitens[1].setLabel('left', 'Gyro (dg/s)')
        self.pitens[2].setLabel('left', 'ADC (Volts)')
        # self.getMenu(0).disableSensor()
        # self.getMenu(1).disableSensor()
        self.getMenu(2).disableSensor()
        self.p1Enable = [True, True, True]
        self.p2Enable = [True, True, True]
        # self.adcEnableMap = [False, False, False, False]
        self.adcenable = False
        self.plotchoice = [0, 0]
        self.getComboSensor(0).activated.connect(self.loadSensorChoices)
        self.getComboSensor(1).activated.connect(self.loadSensorChoices)        
        ffs = [lambda: self.toggleXYZSignal(0),lambda: self.toggleXYZSignal(1)]
        for k in range(2):            
            for chk in self.getXYZChecks(k):
                chk.toggled.connect(ffs[k])

    def toggleXYZSignal(self,plotid):
        if plotid == 0:
            self.p1Enable = self.getXYZCheckMap(plotid)
        else:
            self.p2Enable = self.getXYZCheckMap(plotid)
        # self.getMenu(plotid).hide()
        
    def setPlotChoice(self, idx1, idx2):
        if idx1 is not None:
            self.plotchoice[0] = idx1
        if idx2 is not None:
            self.plotchoice[1] = idx2        
    
    def loadSensorChoices(self):
        for k in range(2):
            self.getMenu(k).hide()
            newchoicek = self.getSensorChoice(k)
            if newchoicek != self.plotchoice[k]:                
                self.plotchoice[k] = newchoicek                
                if self.plotchoice[k] == 0:
                    self.pitens[k].setLabel('left', f'Accel. (m/s²)')
                elif self.plotchoice[k] < 4:
                    # self.pitens[k].setTitle(f'IMU { ((self.plotchoice[0]-1) % 3) + 1 }')
                    self.pitens[k].setLabel('left', f'IMU { ((self.plotchoice[k]-1) % 3) + 1 }<br>Accel. (m/s²)')
                else:
                    # self.pitens[k].setTitle(f'IMU { ((self.plotchoice[0]-1) % 3) + 1 }')
                    self.pitens[k].setLabel('left', f'IMU { ((self.plotchoice[k]-1) % 3) + 1 }<br>Gyro (dg/s)')
        
    def removeADCPlot(self):
        if self.getItem(2,0) is not None:
            self.removeItem(self.pitens[2])

    def addADCPlot(self):
        if self.getItem(2,0) is None:
            self.addItem(self.pitens[2],row=2,col=0)

    def updateFig(self):
        limi = [0, 0, 0]
        limf = [0, 0, 0]
        npontos = [0, 0, 0]
        for k in range(3):
            limi[k] = self.dman.globalctreadings - self.npontosjanela[k]
            limf[k] = self.dman.globalctreadings
            if limi[k] < 0:
                limi[k] = 0
                npontos[k] = limf[k] - limi[k]
            else:
                npontos[k] = self.npontosjanela[k]

        if (self.plotchoice[0] > 0) and (npontos[0] > 0):    
            imuidx = (self.plotchoice[0]-1) % 3
            if self.plotchoice[0] < 4:
                for k in range(3):
                    if self.p1Enable[k]:
                        self.plotlines[k].setData(self.vetoreixox[0][-npontos[0]:], self.dman.accdata[imuidx][k][limi[0]:limf[0]])
                    else:
                        self.plotlines[k].setData([], [])
            else:
                for k in range(3):
                    if self.p1Enable[k]:
                        self.plotlines[k].setData(self.vetoreixox[0][-npontos[0]:], self.dman.gyrodata[imuidx][k][limi[0]:limf[0]])
                    else: 
                        self.plotlines[k].setData([], [])
        else: 
            for k in range(3):
                self.plotlines[k].setData([], [])
        

        if (self.plotchoice[1] > 0) and (npontos[1] > 0):    
            imuidx = (self.plotchoice[1]-1) % 3
            if self.plotchoice[1] < 4:
                for k in range(3):
                    if self.p2Enable[k]:
                        self.plotlines[k+3].setData(self.vetoreixox[1][-npontos[1]:], self.dman.accdata[imuidx][k][limi[1]:limf[1]])
                    else:
                        self.plotlines[k+3].setData([], [])
            else:
                for k in range(3):
                    if self.p2Enable[k]:
                        self.plotlines[k+3].setData(self.vetoreixox[1][-npontos[1]:], self.dman.gyrodata[imuidx][k][limi[1]:limf[1]])
                    else: 
                        self.plotlines[k+3].setData([], [])
        else: 
            for k in range(3):
                self.plotlines[k+3].setData([], [])   
                
        if self.adcenable:
            if (npontos[2] > 0):
                self.lineadc.setData(self.vetoreixox[2][-npontos[2]:],self.dman.adcdata[0][limi[2]:limf[2]])
            else:
                self.lineadc.setData([],[])
            


class FigOutputQtGraph(BaseFigQtGraph):

    def __init__(self, dman, app):
        self.dman = dman
        self.app = app
        super().__init__(1, self.dman.samplingperiod)
        self.pitem = self.pitens[0]
        pens = [pg.mkPen('r', width=1), pg.mkPen('b', width=1), pg.mkPen('g', width=1), pg.mkPen('y', width=1)]
        self.lines = []
        for k in range(4):
            self.lines.append(self.pitem.plot(np.array([]), np.array([]), pen=pens[k]))
            self.lines[k].setAlpha(0.5, False)
        self.pitem.disableAutoRange()
        self.pitem.setXRange(-self.janelax[0], 0, padding=0.01)
        self.pitem.setYRange(self.miny[0], self.maxy[0], padding=0.01)
        self.pitem.setLabel('bottom', 'Time (s)')
        self.pitem.setLabel('left', 'Normalized output')
        self.oldctr = 0
        self.dacenable = [True, False, False, False]
        self.getMenu(0).disableSensor()

    def updateFig(self):

        limi = self.dman.globalctreadings - self.npontosjanela[0]
        limf = self.dman.globalctreadings

        if limi < 0:
            limi = 0
            npontos = limf - limi
        else:
            npontos = self.npontosjanela[0]

        if self.dman.ctrlmode:
            for k in range(2):
                if (npontos > 0):
                    self.lines[k].setData(self.vetoreixox[0][-npontos:], self.dman.dacoutdata[k][limi:limf])
                else:
                    self.lines[k].setData([], [])
            self.lines[2].setData([], [])
            self.lines[3].setData([], [])
        else:
            for k in range(4):
                if self.dacenable[k] and (npontos > 0):
                    self.lines[k].setData(self.vetoreixox[0][-npontos:], self.dman.dacoutdata[k][limi:limf])
                else:
                    self.lines[k].setData([], [])
