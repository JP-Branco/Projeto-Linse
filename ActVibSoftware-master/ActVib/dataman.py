"""
    Data Manager Module
"""
from threading import Thread
import time
import numpy as np
import pandas as pd

from .driverhardware import driverhardware

from PySide2.QtCore import Signal,QObject

class dataman (QObject):

    updateFigures = Signal()
    reset = Signal()
    statusMessage = Signal(str)
    stopped = Signal()
    started = Signal()
    logMessage = Signal((int,str))

    def __init__(self, driver : driverhardware):
        super().__init__()
        self.plotupdatesec = 1
        self.samplingperiod = 4e-3  # In seconds
        self.wsize = 200000
        self.driver = driver
        self.timereads = np.zeros(self.wsize)
        self.flagparar = True
        self.flagrodando = False
        self.flagsaved = True
        self.ctreadings = 0
        self.globalctreadings = 0
        self.readtime = 0
        self.accdata = [[np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)],
                        [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)],
                        [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)]]
        self.gyrodata = [[np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)],
                        [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)],
                        [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)]]
        self.dacoutdata = [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)]
        self.adcdata = [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)]
        self.xrefdata = np.zeros(self.wsize)
        self.xerrodata = np.zeros(self.wsize)
        self.maxtime = int(self.wsize * self.samplingperiod)
        self.realtime = 0.0
        self.hasPaths = False
        self.ctrlmode = False
        self.taskisctrl = False
        self.lastdatafolder = None
        self.errorlevel = 0


    def ParaLeituras(self):
        self.flagparar = True
        self.statusMessage.emit("Stopping...")

    def IniciaLeituras(self,stoptime=3600.0):
        self.statusMessage.emit(None)      
        self.flagparar = False
        self.stoptime = stoptime
        self.mythread = Thread(target=self.LeDados)
        self.mythread.start()

    def LeDados(self):
        try:
            # trd = Thread(target=self.updateFigs)
            # trd.start()
            self.errorlevel = 0
            self.ctrlmode = self.driver.controlMode
            self.taskisctrl = self.driver.taskIsControl
            self.debugmode = self.driver.debugMode
            if self.debugmode:
                self.driver.debugSetup()
            self.flagrodando = True
            self.statusMessage.emit("Opening port...")
            self.driver.openSerial()
            self.statusMessage.emit("Handshaking...")
            self.driver.handshake()
            self.statusMessage.emit("Writing configs...")
            self.driver.writeSamplingMicros(int(self.samplingperiod*1e6))            
            for k in range(3):
                self.driver.writeIMUConfig(k)
                if self.driver.IMUEnableFlags[k]:
                    self.driver.initHardware(k)
            self.driver.writeADCConfig()   
            # for k in range(4):
            #     self.driver.writePredistConfig(id=k)         
            for k in range(4):
                self.driver.writeGeneratorConfig(id=k)
                time.sleep(0.1)
            if self.ctrlmode:  # If control on and control task is control (not path modelling)                
                self.driver.writeControlConfig()
                self.driver.setAlgOn(False, 0, forcewrite=True)                
                self.driver.startControl()
            else:
                self.driver.startReadings()
            self.statusMessage.emit(None)
            self.started.emit()
            self.flagsaved = False
            self.ctreadings = 0
            self.starttime = round(time.time() * 1000) / 1000
            lastfigrefresh = 0
            if self.globalctreadings == 0:
                self.globalstarttime = self.starttime
            timedelta = self.starttime - self.globalstarttime
            self.logMessage.emit(timedelta,"Started")            
            while (not self.flagparar) and (self.globalctreadings < self.wsize):
                if self.debugmode:
                    self.driver.debugTalk()
                self.driver.getReading()
                self.readtime = self.ctreadings * self.samplingperiod + timedelta
                self.timereads[self.globalctreadings] = self.readtime
                if self.ctrlmode:
                    self.xrefdata[self.globalctreadings] = self.driver.xref
                    self.xerrodata[self.globalctreadings] = self.driver.xerro
                    # if self.taskisctrl:
                    self.dacoutdata[0][self.globalctreadings] = self.driver.dacout[0]
                    self.dacoutdata[1][self.globalctreadings] = self.driver.dacout[1]
                    # else:   
                    #     self.dacoutdata[0][self.globalctreadings] = self.driver.dacout[self.driver.perturbChannel]
                    #     self.dacoutdata[1][self.globalctreadings] = self.driver.dacout[self.driver.controlChannel]
                else:
                    for j in range(3):
                        if self.driver.IMUEnableFlags[j]:
                            for k in range(3):
                                self.accdata[j][k][self.globalctreadings] = self.driver.accreadings[j][k]
                                self.gyrodata[j][k][self.globalctreadings] = self.driver.gyroreadings[j][k]
                    self.dacoutdata[0][self.globalctreadings] = self.driver.dacout[0]
                    self.dacoutdata[1][self.globalctreadings] = self.driver.dacout[1]
                    self.dacoutdata[2][self.globalctreadings] = self.driver.dacout[2]
                    self.dacoutdata[3][self.globalctreadings] = self.driver.dacout[3]
                for k in range(4):
                    # if self.driver.adcenablemap[k] != 0:
                    self.adcdata[k][self.globalctreadings] = self.driver.adcin[k]
                self.ctreadings += 1
                self.globalctreadings += 1
                if (self.readtime - lastfigrefresh) >= self.plotupdatesec:
                    # print(self.readtime)
                    self.realtime = self.readtime - (time.time() - self.starttime)
                    lastfigrefresh = self.readtime
                    if self.driver.algonchanged and (self.readtime >= self.driver.algontime):
                        self.driver.writeAlgOn()
                        self.logMessage.emit(self.readtime,"Alg")
                        # print("AlgOn!!!")
                    for k in range(4):
                        if not self.driver.genConfigWritten[k]:
                            self.driver.writeGeneratorConfig(k)
                            self.logMessage.emit(self.readtime,"Gen" + str(k))
                    # if not trd.isAlive():
                    #     trd = Thread(target=self.updateFigs)
                    #     trd.start()
                    # print(f'{int(self.driver.calctime)} us')
                    self.updateFigures.emit()
                    if self.readtime >= self.stoptime:
                        self.ParaLeituras()
                ctaux = 0            
            self.driver.stopReadings()
            self.stopped.emit()
            self.logMessage.emit(self.readtime,"Stopped")
            self.statusMessage.emit("Stopped.")            
            self.flagrodando = False
            self.flagparar = False
            self.readtime = self.ctreadings * self.samplingperiod + timedelta
            self.updateFigures.emit()
            # app.save(srcdir,"backup" + str(time.time()) + ".txt")
        except Exception as err:
            if self.driver.serial.isOpen():
                self.driver.stopReadings()
            self.flagrodando = False
            self.statusMessage.emit("Error: " + str(err))
            self.errorlevel = 1
            self.stopped.emit()


    def resetData(self, samplingperiod=4e-3, maxtime=10):
        """ Reset data with:
              - samplingperiod in seconds
              - maxtime in minutes
        """
        self.samplingperiod = samplingperiod
        self.maxtime = maxtime * 60
        self.wsize = int(self.maxtime/self.samplingperiod)
        self.flagsaved = True
        self.globalctreadings = 0
        self.timereads = np.zeros(self.wsize)
        self.accdata = [[np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)],
                        [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)],
                        [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)]]
        self.gyrodata = [[np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)],
                        [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)],
                        [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)]]
        self.dacoutdata = [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)]
        self.adcdata = [np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize), np.zeros(self.wsize)]
        self.xrefdata = np.zeros(self.wsize)
        self.xerrodata = np.zeros(self.wsize)
        self.reset.emit()        
        

    def salvaArquivo(self, filename, setsaved, loglist=None):
        if self.driver.controlMode:
            limf = self.globalctreadings
            df = pd.DataFrame({
                'time': self.timereads[0:limf],
                'perturb': self.dacoutdata[0][0:limf],
                'ctrl': self.dacoutdata[1][0:limf],
                'ref': self.xrefdata[0:limf],
                'err': self.xerrodata[0:limf]
            })
            
            if loglist:
                df = pd.concat([df,pd.DataFrame({"log":[" "]})],axis=1)
                tempo = df["time"].values
                idx = 0
                for item in loglist:
                    while tempo[idx] < item[0]:
                        idx += 1
                        pass
                    df.at[idx, "log"] = item[1]
                    idx += 1
            if filename.endswith("feather"):
                df.to_feather(filename)
            elif filename.endswith("csv"):
                df.to_csv(filename,sep="\t")
            self.statusMessage.emit(f"File {filename} saved successfully.")

        else:
            limf = self.globalctreadings
            thedict = {
                "time": self.timereads[0:limf],                
                "dac1": self.dacoutdata[0][0:limf],
                "dac2": self.dacoutdata[1][0:limf],
                "dac3": self.dacoutdata[2][0:limf],
                "dac4": self.dacoutdata[3][0:limf]                
            }
            if (self.driver.adcconfig[0] & 0x0F) > 0:
                for k in range(4):
                    thedict[f"adc{k+1}.{self.driver.adcsel+1}"] = self.adcdata[k][0:limf]
            for k in range(3):
                if self.driver.IMUEnableFlags[k]:
                    thedict[f"imu{k+1}accx"] = self.accdata[k][0][0:limf]
                    thedict[f"imu{k+1}accy"] = self.accdata[k][1][0:limf]
                    thedict[f"imu{k+1}accz"] = self.accdata[k][2][0:limf]
                    thedict[f"imu{k+1}gyrox"] = self.gyrodata[k][0][0:limf]
                    thedict[f"imu{k+1}gyroy"] = self.gyrodata[k][1][0:limf]
                    thedict[f"imu{k+1}gyroz"] = self.gyrodata[k][2][0:limf]
                    
            df = pd.DataFrame(thedict)

            # df.to_feather(filename + "2")
            
            # df.to_csv(filename)
            if loglist:
                # print(loglist)
                df = pd.concat([df,pd.DataFrame({"log":[" "]})],axis=1)
                tempo = df["time"].values
                idx = 0
                for item in loglist:
                    while tempo[idx] < item[0]:
                        idx += 1
                        pass
                    df.at[idx, "log"] = item[1]
                    idx += 1
                # for item in loglist:
                #     if item[0] == 0:
                #         zeroitens.append(item[1])
                #         idx += 1
                #     else:
                #         if not concatenated:
                            
                #             concatenated = True
                #         pos = df["Tempo (s)"][data["Tempo (s)"] >= item[0]].index[0]
                #         if item[1] == "Started":
                            
                #         df.loc[np.isclose(df["Tempo (s)"],item[0],atol=1e-3), "Log"] = item[1]
                #         print(item[1])
                #         # df["Log"][df["Tempo (s)"] == float(item[0])] = item[1]
                # # lista = [f"{item[0]}: {item[1]}" for item in loglist]

            if filename.endswith("feather"):
                df.to_feather(filename)
            elif filename.endswith("csv"):
                df.to_csv(filename,sep="\t")
            self.statusMessage.emit(f"File {filename} saved successfully.")
            

        self.flagsaved = setsaved
