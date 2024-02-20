
from threading import Thread,Event
import time
import pandas as pd
from PySide2 import QtCore,QtGui,QtWidgets
from PySide2.QtWidgets import QDialog
from .AutomatorDialog import Ui_AutomatorDialog as AutomatorDialog

class Automator (QtCore.QObject):

    """
        Action List:
         - ConfigOutput: [Number (from 1 to 4), Enable, Type (Noise, Harmonic, ...), Ampl., Freq.]
         - ConfigIMU: [Number (from 1 to 3), Type (Disabled, MPU6050, LSM6DS3), Addr., Bus (I2C-1, I2C-2, SPI), Accr (0 to 3), GyroR (0 to 4), Filter1 (0 to 5), Filter2 (0 to 3)]
         - SetReadingMode: []
         - ControlChannels: [Perturbation, Control, RefIMU, RefSensor, ErrIMU, ErrSensor]
         - SetControlMode: [Algorithm (by index), AlgOnTime, MemSize, StepSize, Penalization]
         - SetPathModeling: []
         - AlgOn: [0 for False 1 for True]
         - Start: [stoptime]
         - Reset: []
    """
    COMMANDS = ["ConfigOutput","SetReadingMode","ControlChannels","SetControlMode","SetPathModeling","AlgOn","Start","SaveFile","StartWait","ConfigIMU"]

    actionMessage = QtCore.Signal((str,list))
    logMessage = QtCore.Signal((int,str))

    def __init__(self,parentapp):
        super().__init__()   
        self.parentapp = parentapp           
        self.seqthread = None
        self.pauseevent = Event()
        self.starttime = 0
        self.adialog = QDialog()
        self.adialog.ui = AutomatorDialog()
        self.pasteshortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+V"), self.adialog, self.catchPaste)
        self.adialog.ui.setupUi(self.adialog)  
        self.adialog.ui.bStop.clicked.connect(self.stop)
        self.adialog.ui.bStop.setDisabled(True)
        self.adialog.ui.bStart.clicked.connect(self.runSeq)
        self.adialog.ui.bLoad.clicked.connect(self.load)
        self.adialog.ui.bClear.clicked.connect(self.clearMessages)
        self.flagstop = False
        self.running = False
        self.pmodel = None
        self.errorcount = 0
        self.flagretry = False
        self.lista = [
            ["Reset",[]],
            ["ConfigOutput",[1,False]],
            ["ConfigOutput",[2,True,"Noise",0.75,10.0]],
            ["SetPathModeling",[]],
            ["Start",[10.0]],
            ["Waiting",[]],
            ["Reset",[]],
            ["ConfigOutput",[1,True,"Harmonic",0.25,38.0]],
            ["ConfigOutput",[2,False]],
            ["SetControlMode",[2,5,250,0.001,1e-2]],
            ["Start",[5.0]],
            ["Delay",[2.0]],
            ["AlgOn",[True]],
            ["Waiting",[]],
            ["Print",["Terminou!"]]
        ]

    def catchPaste(self):
        text = self.parentapp.clipboard().text()
        cmdlist = []
        for rr in text.split("\n"):
            aux = rr.split("\t")
            if len(aux) != 2:
                self.printErrorMessage("Error parsing command list!")
                return
            aux2 = [a.strip() for a in aux[1].split(",")]
            cmdlist.append([aux[0],aux2])
        self.lista = cmdlist
        self.load()

    def clearMessages(self):
        self.adialog.ui.messageArea.clear()

    def load(self):
        data = pd.DataFrame(self.lista,columns=["Command","Parameters"])
        self.pmodel = PandasModel(data)
        self.adialog.ui.tableView.setModel(self.pmodel)        
        self.adialog.ui.tableView.horizontalHeader().setStretchLastSection(True)
        self.adialog.ui.tableView.resizeRowsToContents()
        # self.adialog.ui.cmdTable.clear()
        # for ll in self.lista:
        #     # self.adialog.ui.sequenceArea.insertPlainText(f"{ll[0]} - {ll[1]}\n")
        #     self.adialog.ui.cmdTable.insert

    def stop(self):
        self.flagstop = True
        if self.seqthread:
            self.pauseevent.set()
        self.actionMessage.emit("Stopping",[])
        self.adialog.ui.bStop.setDisabled(True)
    
    def processerror(self,errorcode):
        print(f"Error in Automator task: {errorcode}.")
        self.errorcount += 1
        if self.errorcount >= 3:
            self.stop()
        else:
            self.flagretry = True
            print(self.errorcount)
            if self.seqthread:
                self.pauseevent.set()

    def showAutomatorDialog(self):             
        # self.adialog.exec_()
        self.adialog.show()

    def printMessage(self,msg):
        self.adialog.ui.messageArea.insertPlainText(msg)
        self.adialog.ui.messageArea.moveCursor(QtGui.QTextCursor.End)

    def printErrorMessage(self,msg):
        self.adialog.ui.messageArea.insertHtml(f'<span style="color: red; font-weight: bold;">{msg}</span><br>')
        self.adialog.ui.messageArea.moveCursor(QtGui.QTextCursor.End)

    def elapsedTime(self):
        return (time.time() - self.starttime)

    def runSeq(self):
        self.errorcount = 0
        while self.pauseevent.is_set():
            self.pauseevent.clear()
        self.seqthread = Thread(target=self.Seq,args=(self.pauseevent,0))
        self.starttime = time.time()
        self.flagstop = False
        self.adialog.ui.bStop.setDisabled(False)
        self.adialog.ui.bStart.setDisabled(True)
        self.seqthread.start()

    def resume(self):
        if self.seqthread:
            self.pauseevent.set()

    def Seq(self,pevent,val):
        self.running = True 
        n = 0     
        # for n,ll in enumerate(self.lista):
        while n < len(self.lista):
            self.flagretry = False
            if self.lista[n][0] == "Delay":
                self.actionMessage.emit(self.lista[n][0],self.lista[n][1])
                remtime = int(self.lista[n][1][0])
                delayend = time.time() + remtime
                while (remtime > 0):
                    time.sleep(5 if (remtime > 5) else remtime)
                    remtime = delayend - time.time()
                    if self.flagstop:
                        break
            else:
                self.actionMessage.emit(self.lista[n][0],self.lista[n][1])
                pevent.wait()
                pevent.clear()             
            if self.flagstop:                    
                self.flagstop = False
                break
            if self.pmodel:
                self.pmodel.actualrow = n
            if not self.flagretry:
                self.errorcount = 0
                n += 1
        self.adialog.ui.bStart.setDisabled(False)
        self.adialog.ui.bStop.setDisabled(True)
        self.actionMessage.emit("Finished!",[''])
        self.running = False


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
        self.actualrow = 0

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row()][index.column()])
            if role == QtCore.Qt.BackgroundRole:
                if index.row() < self.actualrow:
                    return QtGui.QColor(240,240,240)
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None

    # def flags(self, index):
    #     return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable