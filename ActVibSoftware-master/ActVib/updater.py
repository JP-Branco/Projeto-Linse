from pathlib import Path
import urllib.request 
import zipfile
import subprocess
import shlex
import shutil
from threading import Thread

from PySide2 import QtCore
from PySide2.QtWidgets import QDialog

from .UpdaterDialog import Ui_UpdaterDialog as UpdaterDialog

class Updater(QtCore.QObject):

    actionMessage = QtCore.Signal((str,bool)) # message,isHtml

    def __init__(self):
        super().__init__()
        self.actvibpath = Path(__file__).parent.parent
        self.seqthread = None
        self.flagrunning = False
        self.swzipurl = "https://github.com/eduardobatista/ActVibSoftware/archive/refs/heads/master.zip"
        self.fwzipurl = "https://github.com/eduardobatista/ActVibFirmware/archive/refs/heads/beta.zip" 
        self.udialog = QDialog()
        self.udialog.ui = UpdaterDialog()
        self.udialog.ui.setupUi(self.udialog)
        self.udialog.ui.startFirmware.clicked.connect(self.startFWUpdate)       
        self.udialog.ui.startSoftware.clicked.connect(self.startSWUpdate)        
    
    def showUpdaterDialog(self,port):
        self.port = port
        self.udialog.ui.messageArea.clear()                 
        self.udialog.exec_()

    def printMessage(self,msg,isHtml):
        if isHtml:
            self.udialog.ui.messageArea.insertHtml(msg)
        else:
            self.udialog.ui.messageArea.insertPlainText(msg)
        self.udialog.ui.messageArea.ensureCursorVisible()

    def startFWUpdate(self):        
        self.udialog.ui.messageArea.clear()
        self.seqthread = Thread(target=self.runUpdate)
        self.seqthread.start()
        self.flagrunning = True

    def startSWUpdate(self):
        self.udialog.ui.messageArea.clear()
        self.seqthread = Thread(target=self.runUpdate,args=[True])
        self.seqthread.start()
        self.flagrunning = True

    def runUpdate(self,isSoftware=False):        
        try:
            tmpdir = Path.home() / ".actvibtemp"
            tmpdir.mkdir(parents=True,exist_ok=True)
            extractedfiles = []
            self.actionMessage.emit("Tempdir created.<br>Downloading file...<br>",True)
            dfile = tmpdir / "downloadedfile.zip"
            urllib.request.urlretrieve(self.swzipurl if isSoftware else self.fwzipurl, dfile)
            self.actionMessage.emit("File downloaded.<br>Extracting...<br>",True)
            extractedfiles.append(dfile)
            if isSoftware:
                with zipfile.ZipFile(dfile,"r") as zipref:
                    for ff in zipref.namelist():
                        zipref.extract(ff,tmpdir)
                        extractedfiles.append(tmpdir / ff)
                if (self.actvibpath / ".git").exists():
                    self.actionMessage.emit('<br><span style="color: red;"><strong>Automatic software update not allowed in developer mode!</strong></span><br>',True)
                else:
                    self.actionMessage.emit('<br><span style="color: blue;"><strong>Updating files...</strong></span><br>',True)
                    for ff in extractedfiles:
                        if not ff.is_dir():
                            if ff.parent.name == "ActVib":
                                shutil.copy(ff,self.actvibpath / "ActVib")
                            elif ff.parent.name == "ActVibSoftware-master":
                                shutil.copy(ff,self.actvibpath)
                    self.actionMessage.emit('<br><span style="color: blue;"><strong>Update successful.</strong></span><br>',True)
                    self.actionMessage.emit('<br><span style="color: red;"><strong>RESTART SOFTWARE FOR FINISHING THE UPDATE.</strong></span><br>',True)
            else: 
                flashcmd = None
                with zipfile.ZipFile(dfile,"r") as zipref:
                    for ff in zipref.namelist():
                        if "update" in ff:
                            zipref.extract(ff,tmpdir)
                            if "flashcommand.sh" in ff:
                                flashcmd = tmpdir / ff
                            extractedfiles.append(tmpdir / ff)
                self.actionMessage.emit("Files extracted.<br>Flashing...<br>",True)
                if flashcmd:
                    with open(flashcmd,"r") as ff:
                        fcmd = shlex.split(ff.read().replace("<SERIALPORT>",self.port))
                        self.actionMessage.emit(f'<br><strong>Running command:</strong> {(fcmd)} <br>',True)
                        self.actionMessage.emit(f'<br><span style="color: red;"><strong>WARNING: If connection fails, press and hold the "BOOT" button at the ESP32 board when trying to connect.</strong></span><br><br>',True)
                        try:
                            proc =  subprocess.Popen([fcmd[0],"--version"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=flashcmd.parent)
                            if len(proc.stderr.readline()) > 0:
                                self.actionMessage.emit("Python3 not found, attempting to use python...<br><br>",True)
                                fcmd[0] = "python"
                        except BaseException as ex:
                            fcmd[0] = "python"
                            self.actionMessage.emit("Python3 not found, attempting to use python....<br><br>",True)
                        proc =  subprocess.Popen(fcmd,stdout=subprocess.PIPE,cwd=flashcmd.parent)
                        while True:
                            # line = proc.stdout.readline().decode()
                            line = proc.stdout.read(1).decode()
                            self.actionMessage.emit(line,False)
                            if not line:
                                break
            self.actionMessage.emit("<br>Cleaning up...<br>",True)
            # TODO: Cleaning up needs to be improved.
            for fff in extractedfiles:
                if not fff.is_dir():
                    fff.unlink()
            for ddir in tmpdir.glob("*"):
                if ddir.is_dir():
                    for ddir2 in ddir.glob("*"):
                        if ddir2.is_dir():
                            ddir2.rmdir()
                    ddir.rmdir()
            tmpdir.rmdir()
            self.actionMessage.emit("<strong>Finished!</strong>",True)
        except BaseException as ex:
            self.actionMessage.emit(f'<br><br><span style="color: red;"><strong>Error!</strong></span><br>{str(ex)}',True)
        self.flagrunning = False    
