"""
  Driver Hardware
"""
import time
import serial
import serial.tools.list_ports
import struct
from .CantileverBeam import CantileverBeam
import numpy as np

class driverhardware:

    def __init__(self):
        self.accscaler = [9.80665 * 2.0 / (2**15)] * 3
        self.gyroscaler = [250.0 / (2**15)] * 3
        self.accrangeselection = [0] * 3
        self.gyrorangeselection = [0] * 3
        self.filter = 0
        self.serial = None
        self.buf = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.accreadings = [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]  # x,y,z
        self.gyroreadings = [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]  # x,y,z
        self.dacout = [0.0, 0.0, 0.0, 0.0]
        self.gentipo = [0, 0, 0, 0]
        self.genamp = [0.0, 0.0, 0.0, 0.0]
        self.genfreq = [0.0, 0.0, 0.0, 0.0]
        self.dclevel = [1668,1668,104,104]
        self.iampscaler = [1/float(self.dclevel[0]),1/float(self.dclevel[1]),1/float(self.dclevel[2]),1/float(self.dclevel[3])]
        self.chirpconf = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        self.pwmduty = [0,0,0,0]
        self.imuconfigdata = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
        self.IMUEnableFlags = [False,False,False]
        self.IMUTypes = [False,False,False]
        self.genConfigWritten = [False, False, False, False]
        self.adcconfig = [0, 0, 0]
        self.adcenablemap = [False, False, False, False]
        self.adcin = [0.0,0.0,0.0,0.0]
        self.adcmultiplier = 0
        # self.adcseq = [0,0,0,0]
        self.adcsel = 0
        self.adcturbo = 4
        self.setGeneratorConfig(id=0)
        self.setGeneratorConfig(id=1)
        self.setGeneratorConfig(id=2)
        self.setGeneratorConfig(id=3)
        self.serial = serial.Serial(port=None,
                                    # baudrate=115200,
                                    baudrate = 500000,
                                    # baudrate = 1000000,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
                                    bytesize=serial.EIGHTBITS,
                                    timeout=1,
                                    rtscts=False)
        if self.serial.isOpen():
            self.serial.close()
        self.controlMode = False
        self.taskIsControl = 0  # True for Control, False for Path Modelling
        self.debugMode = False
        self.controlChannel = 0
        self.perturbChannel = 0
        self.ctrlalg = 0
        self.algon = False
        self.ctrlmem = 100
        self.ctrlmu = 0.1
        self.ctrlfi = 1e-4
        self.refid = 0
        self.erroid = 0
        self.ctrltask = 0
        self.refimuid = 0
        self.errimuid = 1
        self.packetsize = 17
        self.xref = 0
        self.xerro = 0
        self.algonchanged = False
        self.calctime = [0,0,0]
        self.algontime = 0.0 
        self.debugErr = 1.0
        self.debugRef = -1.0
        self.debugPerturb = 0.0
        self.debugControl = 0.0
        self.predistenablemap = [False,False,False,False]
        self.predistcoefs = [np.array([1.0,0.0]),np.array([1.0,0.0]),np.array([1.0,0.0]),np.array([1.0,0.0])]
        self.fusionweights = [0.5,0.5]

    def listPorts(self):
        ports = serial.tools.list_ports.comports()
        return ports

    def setPort(self,port):
        self.serial.port = port

    def setBaudRate(self,newbaudrate):
        self.serial.baudrate = newbaudrate

    def closeSerial(self):
        self.serial.close()

    def openSerial(self):        
        if not self.serial.isOpen():
            # self.serial.dtr = True
            self.serial.open()
            # Forcing device bootup:
            self.serial.rts = False
            self.serial.dtr = False
            time.sleep(0.1)
            self.serial.rts = True
            self.serial.dtr = True
            time.sleep(0.2)        
            aux = self.serial.read(2048)
            print(len(aux))
    
    def setControlMode(self,mode=False,task=0,debugmode=False):
        self.controlMode = mode
        if mode == False:  # if control is off
            self.taskIsControl = True
            self.debugMode = False
        else:
            self.debugMode = debugmode
            if task == 0:
                self.taskIsControl = True
            else:
                self.taskIsControl = False


    def setGeneratorConfig(self, id=0, tipo=0, amp=0.0, freq=10.0, dclevel=128, chirpconf=[0, 0, 0, 0, 0], pwmduty=0.0):
        # if tipo != 2:
        if (self.gentipo[id] != tipo) or (self.genamp[id] != amp) or (self.genfreq[id] != freq) or (self.dclevel[id] != dclevel):
            self.genConfigWritten[id] = False
        if tipo == 2:
            if chirpconf[4] > 1.0:
                chirpconf[4] = 1.0
            if id <= 2:  # MCP4725
                ampaux = int(chirpconf[4] * 2047)
            else:  # Saída ESP32
                ampaux = int(chirpconf[4] * 127)
            self.chirpconf[id] = [int(chirpconf[0]), int(10 * chirpconf[1]),
                                  int(chirpconf[2]), int(10 * chirpconf[3]), (ampaux >> 8) & 0xFF, ampaux & 0xFF]
            self.genConfigWritten[id] = False
        elif tipo == 4:
            dutyaux = int(np.round(pwmduty * 100 * 2))
            if self.pwmduty[id] != dutyaux:
                self.pwmduty[id] = dutyaux 
                self.genConfigWritten[id] = False
        self.gentipo[id] = tipo
        self.genamp[id] = amp
        self.genfreq[id] = freq
        self.dclevel[id] = dclevel
        self.iampscaler[id] = 0 if (dclevel == 0) else 1/float(self.dclevel[id])

    def setADCConfig(self, adcconfigs=[0, 0, 0]):
        self.adcconfig = adcconfigs
        multipliers = [6.144, 4.096, 2.048, 1.024, 0.512, 0.256]
        self.adcmultiplier = multipliers[adcconfigs[1]] / 2**15
        if (adcconfigs[0] >> 4) != 0:
            self.adcmultiplier = self.adcmultiplier * 16  # Correction for ADS1015
        self.adcenablemap = [ (self.adcconfig[0] & 0x01) == 1, 
                              ((self.adcconfig[0] >> 1) & 0x01) == 1,
                              ((self.adcconfig[0] >> 2) & 0x01) == 1,
                              ((self.adcconfig[0] >> 3) & 0x01) == 1]

    """ Seta range do acelerômetro:
        0 = -2 a +2 g
        1 = -4 a +4 g
        2 = -8 a +8 g
        3 = -16 a +16 g """
    def setAccRange(self, id, nrange):
        self.accrangeselection[id] = nrange
        self.accscaler[id] = 2.0**(nrange + 1) * 9.80665 / (2**15)


    """ Seta range do giroscópio:
        0 = -125 a +125 graus/s
        1 = -250 a +250 graus/s
        1 = -500 a +500 graus/s
        2 = -1000 a +1000 graus/s
        3 = -2000 a +2000 graus/s """
    def setGyroRange(self, id, nrange):
        self.gyrorangeselection[id] = nrange
        self.gyroscaler[id] = 125.0 * (2.0**nrange) / (2**15)


    def initHardware(self,id=0):
        ntries = 3
        while ntries > 0:
            aux = bytearray([ord('i')] + [id])
            self.serial.write(aux)
            self.serial.flush()
            time.sleep(0.1)
            aux = self.serial.read(3)
            # print(aux)
            if aux == b'ok!':
                return
            else:
                ntries -= 1
                self.serial.reset_output_buffer()
                self.serial.reset_input_buffer()
                time.sleep(0.1)
        raise Exception(f'Fail initializing the IMU with id={id}.')        


    def setIMUConfig(self,id: int, imucfgdata: list):
        self.imuconfigdata[id] = imucfgdata
        self.setAccRange(id, imucfgdata[2] & 0x03)
        self.setGyroRange(id, (imucfgdata[2]>>2) & 0x07)
        self.IMUEnableFlags[id] = ((imucfgdata[0] & 0x01) == 1)
        self.IMUTypes[id] = (imucfgdata[0]>>1) & 0x01 
        

    def writeIMUConfig(self,id: int):
        # print("writeIMU")
        aux = bytearray([ord('I')] + [id] + self.imuconfigdata[id])
        # print(aux)
        self.serial.write(aux)
        self.serial.flush()
        aux = self.serial.read(2)
        if aux != b'KI':
            print(aux)
            raise Exception(f'Error writing IMU{id+1} Config.')
        

    def writeSamplingMicros(self,TSamplingMicros: int):
        # print("writeIMU")
        print(TSamplingMicros)
        if TSamplingMicros in [1000,2000,3000,3003,4000]:
            TSampling = int(TSamplingMicros / 1000)
            aux = bytearray([ord('f')] + [TSampling])
            tries = 0
            while tries < 5:
                # print(aux)
                self.serial.write(aux)
                self.serial.flush() 
                aux = self.serial.read(2)
                if (aux[0] != ord('f')) or (aux[1] != TSampling):
                    print(aux)
                    tries += 1
                    self.serial.reset_output_buffer()
                    self.serial.reset_input_buffer()
                    # time.sleep(0.1)
                else: 
                    return True
            raise Exception(f'Error writing sampling rate.')
        else:
            aux = bytearray([ord('m')] + [(TSamplingMicros >> 8) & 0xFF,TSamplingMicros & 0xFF])
            tries = 0
            while tries < 5:
                # print(aux)
                self.serial.write(aux)
                self.serial.flush() 
                aux = self.serial.read(3)
                if (aux[0] != ord('m')) or (aux[1] != ((TSamplingMicros >> 8) & 0xFF)) or (aux[2] != (TSamplingMicros & 0xFF)):
                    print(aux)
                    tries += 1
                    self.serial.reset_output_buffer()
                    self.serial.reset_input_buffer()
                    # time.sleep(0.1)
                else: 
                    return True
            raise Exception(f'Error writing sampling rate (micros).')
        
    
    
    def writeSampling(self,TSampling: int):
        # print("writeIMU")
        # print(TSampling)
        aux = bytearray([ord('f')] + [TSampling])
        tries = 0
        while tries < 5:
            # print(aux)
            self.serial.write(aux)
            self.serial.flush() 
            aux = self.serial.read(2)
            if (aux[0] != ord('f')) or (aux[1] != TSampling):
                print(aux)
                tries += 1
                self.serial.reset_output_buffer()
                self.serial.reset_input_buffer()
                # time.sleep(0.1)
            else: 
                return True
        raise Exception(f'Error writing sampling rate.')

    
    def writePredistConfig(self, id=0):
        self.serial.reset_output_buffer()
        self.serial.reset_input_buffer()
        order = (self.predistcoefs[id].shape[0] - 1) if self.predistenablemap[id] else 0
        aux = 'p'.encode() + bytes([id, order])
        if order > 0:
            for k in range(order+1):
                aux = aux + struct.pack("f",self.predistcoefs[id][k])
        tries = 0
        while tries < 5:
            self.serial.write(aux)   
            self.serial.flush()   
            auxr = self.serial.read(2)
            if auxr != b'ok':
                print(f"id={id}: {auxr}")
                tries += 1
                time.sleep(0.1)
                self.serial.reset_output_buffer()
                self.serial.reset_input_buffer()                
            else:
                if order > 0:
                    auxr = self.serial.read((order+1)*4)
                    if auxr != aux[3:]:
                        raise Exception(f'Byte check failed for predistorter {id}. Please try again.')
                return
        raise Exception(f'Error configuring predistorter {id}.')

    def writeFusionConfig(self):
        self.serial.reset_output_buffer()
        self.serial.reset_input_buffer()
        aux = 'F'.encode()
        for k in range(2):
            aux = aux + struct.pack("f",self.fusionweights[k])
        tries = 0
        while tries < 5:
            self.serial.write(aux)   
            self.serial.flush()   
            auxr = self.serial.read(2)
            if auxr != b'ok':
                print(f"{auxr}")
                tries += 1
                time.sleep(0.1)
                self.serial.reset_output_buffer()
                self.serial.reset_input_buffer()                
            else:
                auxr = self.serial.read(8)
                if auxr != aux[1:]:
                    raise Exception(f'Byte check failed for sensor fusion weights. Please try again.')
                return
        raise Exception(f'Error configuring fusion coefficients.')

    def recordAdditionalConfigs(self):
        aux = 'w'.encode()
        self.serial.write(aux)
        auxr = self.serial.read(3)
        if auxr != b'ok!':
            raise Exception("Failed record predist data in device memory.")


    def writeGeneratorConfig(self, id=0):        
        freqaux = [int(self.genfreq[id]), int(round((self.genfreq[id] - int(self.genfreq[id])) * 100))]
        ampaux = int(round(self.genamp[id] * 2047))
        dcl = self.dclevel[id]  
        aux = 'G'
        aux = aux.encode() + bytes([id, self.gentipo[id], (ampaux >> 8) & 0xFF, ampaux & 0xFF] + freqaux + [dcl >> 8, dcl & 0xFF])
        if self.gentipo[id] == 2:  # Chirp, manda mais 4 bytes [tinicio,deltai,tfim,deltaf]
            aux = aux + bytes(self.chirpconf[id])
        if self.gentipo[id] == 4:
            aux = aux + bytes([self.pwmduty[id]])
        self.serial.write(aux)
        self.genConfigWritten[id] = True
        # aux = self.serial.read(2);
        # if aux != b'ok':
        #     print(aux)
        #     raise Exception('Config. de Gerador sem resposta.')

    def writeADCConfig(self):
        aux = 'd'.encode() + bytes(self.adcconfig)
        self.serial.write(aux)
        aux = self.serial.read(2)
        if aux != b'KA':
            # print(aux)
            raise Exception('Erro na configuração do ADC.')
        else:
            # self.adcseq = [aa[0] for aa in struct.iter_unpack("b",self.serial.read(4))] 
            auxx = [aa[0] for aa in struct.iter_unpack("b",self.serial.read(4))] 
            self.adcsel = auxx[0]
            if ((self.adcconfig[0] & 0x0F) == 0):
                # self.adcseq = [0,0,0,0]
                self.adcsel = 0
    
    def writeBaudRate(self,newbaud):
        baudcode = 0 if newbaud == 115200 else (2 if newbaud == 921600 else (3 if newbaud == 1000000 else 1)) 
        cmd = 'b'.encode() + bytes([baudcode])
        self.serial.write(cmd)
        aux = self.serial.read(2)
        if aux != ('k'.encode() + bytes([baudcode])):
            raise Exception('Error setting a new baud rate.')        

    def handshake(self):
        if not self.serial.is_open:
            self.openSerial()
        try:
            self.serial.reset_output_buffer()
            self.serial.reset_input_buffer()
        except BaseException:
            self.closeSerial()
            self.openSerial()
        # Try to perform handshake twice:
        for k in range(2):            
            self.serial.write(b'h')
            if self.serial.read(1) == b'k':
                # print("Clean handshake.")
                return True  # In case of success, return.
            self.serial.reset_output_buffer()
            self.serial.reset_input_buffer()
            time.sleep(0.1)
        # If handshake fails, try to adjust the baud rate:
        newbaud = self.serial.baudrate
        baudrates = [500000,921600,1000000,115200] 
        for bd in baudrates:
            if bd == newbaud:
                continue
            self.serial.baudrate = bd
            time.sleep(0.05)
            self.serial.write(b'h')
            if self.serial.read(1) == b'k':
                print(f"Found baud rate: {bd}, adjusting to {newbaud}.")
                self.serial.reset_output_buffer()
                self.serial.reset_input_buffer()
                self.writeBaudRate(newbaud)
                self.serial.baudrate = newbaud
                time.sleep(0.15)
                for k in range(2):
                    time.sleep(0.1)                
                    self.serial.reset_output_buffer()
                    self.serial.reset_input_buffer()
                    self.serial.write(b'h')
                    if self.serial.read(1) == b'k':
                        print("Baud adjusted and handshake ok.")
                        return True
                    break
        self.serial.baudrate = newbaud
        self.serial.close()
        raise Exception("Handshake com dispositivo falhou.")
    

    def setControlConfig(self, alg=0, mem=0, mu=0, fi=0, refimuid=0, errimuid=0, refid=0, erroid=0, ctrltask=0):
        self.ctrlalg = alg
        self.ctrlmem = mem
        self.ctrlmu = mu
        self.ctrlfi = fi
        self.refimuid = refimuid
        self.refid = refid
        self.errimuid = errimuid
        self.erroid = erroid
        self.ctrltask = ctrltask

    def setDebugMode(self,debugmode):
        self.debugMode = debugmode

    def writeControlConfig(self):
        # Data to send: 
        #   Byte 0: 4 bits for perturbation channel (MSBs) and 4 bits for control channel (LSBs)
        #   Byte 1: 4 bits for REF imu id (MSBs) and 4 bits for sensor choice (AccX = 0 to GyroZ = 5)
        #   Byte 2: 4 bits for ERROR imu id (MSBs) and 4 bits for sensor choice (AccX = 0 to GyroZ = 5)
        #   Byte 3: Algorithm choice
        #   Bytes 4 and 5: Memory size (from 0 to 65535)
        #   Bytes 6 to 9: Step size value (float encoded in 4 bytes)
        #   Bytes 10 to 13: Regularization factor (float enconded in 4 bytes) 
        #   Byte 14: Bit 0 for Control task (control or path modeling) and bit 1 for debugmode
        self.serial.write(b'!')
        buf = [0] * 6
        buf[0] = (self.perturbChannel << 4) + self.controlChannel
        buf[1] = (self.refimuid << 4) + self.refid
        buf[2] = (self.errimuid << 4) + self.erroid
        buf[3] = self.ctrlalg
        buf[4] = (self.ctrlmem >> 8) & 0xFF
        buf[5] = self.ctrlmem & 0xFF
        self.serial.write(bytearray(buf[0:6]))
        self.serial.write(bytearray(struct.pack("f", self.ctrlmu)))
        self.serial.write(bytearray(struct.pack("f", self.ctrlfi)))
        self.serial.write((self.ctrltask + (2 if self.debugMode else 0) ).to_bytes(1,'big'))
        if self.serial.read(3) != b'ok!':
            raise Exception("Fail recording control configurations.")

    def setAlgOn(self, status=False, algontime=0.0, forcewrite=False):
        if (status != self.algon) or forcewrite:
            self.algontime = algontime
            self.algon = status
            self.algonchanged = True

    def writeAlgOn(self):
        if self.algon:
            # elf.serial.write(b'a\x01')
            self.serial.write(b'a\x02')
            self.serial.write(bytearray(struct.pack("f", self.ctrlmu)))
            self.serial.write(bytearray(struct.pack("f", self.ctrlfi)))
            # print(self.serial.readline())
            # print(self.serial.readline())
        else:
            self.serial.write(b'a\x00')
        self.algonchanged = False

    def startReadings(self):
        self.readsize = 0
        for k in range(3):
            if self.IMUEnableFlags[k]:
                self.readsize += 14 if (self.IMUTypes[k] == 0) else 12 
        nADCs = ((self.adcconfig[0] >> 3) & 0x01) + ((self.adcconfig[0] >> 2) & 0x01) + ((self.adcconfig[0] >> 1) & 0x01) + (self.adcconfig[0] & 0x01)
        # self.readsize += 6 + 2*nADCs + 2
        if nADCs == 0:
            self.readsize += 6 + 0 + 2 + 2 + 2 + 1
        else:
            self.readsize += 6 + 2*4 + 2 + 2 + 2 + 1
        self.serial.write(b's')
        if (len(self.serial.read(10)) < 10):
            raise Exception('Sem resposta nas leituras.')
        # if self.serial.read != b'k':
        #     raise Exception('Failed starting readings.')
        self.buf = [0] * self.readsize

    def startControl(self):
        self.packetsize = 14
        self.serial.write(b'S')
        if self.serial.read() != b'K':
            # if (len(self.serial.read(14)) < 14):
            raise Exception('Sem resposta nas leituras.')
        self.buf = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def stopReadings(self):
        self.serial.write(b't')
        while True:
            aux = self.serial.read(1000)
            if aux and len(aux) > 0:
                pass
                # print(aux)
                # print(len(aux))
            else:
                break
            time.sleep(0.05)
        # self.serial.flush()
        # self.serial.rts = True
        # self.serial.close()

    def gravaFlash(self):
        self.serial.write(b'P')
        time.sleep(1.0)
        if self.serial.read(2) != b'ok':
            raise Exception("Sem resposta na gravação da flash.")

    def gravaCaminho(self, tipo, dados, pbar):
        BUFFER_SIZE = 64
        step = BUFFER_SIZE / 4
        pbar.setMaximum(dados.shape[0] - 2)
        pbar.setValue(0)
        npacotes = (dados.shape[0] * 4) // BUFFER_SIZE
        if ((dados.shape[0] * 4) % BUFFER_SIZE) > 0:
            pacoteextra = True
        else:
            pacoteextra = False
        self.serial.write(('W' + tipo).encode())
        self.serial.write(bytes([(dados.shape[0] * 4) >> 8, (dados.shape[0] * 4) & 0xFF]))
        if self.serial.read(1) == b'k':
            for k in range(npacotes):
                for w in dados[k * BUFFER_SIZE:k * BUFFER_SIZE + BUFFER_SIZE]:
                    self.serial.write(bytearray(struct.pack("f", w)))
                bb = self.serial.read(2)
                pbar.setValue(pbar.value() + (bb[0] << 8) + bb[1])
            if pacoteextra:
                for w in dados[npacotes * BUFFER_SIZE:]:
                    self.serial.write(bytearray(struct.pack("f", w)))
                buf = self.serial.read(2)
                pbar.setValue(pbar.value() + (bb[0] << 8) + bb[1])
        pbar.setValue(dados.shape[0] - 2)


    def readPaths(self):
        wsectemp = []
        wfbktemp = []
        self.serial.write('c'.encode())
        aux = self.serial.readline().decode("utf-8")
        if not aux.startswith("WSec ="):
            print(aux)
            raise BaseException("Error reading paths (at the beginning).")
        aux = self.serial.readline().decode("utf-8")
        while aux and (not aux.startswith("WFbk =")):
            wsectemp.append(float(aux))
            aux = self.serial.readline().decode("utf-8")
        aux = self.serial.readline().decode("utf-8")
        while aux and (not aux.startswith("End")):
            wfbktemp.append(float(aux))
            aux = self.serial.readline().decode("utf-8")
        return wsectemp,wfbktemp

        
    def debugSetup(self):
        Tamostragem = 4e-3
        dadosviga = {'length':0.538,'width':0.0507,'thickness':0.00474-0.0001*2,'damp':[0.010826771653543302,0.0076347771891554575,0,0,0],
             'density':7850,'elasticmod':200e9}
        self.cbeam = CantileverBeam(Tsampling=Tamostragem,nmodes=2,**dadosviga)
        self.cbeam.configforcescaler(25.0/12.0,3e-3)
        # Definições:
        self.pperturb = round((0.34 / self.cbeam.length) * self.cbeam.npoints)
        self.pref = round((0.29 / self.cbeam.length) * self.cbeam.npoints)
        self.pcanc = round((0.20 / self.cbeam.length) * self.cbeam.npoints)
        self.perro = round((0.53 / self.cbeam.length) * self.cbeam.npoints)
        self.cbeam.noisestd = 0.0045 # Sempre em m/s^2
        self.cbeam.setaccelg(False)
        self.cbeam.reset()
        self.debugControl = 0.0
        self.debugPerturb = 0.0

        
    def debugTalk(self):       
        # self.cbeam.setforce(self.pperturb, self.debugPerturb)
        # self.cbeam.setforce(self.pcanc, self.debugControl)        
        self.debugErr = self.cbeam.getaccel(self.perro)
        self.debugRef = self.cbeam.getaccel(self.pref)
        # print(self.debugRef)
        transmitok = False
        while not transmitok:
            transmitok = True
            self.serial.write(b'T')
            mybytes = struct.pack("f", self.debugErr) + struct.pack("f", self.debugRef)
            self.serial.write(mybytes)
            buf = self.serial.read(8)
            if len(buf) < 8:
                transmitok = False
                print("!!!")
                self.serial.reset_output_buffer()
                self.serial.reset_input_buffer()                
            else:
                if mybytes != buf[0:8]:
                    print(mybytes)
                    print(buf[0:8])
                    transmitok = False
                    print("!!")
                    self.serial.reset_output_buffer()
                    self.serial.reset_input_buffer()
        self.serial.write(b'N')
        buf = self.serial.read(4)
        if (len(buf) == 4):
            self.debugPerturb = -float(struct.unpack_from(">h",buf,0)[0] - self.dclevel[self.perturbChannel]) * self.iampscaler[self.perturbChannel]
            self.debugControl = -float(struct.unpack_from(">h",buf,2)[0] - self.dclevel[self.controlChannel]) * self.iampscaler[self.controlChannel]
            # print(self.debugPerturb)
            # print(self.debugControl)
        else:
            self.debugPerturb = 0
            self.debugControl = 0
            print("!")
        self.cbeam.setforce(self.pperturb, self.debugPerturb)
        self.cbeam.setforce(self.pcanc, self.debugControl)
        self.cbeam.update()  
        
    def getReading(self):
        ctaux = 0
        val = 0
        ptr = 0
        self.buf[0] = 0        
        while not((self.buf[0] == 0xF) and (self.buf[1] == 0xF) and (self.buf[2] == 0xF)) and (ctaux < 64):
            self.buf[2] = self.buf[1]
            self.buf[1] = self.buf[0]
            self.buf[0] = (self.serial.read()[0])
            ctaux = ctaux + 1
        if ctaux == 64:
            raise Exception('Falha na leitura de pacote: cabeçalho não encontrado.')
        # if ctaux > 3:
        #     print(f"wtf:{ctaux}!")
        if self.controlMode: # and self.taskIsControl:
            buf = self.serial.read(20)            
            self.dacout[0] = float(struct.unpack_from(">h",buf,0)[0]) * self.iampscaler[self.perturbChannel]          
            self.dacout[1] = float(struct.unpack_from(">h",buf,2)[0]) * self.iampscaler[self.controlChannel]
            # if self.debugPerturb != self.dacout[0]:
            #     print(f"Perturb: {self.debugPerturb} - {self.dacout[0]}")   
            # if self.debugControl != self.dacout[1]:
            #     print(f"Ctrl: {self.debugControl} - {self.dacout[1]}") 
            self.xref = struct.unpack_from("f",buf,4)[0]
            self.xerro = struct.unpack_from("f",buf,8)[0]
            # print(self.xref)
            self.calctime[0] = (struct.unpack_from(">H",buf,13)[0]) # << 4) / 240 # (((self.buf[13] << 8) + self.buf[14]) << 4) / 240
            self.calctime[1] = (struct.unpack_from(">H",buf,15)[0]) # << 4) / 240
            self.calctime[2] = (struct.unpack_from(">H",buf,17)[0]) # << 4) / 240
            self.errorflag = struct.unpack_from("B",buf,19)[0]
            # if self.errorflag != 0:
            #     print(self.errorflag)  # TODO: generate error message in UI.
        else:
            buf = self.serial.read(self.readsize)
            for j in range(3):
                if self.IMUEnableFlags[j]:
                    if self.IMUTypes[j] == 0:
                        for k in range(3):
                            self.accreadings[j][k] = float(struct.unpack_from(">h",buf,2*k+ptr)[0]) * self.accscaler[j]
                        for k in range(3): 
                            self.gyroreadings[j][k] = float(struct.unpack_from(">h",buf,2*k+8+ptr)[0]) * self.gyroscaler[j]
                        ptr += 14
                    else:
                        for k in range(3):
                            self.gyroreadings[j][k] = float(struct.unpack_from("<h",buf,2*k+ptr)[0]) * self.gyroscaler[j]
                        for k in range(3): 
                            self.accreadings[j][k] = float(struct.unpack_from("<h",buf,2*k+6+ptr)[0]) * self.accscaler[j]
                        ptr += 12  
            self.dacout[0] = float(struct.unpack_from(">h",buf,ptr)[0]) * self.iampscaler[0]
            self.dacout[1] = float(struct.unpack_from(">h",buf,ptr+2)[0]) * self.iampscaler[1]
            self.dacout[2] = float(struct.unpack_from(">b",buf,ptr+4)[0]) * self.iampscaler[2]
            self.dacout[3] = float(struct.unpack_from(">b",buf,ptr+5)[0]) * self.iampscaler[3]
            ptr += 6
            if (self.adcconfig[0] & 0x0F) > 0:
                for k in range(4):
                    self.adcin[k] = float( struct.unpack_from(">h",buf,ptr)[0] ) * self.adcmultiplier
                    ptr += 2
            self.calctime[0] = (struct.unpack_from(">H",buf,ptr)[0]) #<< 4) #/ 240
            self.calctime[1] = (struct.unpack_from(">H",buf,ptr+2)[0]) # << 4) #/ 240
            self.calctime[2] = (struct.unpack_from(">H",buf,ptr+4)[0]) #<< 4) #/ 240
            self.errorflag = struct.unpack_from("B",buf,ptr+6)[0]
            # if self.errorflag != 0:
            #     print(self.errorflag)  # TODO: generate error message in UI.
            # print(self.calctime)
