# %%
# Import Libraries
import contextlib
import queue
import sys
import tempfile
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog, QWidget, QApplication, QMainWindow, QVBoxLayout, QLabel, QProgressBar, QComboBox, QPushButton
from PyQt5.QtCore import Qt, QSize, QRect, pyqtSlot, QThread
from PyQt5.QtGui import QPalette, QColor


class Recording:

    stream = None

    def __init__(self): 


        self.recording = self.previously_recording = False
        self.audio_q = queue.Queue()
        
        self.create_stream()

    def create_stream(self, device = None):
        if self.stream is not None:
            self.stream.close()
        
        self.stream = sd.InputStream(
            samplerate=48000, device=(0,1), channels=2, callback= self.audio_callback)
        self.stream.start()

    def file_writing_thread(self,*, q, **soundfile_args):
        """Write data from queue to file until *None* is received."""
        
        with sf.SoundFile(**soundfile_args) as file:
            while True:
                data = q.get()
                
                if data is None:
                    break
                file.write(data)

    def audio_callback(self, indata, frames, time, status):

        """This is called (from a separate thread) for each audio block."""
        

        if self.recording == True:
            self.audio_q.put(indata.copy())
            self.previously_recording = True

        else:
            if self.previously_recording:
                self.audio_q.put(None)
                self.previously_recording = False


    def on_rec(self):

        self.recording = True

        filename = tempfile.mktemp(prefix='test_recording_', suffix='.wav', dir ='')

        if self.audio_q.qsize() != 0:
            print('WARNING:QUEUE NOT EMPTY')

        
        self.thread = threading.Thread(
            target=self.file_writing_thread,
            kwargs=dict(
                file=filename,
                mode='x',
                samplerate=int(self.stream.samplerate),
                channels=self.stream.channels,
                q=self.audio_q,
            ),
        )
        self.thread.start()

        print('Recording')



    def on_stop(self):

        self.recording = False
        self.thread.join()
        print('Stopped')



rec = Recording()
# Define Signal / Slots
""" def play_button_clicked():
    import play
    play = play.AudioPlayback(recording, sr) """

@pyqtSlot()
def rec_button_on_click():
    #import record2
    
    rec.on_rec()
 
@pyqtSlot()   
def stop_button_on_click():
    #import record2
    
    rec.on_stop()

@pyqtSlot()
def settings_button_on_click():
    d = settings()

@pyqtSlot()
def on_host_changed(host):
    print("Host API changed:", host)

@pyqtSlot()
def on_device_changed(device):
    print("Sound device changed:", device)

# Access Sound Devices
class settings(QDialog):
    """Dialog window for choosing sound device."""
    def __init__(self):
        QDialog.__init__(self)
        
        self.setWindowTitle('Settings')
        # self.b1 = QPushButton("ok", self)
        # self.b1.move(50, 50)
        self.resize(size[0]/2, size[1]/2)
        self.move(450, 250)
        self.setWindowModality(Qt.ApplicationModal)
        self.layout = QVBoxLayout()

        # Create central Widget
        self.centralWidget = QWidget(self)

        # Create combobox and add available Host APIs   
        self.host = QComboBox(self.centralWidget)
        self.host.setToolTip('This are the HOST APIs:')
        self.hostapi_list = [hostapi['name'] for hostapi in sd.query_hostapis()]
        for i in self.hostapi_list:
            self.host.addItem(i)
        self.layout.addWidget(self.host)
        self.host.currentTextChanged.connect(on_host_changed)

        # create combobox and add available outputs
        self.device = QComboBox(self.centralWidget)
        self.host.setToolTip('Choose your sound device:')
        self.device_ids = []
        self.hostapi = sd.query_hostapis(self.host.currentIndex())
        self.device_ids = [idx
                for idx in self.hostapi['devices']
                if sd.query_devices(idx)['max_output_channels'] > 0]
        #print('device_ids:', self.device_ids)
        self.device_list = [sd.query_devices(idx)['name'] 
                            for idx in self.device_ids]
        default = self.hostapi['default_output_device']
        #print('device_list:', self.device_list)
        for i in self.device_list:
            self.device.addItem(i)
        self.layout.addWidget(self.device)
        self.device.currentTextChanged.connect(on_device_changed)
        
        self.default = self.hostapi['default_output_device']
       
        """ if self.default >= 0:
            self.device_list.currentText(self.device_ids.currentIndex(self.default))
            self.device_list.currentTextChanged.connect(update_device_list)
             """
            #self.result = device_ids[device_list.currentText()]

            #hostapi_list.bind('<<ComboboxSelected>>', update_device_list)
            #device_list.bind('<<ComboboxSelected>>', select_device)

            #with contextlib.suppress(sd.PortAudioError):
                #hostapi_list.current(sd.default.hostapi)
                #hostapi_list.event_generate('<<ComboboxSelected>>')

        self.setLayout(self.layout)
        self.exec_()

# Buttons / Layout 
class Gui(QWidget):   
    
    def __init__(self):
        QWidget.__init__(self)

        # size and title
        global size
        size = [400,360]
        self.resize(size[0],size[1])   
        self.setWindowTitle("HAIOPY GUI")

        # use layout
        self.layout = QVBoxLayout()
        self.centralWidget = QWidget(self)
        self.rec_button = QPushButton('Start Recording')
        self.stop_button = QPushButton('Stop Recording')
        self.settings_button = QPushButton('Settings')
        #self.play_button = QPushButton('Play Recording')

        # create a progressbar
        # self.meter = QProgressBar(self.centralWidget)
        # self.meter.setOrientation(QtCore.Qt.Vertical)
        # self.meter.setMaximum(1)
        # self.meter.setMinimum(0)

        # layout add/set
        self.layout.addWidget(self.rec_button)
        self.layout.addWidget(self.stop_button)
        self.layout.addWidget(self.settings_button)
        #self.layout.addWidget(self.play_button)
        # self.layout.addWidget(self.meter, alignment=Qt.AlignCenter)
        self.setLayout(self.layout)

        # Click Actions
        self.rec_button.clicked.connect(rec_button_on_click)
        self.stop_button.clicked.connect(stop_button_on_click)
        self.settings_button.clicked.connect(settings_button_on_click)
        # self.meter.setValue(int(0.75))
        #self.play_button.clicked.connect(play_button_clicked)



# App exe
if __name__ == "__main__":
    # Start App
    app = QtWidgets.QApplication(sys.argv)
    # Layout Colors
    p = QPalette()
    p.setColor(p.Window, QColor(53, 53, 53))
    p.setColor(p.WindowText, Qt.white)
    p.setColor(p.Button, QColor(53, 53, 53))
    p.setColor(p.ButtonText, Qt.white)
    app.setPalette(p)
    app.setStyle('Fusion')
    # Run App and Exit on close
    main = Gui()
    main.show()
    sys.exit( app.exec_() )


# %%
