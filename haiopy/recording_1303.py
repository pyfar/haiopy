#%%


import numpy as np
import sounddevice as sd
import soundfile as sf 
import queue 
import sys
import tempfile 
import time

import threading 




#### Tasks: 

#inheritance von device class
#recording duration 




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
            samplerate=48000, device=(0,2), channels=2 ,callback= self.audio_callback)
        self.stream.start()

    def file_writing_thread(self,*, q, **soundfile_args):
        """Write data from queue to file until *None* is received."""
        
        with sf.SoundFile(**soundfile_args) as file:
            while True:
                data = q.get()
                print(data)
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



        




#%%

# hier kommt der aufruf dann mit 5 sek verzögerung dann der record stop. i hope :D 
rec1 = Recording()

time.sleep(1)
rec1.on_rec()

time.sleep(5)

rec1.on_stop()


#%%

rec1 = Recording()

x

rec1.record(4)







