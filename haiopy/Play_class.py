#%%

import numpy as np
import sounddevice as sd
import soundfile as sf 
import queue 
import sys
import tempfile 
import time
import threading




class Play(): 

    stream = None

    def __init__(self, filename): 

        self.audio_q = queue.Queue()
        self.event = threading.Event()

        self.blocksize = 2046
        self.buffersize = 20 

        self.filename = filename

        


    def audio_callback(self,outdata, frames, time, status):
        
        assert frames == self.blocksize

        try:
            data = self.audio_q.get_nowait()
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)
            raise sd.CallbackAbort

        if len(data) < len(outdata):
            
            outdata[len(data):] = 0
            raise sd.CallbackStop
        else:
            #outdata[:,0] = data #für mono.wavs

            outdata[:] = data #für stereo.wavs


    def play(self):

        

        with sf.SoundFile(self.filename) as f:
            for _ in range(self.buffersize):
                data = f.read(frames=self.blocksize, dtype='float32')
                if data.size == 0:
                    break
                self.audio_q.put_nowait(data)  # Pre-fill queue
            stream = sd.OutputStream(  
                samplerate=f.samplerate, blocksize=self.blocksize,
                device=(0,1), channels=f.channels, dtype='float32',
                callback=self.audio_callback, finished_callback=self.event.set)
            with stream:
                timeout = self.blocksize * self.buffersize / f.samplerate
                while data.size != 0:
                    data = f.read(self.blocksize, dtype='float32')
                    self.audio_q.put(data, timeout=timeout)
                self.event.wait()  # Wait until playback is finished
   



# %%


playtest = Play('playtest3.wav')

playtest.play()
 # %%
