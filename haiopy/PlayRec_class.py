
# %%



import numpy as np
import sounddevice as sd
import soundfile as sf 
import queue 
import sys
import tempfile 
import time
import threading


class PlayRec():

    stream = None

    def __init__(self):

        self.inputq = queue.Queue()
        self.outputq = queue.Queue()

        self.event = threading.Event()

        self.blocksize = 2046
        self.buffersize = 20

        self.outfilename = tempfile.mktemp(prefix='playrec_recording_', suffix='.wav', dir ='')


    def callback(self,indata, outdata, frames, time, status):
        assert  frames == self.blocksize


        if status:
            print(status)
        #1st outputq:

        try:
            outputdata = self.outputq.get_nowait()
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)
            raise sd.CallbackAbort
        if len(outputdata) < len(outdata):
        
            outdata[len(outputdata):] = 0
            
            self.inputq.put(None) #Put NONE to finish Recording

            raise sd.CallbackStop
            

        else:
        
            #output:
            outdata[:] = outputdata

            # input:
            
            self.inputq.put(indata.copy())


    def playrec(self,filename):
        with sf.SoundFile(filename) as f:
            for _ in range(self.buffersize):
                data = f.read(frames=self.blocksize, dtype='float32')
                if data.size == 0:
                    break
                self.outputq.put_nowait(data)

            self.stream = sd.Stream(samplerate=f.samplerate, blocksize=self.blocksize,
                device=(0,2), channels=f.channels, dtype='float32',
                callback=self.callback, finished_callback=self.event.set())

            self.stream.start()

            self.timeout = self.blocksize * self.buffersize / f.samplerate

            while data.size != 0:
                data = f.read(self.blocksize, dtype='float32')
                self.outputq.put(data, timeout=self.timeout)
                self.event.wait()  # Wait until playback is finished



            with sf.SoundFile(file=self.outfilename,mode='x',
                samplerate=int(self.stream.samplerate),
                channels=f.channels) as file:
                while True:
                    inputdata = self.inputq.get()
                    self.event.wait() # waiting until callbacks-finsished
                    if inputdata is None:
                        break
                    
                    #event.wait() #waiting for stream to close
                    file.write(inputdata)



        
#%%

playrec_test = PlayRec()

time.sleep(2)

playrec_test.playrec('playtest2.wav')

# %%
