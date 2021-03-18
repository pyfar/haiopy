"""Main module."""


import warnings
import numpy as np

import scipy.io.wavfile as wavfile

import pyfar
from pyfar import Signal                     # managing audio signals
from pyfar.coordinates import Coordinates    # managing satial sampling points
from pyfar.orientations import Orientations  # managing orientation vectors
from pyfar.spatial import samplings          # creating predefined spatial sampling grids
import pyfar.plot as plot                    # beatiful plots
import pyfar.dsp.filter as filt



import sounddevice as sd
import soundfile as sf 
import queue 
import sys
import tempfile 
import time
import threading



class _AudioIO(object):

    """Abstract Container Class for haiopy-classes"""

    def __init__(self, blocksize=2048, buffersize=20):

        
        # initialize global-parameters
        self.blocksize = blocksize
        self.buffersize = buffersize
        #self.sampling_rate = sampling_rate


    @property
    def blocksize(self):
        """Get Blocksize"""
        return self._blocksize

    @blocksize.setter 
    def blocksize(self, value):
        """Set Blocksize"""
        self._blocksize = value


    @property
    def buffersize(self):
        """Get Buffersize"""
        return self._buffersize

    @buffersize.setter 
    def buffersize(self,value):
        """Set Buffersize"""
        self._buffersize = value 


    #Sampling_rate- setter fehlt noch. und einbindung zu device class

    

class Play(_AudioIO): 

    """ Class for Playback of .wav or Signal-Objects with chosen Sounddevice """

    def __init__(self, 
                audio_in, # isinstance entweder wav string oder pyfar signal type
                blocksize=2048,
                buffersize=20,
                device_out = 1,
                channels_out = 2): 

        self.audio_q = queue.Queue() 
        self.event = threading.Event()


        # initialize global parameter and valid parameter spaces

        _AudioIO.__init__(self, blocksize, buffersize)  

        self.blocksize = blocksize 
        self.buffersize = buffersize 

        # attribute:
        self.audio_in = audio_in 
        self.device_out = device_out     #   device object for playback
        self.channels_out = channels_out # number of output channels


    @property
    def device_out(self):
        """Get Index of Output Device"""
        return self._device_out

    @device_out.setter 
    def device_out(self,value):
        """Set Index of Output Device"""
        self._device_out = int(value)

    @property
    def channels_out(self):
        """ Get number of Output Channels """
        return self._channels_out

    @channels_out.setter
    def channels_out(self,value):
        """ Set number of Output Channels """
        self._channels_out = value


    def check_audio_in(self):


        """ if isinstance(self.audio_in, str):
            self.filename = self.audio_in

            self.type = 'wav' """


        if isinstance(self.audio_in, Signal):
            self.signal = self.audio_in

            self.type_in = 'signal'

        else:
            self.filename = self.audio_in

            self.type_in = 'wav'
            #raise TypeError('audio_in must be of type Signal or String(Wav)')

    #Methoden zum Playback:::

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
            outdata[:] = data #für stereo.wavs


    def play(self):

        self.check_audio_in()

        if self.type_in == 'wav':

            with sf.SoundFile(self.filename) as f:
                for _ in range(self.buffersize):
                    data = f.read(frames=self.blocksize, dtype='float32')
                    if data.size == 0:
                        break
                    self.audio_q.put_nowait(data)  # Pre-fill queue
                stream = sd.OutputStream(  
                    samplerate=f.samplerate, blocksize=self.blocksize,
                    device=self.device_out, channels=f.channels, dtype=str(data.dtype),
                    callback=self.audio_callback, finished_callback=self.event.set)
                
                stream.start()
                
                timeout = self.blocksize * self.buffersize / f.samplerate

                while data.size != 0:
                    data = f.read(self.blocksize, dtype='float32')
                    self.audio_q.put(data, timeout=timeout)
                self.event.wait()  # Wait until playback is finished


        elif self.type_in == 'signal':
        
            read_idx = 0 # reading index for blockwise operation

            for _ in range(self.buffersize):
                data = self.signal.time[read_idx:read_idx + self.blocksize]
                                
                if data.size == 0:
                    break

                self.audio_q.put_nowait(data)
                read_idx += self.blocksize   #Prefille the queue

            self.stream = sd.OutputStream(samplerate=48000, blocksize=self.blocksize,
                    device=self.device_out, channels=2, dtype=str(data.dtype),
                    callback=self.audio_callback, finished_callback=self.event.set)

            self.stream.start()

            self.timeout = self.blocksize * self.buffersize / self.stream.samplerate

            while data.size != 0:
                for blocks in range(self.blocksize):          
                
                    data = self.signal.time[read_idx:read_idx + self.blocksize]
                    self.audio_q.put(data, timeout=self.timeout)
                    read_idx += self.blocksize
            self.event.wait()  # Wait until playback is finished
   


class Recording(_AudioIO):

    stream = None

    def __init__(self, 
                audio_out,
                blocksize=2048,
                buffersize=20,
                device_in=0,
                duration=None): 

        #initialize valid parameter spaces
        self._VALID_TYPES = ["wav","signal"]


        #initialize global parameters and instances

        _AudioIO.__init__(self, blocksize, buffersize) 

        self.audio_out = audio_out 
        self.device_in = device_in # chosen input device for recording
        self.duration = duration #timing the record


        self.recording = self.previously_recording = False
        self.audio_q = queue.Queue()
        
        self.create_stream()

        self.data_array = [] #Leere Liste/Array


        if audio_out in self._VALID_TYPES:
            self.type_out = self.audio_out
        else:
            raise TypeError("Incorrect type, needs to be wav or Signal.")
 

    @property
    def device_in(self):
        """ Get the Index of the Input Device """
        return self._device_in

    @device_in.setter
    def device_in(self,value):
        """ Set the Index of the Input Device """
        self._device_in = int(value)


    def create_stream(self, device = None):
        if self.stream is not None:
            self.stream.close()
        
        self.stream = sd.InputStream(
            samplerate=48000, device=self.device_in, channels=2,
            blocksize = self.blocksize,callback= self.audio_callback)
        
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):

        """This is called (from a separate thread) for each audio block."""

        if status:
            print(status)

        if self.recording == True:
            self.audio_q.put(indata.copy())
            self.previously_recording = True

        else:
            if self.previously_recording:
                self.audio_q.put(None)
                self.previously_recording = False    

    def file_writing_thread(self,*, q, **soundfile_args):
        """Write data from queue to file until *None* is received."""
        
        with sf.SoundFile(**soundfile_args) as file:
            while True:
                data = q.get()
                if data is None:
                    break
                file.write(data)
## signal

    def data_writing_thread(self,*,q):
        
        while True:
            data = q.get()
            if data is None:
                break
                 
            self.data_array = np.append(self.data_array,np.array(data))
        
        self.data_array = np.reshape(self.data_array, (-1,2))

        self.signal = Signal(self.data_array,self.stream.samplerate,dtype='float32')
        
        return self.signal


    def on_rec(self):

        self.recording = True

        if self.type_out == "wav":

            filename = tempfile.mktemp(prefix='test_recording_', suffix='.wav', dir ='')

            if self.audio_q.qsize() != 0:
                print('WARNING:QUEUE NOT EMPTY')

            
            self.wav_thread = threading.Thread(
                target=self.file_writing_thread,
                kwargs=dict(
                    file=filename,
                    mode='x',
                    samplerate=int(self.stream.samplerate),
                    channels=self.stream.channels,
                    q=self.audio_q,
                ),
            )
            self.wav_thread.start()

            print('Recording')


        elif self.type_out == "signal":
            self.recording = True

            self.data_writing_thread = threading.Thread(
                target=self.data_writing_thread,
                kwargs= dict(q=self.audio_q,
                ),
            )

            self.data_writing_thread.start()


    def on_stop(self):

        self.recording = False

        if self.type_out == "wav":
            self.wav_thread.join()

        elif self.type_out == "signal":
            self.data_writing_thread.join()
            
        print('Stopped')

    
    def record(self,duration):

        self.on_rec() # duration übergeben. duration default None.

        time.sleep(duration)
  
        self.on_stop()

        if self.type_out == "signal":
            return self.signal


####ES FEHLT HIER NOCH DIE ZEITIMPLMENTIERUNG!!!
        


class PlayRec(Play,Recording):

    stream = None

    def __init__(self,
                audio_in, 
                audio_out,
                blocksize=2048,
                buffersize=20,
                device_in=0,
                device_out=1):

        self.inputq = queue.Queue()
        self.outputq = queue.Queue()

        self.event = threading.Event()


        Play.__init__(self, audio_in,blocksize, 
                        buffersize,device_out,channels_out=2)

        Recording.__init__(self,audio_out,blocksize,buffersize,device_in)



        self.device_in = device_in
        self.device_out = device_out
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.blocksize = blocksize
        self.buffersize = buffersize
        

        if audio_out in self._VALID_TYPES:
            self.type_out = self.audio_out
        else:
            raise TypeError("Incorrect type, needs to be wav or Signal.")

        self.data_array = []

        self.outfilename = tempfile.mktemp(prefix='playrec_recording_', suffix='.wav', dir ='')


    def playrec_callback(self,indata, outdata, frames, time, status):
         
        assert frames == self.blocksize
        

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

    
    def create_stream(self):

        self.playrec_stream= sd.Stream(
                samplerate=48000, blocksize=self.blocksize,
                device=(self.device_in,self.device_out), channels=2, dtype='float32',
                callback=self.playrec_callback, finished_callback=self.event.set)

        self.playrec_stream.start()


    def playrec(self):
        self.check_audio_in()

        if self.type_in == 'wav':

            with sf.SoundFile(self.filename) as f:
                for _ in range(self.buffersize):
                    data = f.read(frames=self.blocksize, dtype='float32')
                    if data.size == 0:
                        break
                    self.outputq.put_nowait(data)

                self.create_stream() # create & start the stream
                
                self.timeout = self.blocksize * self.buffersize / f.samplerate

                while data.size != 0:
                    data = f.read(self.blocksize, dtype='float32')
                    self.outputq.put(data, timeout=self.timeout)
                self.event.wait()  # Wait until playback is finished



                with sf.SoundFile(file=self.outfilename,mode='x',
                    samplerate=int(48000),
                    channels=f.channels) as file:
                    while True:
                        inputdata = self.inputq.get()
                        #self.event.wait() # waiting until callbacks-finsished
                        if inputdata is None:
                            print('Finished')
                            break
                        
                        file.write(inputdata)



        elif self.type_in == 'signal':

            read_idx = 0 # reading index for blockwise operation

            for _ in range(self.buffersize):
                data = self.signal.time[read_idx:read_idx + self.blocksize]
                                    
                if data.size == 0:
                    break
                self.outputq.put_nowait(data)
                read_idx += self.blocksize


            self.create_stream() # create & start the stream

            self.timeout = self.blocksize * self.buffersize / 48000

            while data.size != 0:
                for blocks in range(self.blocksize):          
                    
                    data = self.signal.time[read_idx:read_idx + self.blocksize]
                    self.outputq.put(data, timeout=self.timeout)
                    read_idx += self.blocksize
            self.event.wait()  # Wait until playback is finished

            #recording:::
            while True:
                #self.event.wait()  # Wait until playback is finished
                inputdata = self.inputq.get()
                #self.event.wait()  # Wait until playback is finished
                #print(inputdata)
                if inputdata is None:
                    break
                    
                
                self.data_array = np.append(self.data_array,np.array(inputdata))
                

            
            self.data_array = np.reshape(self.data_array, (-1,2))

            self.signal2 = Signal(self.data_array,48000,dtype='float32')

            print('Finished')
            
            return self.signal2

                    



class Device():

    """Wrapper-class for sounddevice for haiopy"""
    def __init__(self, inp=0, out=1):
        # initialize parameters
        self.input = inp
        self.output = out
        sd.default.device = (self.input, self.output)
        sd.default.samplerate = sd.query_devices(device=self.input)["default_samplerate"]

    def set_device(self, inp, out):
        self.input = inp
        self.output = out
        sd.default.device=(self.input, self.output)
        sd.default.samplerate = sd.query_devices(device=self.input)["default_samplerate"]
        
    def show_io(self):
        print("\033[1m" + "Input:\n" + "\033[0m", sd.query_devices(device=self.input))
        print()
        print("\033[1m" + "Output:\n" + "\033[0m", sd.query_devices(device=self.output))

    def show_max_channels(self):
        print('Max Channels for Input Device:', sd.query_devices(device=self.input)['max_input_channels'])
        print('Max Channels for Output Device:', sd.query_devices(device=self.output)['max_output_channels'])

    def set_channels(self, ichan, ochan):
        sd.default.channels = (ichan, ochan)
        return ichan, ochan

    def show_all(self):
        print(sd.query_devices())

    def set_samplerate(self, sr):
        sd.default.samplerate = sr
        
