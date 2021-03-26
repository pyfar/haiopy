"""Main module."""
#%%
import numpy as np                          # scientific computing lib
import scipy.io.wavfile as wavfile          # read/write wavfiles with scipy
from pyfar import Signal                    # managing audio signals
import pyfar.plot as plot                   # beatiful plots with pyfar
import sounddevice as sd                    # sounddevice / hostapi handling
import soundfile as sf                      # cross-platform file reading/writing
import queue                                # information exchange between threads
import sys                                  # used for prunting errors to std stream
import tempfile                             # create temporary files
import threading                            # create threads and (non-blocking) events
import matplotlib.pyplot as plt             # plots and graphs


class Device():
    """Wrapper-class for sounddevice."""
    def __init__(self, inp=0, out=1):
        # initialize parameters
        self.input = inp
        self.output = out
        sd.default.device = (self.input, self.output)
        sd.default.samplerate = sd.query_devices(device=self.input)["default_samplerate"]

    def set_device(self, inp, out):
        self.input = inp
        self.output = out
        sd.default.device = (self.input, self.output)
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

    def show_all(self):
        print(sd.query_devices())

    def set_samplerate(self, sr):
        sd.default.samplerate = sr



class _AudioIO(object):
    """Abstract Container Class for haiopy-classes"""
    def __init__(
        self,
        blocksize=2048,
        buffersize=20,
        sampling_rate=48000,
        dtype='float32'):

        # initialize global-parameters
        self.blocksize = blocksize
        self.buffersize = buffersize
        self.sampling_rate = sampling_rate

        self._VALID_DTYPES = ["int8", "int16", "int32", "float32"]  # provided by sd.Streams
        # if dtype in self._VALID_DTYPES:
        self.dtype = dtype

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
    def buffersize(self, value):
        """Set Buffersize"""
        self._buffersize = value

    @property
    def sampling_rate(self):
        """Get Sampling_Rate"""
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        """Set Sampling_Rate"""
        self._sampling_rate = value

    @property
    def dtype(self):
        """Get dtype"""
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        """Set dtype"""
        if value in self._VALID_DTYPES:

            self._dtype = value
        else:
            raise ValueError('Wrong dtype')

    def check_input_sampling_rate(self, sr):
        if self.sampling_rate is None or self.sampling_rate == sr:
            self.sampling_rate = sr
        else:
            raise ValueError(
                        'Sampling_rates do not Match!', self.sampling_rate, sr)

    def check_input_dtype(self, dt):
        if self.dtype is None or self.dtype == dt:
            self.dtype = dt
        else:
            raise ValueError(
                        'Dtypes do not Match!', self.dtype, dt)


class Play(_AudioIO):
    """
    Class for Playback of WAV or pyfar.Signal-objects with chosen sounddevice.
    """
    def __init__(
            self,
            audio_in,
            blocksize=2048,
            buffersize=20,
            device_out=1,
            channels_out=2,
            sampling_rate=None,
            dtype='float32',
            ):

        self.audio_q = queue.Queue()
        self.event = threading.Event()
        
        # initialize global parameter and valid parameter spaces
        _AudioIO.__init__(self, blocksize, buffersize, sampling_rate, dtype)

        # attribute:
        self.audio_in = audio_in            # audio in object
        self.device_out = device_out        # device object for playback
        self.channels_out = channels_out    # number of output channels
        self.check_audio_in()

    @property
    def device_out(self):
        """Get Index of Output Device"""
        return self._device_out

    @device_out.setter
    def device_out(self, idx):
        """Set Index of Output Device"""
        if idx in range(len(sd.query_devices())) \
                and sd.query_devices(idx)['max_output_channels'] > 0:
            self._device_out = int(idx)
        else:
            raise ValueError('index of output device (device_out) not found')

    @property
    def channels_out(self):
        """ Get number of Output Channels """
        return self._channels_out

    @channels_out.setter
    def channels_out(self, value):
        """ Set number of Output Channels """
        if value <= sd.query_devices(self._device_out)['max_output_channels']:
            self._channels_out = int(value)
        else:
            raise ValueError('number of output channels exceeds output device, max output channels:',
            sd.query_devices(self._device_out)['max_output_channels'])

    @property
    def audio_in(self):
        """ Get the Type of Input Audio """
        return self._audio_in

    @audio_in.setter
    def audio_in(self, value):
        """ Set the Type of Input Audio """
        self._audio_in = value

    def check_audio_in(self):
        """ Test if audio input is WAV (string) or pyfar.Signal.
        And check and recieve Inputs Samplingrate"""
        if isinstance(self.audio_in, Signal):
            self.type_in = 'signal'
            self.check_input_dtype(self.audio_in.dtype)
            self.check_input_sampling_rate(self.audio_in.sampling_rate)
        elif isinstance(self.audio_in, str) \
                and self.audio_in.split('.')[-1] == 'wav':
            self.type_in = 'wav'
            testdata, sr = sf.read(self.audio_in, dtype=self.dtype)
            self.check_input_dtype(testdata.dtype)
            self.check_input_sampling_rate(sr)

    def create_output_stream(self):
        output_stream = sd.OutputStream(
                    samplerate=self.sampling_rate, blocksize=self.blocksize,
                    device=self.device_out, channels=self.channels_out,
                    dtype=self.dtype, callback=self.audio_callback,
                    finished_callback=self.event.set)
        output_stream.start()

    # Methoden zum Playback:::
    def audio_callback(self, outdata, frames, time, status):
        """ .. ."""
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
            outdata[:] = data  # für stereo.wavs <-- was ist mit mono wavs?

    def play(self):
        if self.type_in == 'wav':
            with sf.SoundFile(self.audio_in) as f:
                for _ in range(self.buffersize):
                    data = f.read(frames=self.blocksize, dtype=self.dtype)
                    if data.size == 0:
                        break
                    self.audio_q.put_nowait(data)  # Pre-fill queue
                self.create_output_stream()

                timeout = self.blocksize * self.buffersize / self.sampling_rate

                while data.size != 0:
                    data = f.read(self.blocksize, dtype=self.dtype)
                    self.audio_q.put(data, timeout=timeout)
                self.event.wait()  # Wait until playback is finished

        elif self.type_in == 'signal':
            read_idx = 0  # reading index for blockwise operation
            for _ in range(self.buffersize):
                data = self.audio_in.time[read_idx:read_idx + self.blocksize]     
                if data.size == 0:
                    break
                self.audio_q.put_nowait(data)
                read_idx += self.blocksize   # Prefill the queue

            self.create_output_stream()

            self.timeout = self.blocksize * self.buffersize / self.sampling_rate

            while data.size != 0:
                for blocks in range(self.blocksize):          
                    data = self.audio_in.time[read_idx:read_idx + self.blocksize]
                    self.audio_q.put(data, timeout=self.timeout)
                    read_idx += self.blocksize
            self.event.wait()  # Wait until playback is finished


class Record(_AudioIO):
    """ Class for duration-based or infinite recording of WAV or pyfar.Signal-objects with chosen sounddevice."""
    stream = None

    def __init__(self, audio_out, blocksize=2048, buffersize=20, device_in=0, duration=None):
        _AudioIO.__init__(self, blocksize, buffersize)  # Initialize global parameters and instances

        self._VALID_TYPES = ["wav", "signal"]  # Initialize valid parameter spaces
        self.audio_out = audio_out
        self.device_in = device_in  # chosen input device for recording
        self.duration = duration  # timing the record
        self.recording = self.previously_recording = False
        self.audio_q = queue.Queue()
        self.create_stream()
        self.data_array = []  # Leere Liste/Array

        if audio_out in self._VALID_TYPES:
            self.type_out = self.audio_out
        else:
            raise TypeError("Incorrect type, needs to be wav or Signal.")

    @property
    def device_in(self):
        """ Get the Index of the Input Device """
        return self._device_in

    @device_in.setter
    def device_in(self, idx):
        """ Set the Index of the Input Device """
        if idx in range(len(sd.query_devices())) \
            and sd.query_devices(idx)['max_input_channels'] > 0:
            self._device_in = int(idx)
        else:
            raise ValueError('index of input device (device_in) not found')


    def create_stream(self, device = None):
        if self.stream is not None:
            self.stream.close()
        self.stream = sd.InputStream(
            samplerate=self.sampling_rate, device=self.device_in, channels=2,
            blocksize=self.blocksize, callback=self.audio_callback, dtype='float32')
        self.stream.start()


    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        #if status:
        #    print(status)
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


    def data_writing_thread(self,*,q):
        """Write data from queue to pyfar.Signal until *None* is received."""
        while True:
            data = q.get()
            if data is None:
                break
            self.data_array = np.append(self.data_array,np.array(data))
        self.data_array = np.reshape(self.data_array, (-1,2))
        self.signal = Signal(self.data_array, self.stream.samplerate, dtype='float32')
        return self.signal


    def on_rec(self):
        """Record and write data in a new thread into a tempfile in current directory."""
        self.recording = True
        if self.type_out == "wav":
            filename = tempfile.mktemp(prefix='Record_', suffix='.wav', dir='')
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
                kwargs=dict(q=self.audio_q,
                ),
            )
            self.data_writing_thread.start()

    def on_stop(self):
        """Stop recording and close file/data writing threads."""
        self.recording = False
        if self.type_out == "wav":
            self.wav_thread.join()
        elif self.type_out == "signal":
            self.data_writing_thread.join()
        print('Stopped')

    def record(self, duration):
        """Start recording and stop recording after duration (and return pyfar.Signal)."""
        self.on_rec()
        timer = threading.Timer(duration, self.on_stop)
        timer.start()
        if self.type_out == "signal":
            return self.signal


class PlayRecord(Play, Record):
    """ Class for simultanous playback and recording of WAV or pyfar.Signal-objects with chosen sounddevice."""
    stream = None
    
    def __init__(self, audio_in, audio_out, blocksize=2048, buffersize=20,
                device_in=0, device_out=1, sampling_rate=None, dtype=None,
                channels_in=None, channels_out=2):

        Play.__init__(self, audio_in, blocksize, buffersize, device_out, channels_out, dtype)
        Record.__init__(self, audio_out, device_in, channels_in)

        self.inputq = queue.Queue()
        self.outputq = queue.Queue()
        self.event = threading.Event()
        self.device_in = device_in
        self.device_out = device_out
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.blocksize = blocksize
        self.buffersize = buffersize
        self.data_array = []
        self.outfilename = tempfile.mktemp(prefix='PlayRecord_', suffix='.wav', dir='')

        if audio_out in self._VALID_TYPES:
            self.type_out = self.audio_out
        else:
            raise TypeError("Incorrect type, needs to be wav or Signal.")

    def playrec_callback(self, indata, outdata, frames, time, status):
        """PlayRecord callback function."""
        assert frames == self.blocksize
        #if status:
        #    print(status)

        # 1st outputq:
        try:
            outputdata = self.outputq.get_nowait()
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)
            raise sd.CallbackAbort
        if len(outputdata) < len(outdata):
            outdata[len(outputdata):] = 0
            self.inputq.put(None)  # Put NONE to finish the recording
            raise sd.CallbackStop

        else:
            # output:
            outdata[:] = outputdata
            # input:
            self.inputq.put(indata.copy())
    
    def create_stream(self):
        """Open and start audio stream with sd.Stream."""
        self.playrec_stream= sd.Stream(
                samplerate=self.sampling_rate, blocksize=self.blocksize,
                device=(self.device_in, self.device_out), channels=2, dtype=self.dtype,
                callback=self.playrec_callback, finished_callback=self.event.set)
        self.playrec_stream.start()

    def playrec(self):
        """Open and start audio stream with sd.Stream."""
        self.check_audio_in()
        # for type WAV
        if self.type_in == 'wav':
            with sf.SoundFile(self.audio_in) as f:
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
                with sf.SoundFile(file=self.outfilename, mode='x',
                    samplerate=int(f.samplerate),
                    channels=f.channels) as file:
                    while True:
                        inputdata = self.inputq.get()
                        if inputdata is None:
                            print('Finished')
                            break
                        file.write(inputdata)
        # for type pyfar.Signal
        elif self.type_in == 'signal':
            read_idx = 0  # reading index for blockwise operation
            for _ in range(self.buffersize):
                data = self.audio_in.time[read_idx:read_idx + self.blocksize]
                if data.size == 0:
                    break
                self.outputq.put_nowait(data)
                read_idx += self.blocksize
            self.create_stream()  # create & start the stream
            self.timeout = self.blocksize * self.buffersize / 48000
            while data.size != 0:
                for blocks in range(self.blocksize):          
                    data = self.audio_in.time[read_idx:read_idx + self.blocksize]
                    self.outputq.put(data, timeout=self.timeout)
                    read_idx += self.blocksize
                    self.event.wait()  # Wait until playback is finished
            #recording:::
            while True:
                inputdata = self.inputq.get()
                if inputdata is None:
                    break
                self.data_array = np.append(self.data_array, np.array(inputdata))
            self.data_array = np.reshape(self.data_array, (-1, 2))
            self.signal2 = Signal(self.data_array, 48000, dtype='float32')
            print('Finished')
            return self.signal2


# %%
### TEST SIGNAL

testdata, sr = sf.read('playtest2.wav', dtype='float32')
signal = Signal(testdata, sr, dtype='float32')


#%%
### TEST PLAY CLASS
playsi = Play('playtest2.wav', device_out=2, channels_out=2,sampling_rate=48000,dtype='float32')
#playsi = Play(signal, device_out=6, channels_out=2, sampling_rate=44100)
playsi.play()

#%%
### TEST RECORD CLASS
recordsi = Record('wav', device_in=0)
#recordedsignal = recordsi.record(duration=10)
#playsi2 = Play(recordedsignal.time,device_out=5,channels_out=1)

#%%
### TEST PLAY_RECORD CLASS
#playrectest = PlayRecord(audio_in='playtest2.wav', audio_out='wav', device_in=1, device_out=7)
playrectest = PlayRecord(audio_in='playtest2.wav', audio_out='wav', device_in=0, device_out=2)
vartest = playrectest.playrec()