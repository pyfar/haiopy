from multiprocessing import Event
import asyncio
from os import stat
import sys
import sounddevice as sd
from abc import (ABCMeta, abstractmethod, abstractproperty)

from haiopy.generators import ArrayBuffer, InputArrayBuffer, OutputArrayBuffer


def list_devices():
    pass


class _Device(object):
    def __init__(
            self):
        super().__init__()

    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def playback():
        pass

    @abstractmethod
    def record():
        pass

    @abstractmethod
    def playback_record():
        pass

    @abstractmethod
    def initialize_playback():
        pass

    @abstractmethod
    def initialize_record():
        pass

    @abstractmethod
    def initialize_playback_record():
        pass

    @abstractmethod
    def abort():
        pass


class AudioDevice(_Device):
    def __init__(
            self,
            id=sd.default.device,
            sampling_rate=44100,
            block_size=512,
            dtype='float32',
            latency=None,
            extra_settings=None,
            clip_off=None,
            dither_off=None,
            never_drop_input=None,
            prime_output_buffers_using_stream_callback=None
            ):
        super().__init__()

        id = sd.query_devices(id)['name']
        self._name = sd.query_devices(id)['name']

        n_channels_input = sd.query_devices(id)['max_input_channels']
        n_channels_output = sd.query_devices(id)['max_output_channels']
        sd.check_input_settings(
            device=id,
            channels=n_channels_input,
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)

        sd.check_output_settings(
            device=id,
            channels=n_channels_output,
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)

        self._loop = asyncio.get_event_loop()

        self.id = id
        self.dtype = dtype
        self._block_size = block_size
        self._sampling_rate = sampling_rate
        self._extra_settings = extra_settings

        self._callback = None
        self._stream = None
        self._input_buffer = None
        self._output_buffer = None

        self._stream_finished = Event()

    @property
    def n_channels_input(self):
        """The number of input channels supported by the device"""
        return sd.query_devices(self.id)['max_input_channels']

    @property
    def n_channels_output(self):
        """The number of output channels supported by the device"""
        return sd.query_devices(self.id)['max_output_channels']

    def check_settings(
            self, sampling_rate, dtype, extra_settings,):
        """Check if settings are compatible with the physical devices.

        Parameters
        ----------
        sampling_rate : int
            The audio sampling rate
        dtype : np.float32, np.int8, np.int16, np.int32
            The audio buffer data type
        extra_settings : extra settings
            Audio API specific settings.
        """
        sd.check_input_settings(
            device=self.id,
            channels=self.n_channels_input,
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)

        sd.check_output_settings(
            device=self.id,
            channels=self.n_channels_output,
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)

    @property
    def name(self):
        """The name of the device
        """
        return self._name

    @property
    def sampling_rate(self):
        """The sampling rate of the audio device.
        """
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        self.check_settings(value, self.dtype, self._extra_settings)

    @property
    def block_size(self):
        """The block size of the audio buffer.
        """
        return self._block_size

    @property
    def stream(self):
        """The sounddevice audio stream.
        """
        return self._stream

    def finished_callback(self) -> None:
        """Custom callback after a audio stream has finished."""
        print("I'm finished.")
        pass

    def _finished_callback(self) -> None:
        """Portaudio callback after a audio stream has finished.
        """
        self._stream_finished.set()
        self.finished_callback()
        self.stream.stop()

    def output_callback(self, outdata, frames, time, status) -> None:
        """Portudio callback for output streams

        Parameters
        ----------
        outdata : array
            Output buffer view
        frames : int
            Length of the buffer
        time : PaTimestamp
            Timestamp of the callback event
        status : sounddevice.CallbackFlags
            Portaudio status flags

        Raises
        ------
        sd.CallbackAbort
            Abort the playback if a buffer underflow occurs.
        sd.CallbackStop
            Stop the playback if the output queue is empty.
        """
        assert frames == self.block_size
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort('Buffer underflow')
        assert not status

        try:
            outdata[:] = next(self._output_buffer).T
        except StopIteration:
            raise sd.CallbackStop("Buffer empty")

    def input_callback(self, indata, frames, time, status):
        assert frames == self.block_size
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort('Buffer underflow')
        assert not status

        try:
            next(self._input_buffer)[:] = indata.T
        except StopIteration:
            raise sd.CallbackStop("Buffer empty")

    def init_input_buffer(self, data):
        """Initialize the output buffer.

        Parameters
        ----------
        data : haiopy.ArrayBuffer, generator
            The input buffer to which the input data is written block-wise.
        """
        self._input_buffer = ArrayBuffer(
            self.block_size, data)

    def init_output_buffer(self, data):
        """Initialize the output buffer.

        Parameters
        ----------
        data : haiopy.ArrayBuffer, generator
            The output buffer from which the output data is read block-wise.
        """
        self._output_buffer = ArrayBuffer(
            self.block_size, data)

    def initialize_playback(self, n_channels):
        """Initialize the playback stream for a given number of channels.

        Parameters
        ----------
        n_channels : int
            The number of output channels for which the stream is opened.
        """
        ostream = sd.OutputStream(
            self.sampling_rate,
            self.block_size,
            self.id,
            n_channels,
            self.dtype,
            callback=self.output_callback,
            finished_callback=self._finished_callback)
        self._stream = ostream

    def initialize_record(self, n_channels):
        """Initialize the playback stream for a given number of channels.

        Parameters
        ----------
        n_channels : int
            The number of output channels for which the stream is opened.
        """
        ostream = sd.InputStream(
            self.sampling_rate,
            self.block_size,
            self.id,
            n_channels,
            self.dtype,
            callback=self.input_callback,
            finished_callback=self._finished_callback)
        self._stream = ostream

    def start(self):
        """Start the audio stream
        """
        if not self.stream.closed:
            if not self.stream.active:
                self._stream_finished.clear()
                self.stream.start()
            else:
                print("Stream is already active.", file=sys.stderr)
        else:
            print("Stream is closed. Try re-initializing.", file=sys.stderr)

    def initialize_playback_record(input_channels, output_channels):
        raise NotImplementedError()

    def wait(self):
        """Wait for the audio stream to finish.
        """
        self._stream_finished.wait(timeout=None)

    def abort(self):
        """Stop the audio steam without finishing remaining buffers.
        """
        if self.stream.active is True:
            self.stream.abort()

    def close(self):
        """Close the audio device and release the sound card lock.
        """
        if self.stream is not None:
            self.stream.close()

    def stop(self):
        """Stop the audio stream after finishing the current buffer.
        """
        if self.stream.active is True:
            self.stream.stop()
