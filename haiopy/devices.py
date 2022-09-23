from multiprocessing import Event
import numpy as np
import sys
import sounddevice as sd
from abc import (ABCMeta, abstractmethod, abstractproperty)

from haiopy.generators import (
    ArrayBuffer, InputArrayBuffer, OutputArrayBuffer)


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

    @block_size.setter
    def block_size(self, block_size):
        self._block_size = block_size
        self.output_buffer.block_size = block_size

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

    def start(self):
        """Start the audio stream"""
        if not self.stream.closed:
            if not self.stream.active:
                self._stream_finished.clear()
                self.stream.start()
            else:
                print("Stream is already active.", file=sys.stderr)
        else:
            print("Stream is closed. Try re-initializing.", file=sys.stderr)

    def wait(self):
        """Wait for the audio stream to finish."""
        self._stream_finished.wait(timeout=None)

    def abort(self):
        """Stop the audio steam without finishing remaining buffers."""
        if self.stream.active is True:
            self.stream.abort()

    def close(self):
        """Close the audio device and release the sound card lock."""
        if self.stream is not None:
            self.stream.close()

    def stop(self):
        """Stop the audio stream after finishing the current buffer."""
        if self.stream.active is True:
            self.stream.stop()


class AudioInputDevice(_Device):
    def __init__(
            self,
            id=sd.default.device,
            sampling_rate=44100,
            block_size=512,
            dtype='float32',
            input_buffer=None,
            latency=None,
            extra_settings=None,
            clip_off=None,
            dither_off=None,
            never_drop_input=None,
            prime_output_buffers_using_stream_callback=None
            ):
        super().__init__(
            id=id,
            sampling_rate=sampling_rate,
            block_size=block_size,
            dtype=dtype,
            latency=latency)

        n_channels_input = sd.query_devices(id)['max_input_channels']
        sd.check_input_settings(
            device=id,
            channels=n_channels_input,
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)

    @property
    def n_channels_input(self):
        """The number of input channels supported by the device"""
        return sd.query_devices(self.id)['max_input_channels']

    def _set_block_size(self, block_size):
        self.input_buffer.block_size = block_size

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


class AudioOutputDevice(AudioDevice):

    def __init__(
            self,
            id=sd.default.device,
            sampling_rate=44100,
            block_size=512,
            channels=[1],
            dtype='float32',
            output_buffer=None,
            latency=None,
            extra_settings=None,
            clip_off=None,
            dither_off=None,
            never_drop_input=None,
            prime_output_buffers_using_stream_callback=None):
        super().__init__(
            id=id,
            sampling_rate=sampling_rate,
            block_size=block_size,
            dtype=dtype,
            latency=latency)

        n_channels_output = sd.query_devices(id)['max_output_channels']
        sd.check_output_settings(
            device=id,
            channels=n_channels_output,
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)

        self._output_channels = channels

        if output_buffer is None:
            OutputArrayBuffer(
                self.block_size,
                np.zeros(
                    (self.n_channels_output, self.block_size),
                    dtype=self.dtype))
        if output_buffer.data.shape[0] != self.n_channels_output:
            raise ValueError(
                "The shape of the buffer does not match the channel mapping")
        self.output_buffer = output_buffer
        self.initialize()

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def n_channels_output(self):
        return len(self._output_channels)

    @property
    def max_channels_output(self):
        """The number of output channels supported by the device"""
        return sd.query_devices(self.id)['max_output_channels']

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

    def initialize(self):
        """Initialize the playback stream for a given number of channels."""
        ostream = sd.OutputStream(
            self.sampling_rate,
            self.block_size,
            self.id,
            self.n_channels_output,
            self.dtype,
            callback=self.output_callback,
            finished_callback=self._finished_callback)
        self._stream = ostream

    @property
    def output_buffer(self):
        return self._output_buffer

    @output_buffer.setter
    def output_buffer(self, buffer):
        if buffer.block_size != self.block_size:
            raise ValueError(
                "The buffer's block size does not match. ",
                f"Needs to be {self.block_size}")

        self._output_buffer = buffer
