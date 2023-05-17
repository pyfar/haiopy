
from multiprocessing import Event
import numpy as np
import sys
import sounddevice as sd
from abc import abstractmethod

from haiopy.buffers import SignalBuffer
import pyfar as pf


def list_devices():
    pass


class _Device(object):
    def __init__(
            self,
            name,
            sampling_rate,
            block_size,
            dtype):
        super().__init__()
        self._name = name
        self._sampling_rate = sampling_rate
        self._block_size = block_size
        self._dtype = dtype

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    def sampling_rate(self):
        return self._sampling_rate

    def block_size(self):
        return self._block_size

    def dtype(self):
        return self.dtype


class AudioDevice(_Device):
    def __init__(
            self,
            identifier=0,
            sampling_rate=44100,
            block_size=512,
            dtype='float32',
            ):

        identifier = sd.query_devices(identifier)['name']

        super().__init__(
            name=sd.query_devices(identifier)['name'],
            sampling_rate=sampling_rate,
            block_size=block_size,
            dtype=dtype
        )
        self._id = identifier
        # self._extra_settings = extra_settings

        self._callback = None
        self._stream = None
        self._input_buffer = None
        self._output_buffer = None

        self._stream_finished = Event()

    @property
    def id(self):
        return self._id

    @abstractmethod
    def check_settings(**kwargs):
        raise NotImplementedError('Needs to be implemented in child class.')

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
        self.check_settings(None, value, None, None)
        self._samping_rate = value

    @property
    def block_size(self):
        """The block size of the audio buffer.
        """
        return self._block_size

    @block_size.setter
    def block_size(self, block_size):
        self._block_size = block_size

    @property
    def dtype(self):
        return self._dtype

    @property
    @abstractmethod
    def stream(self):
        """The sounddevice audio stream.
        """
        return self._stream

    def finished_callback(self) -> None:
        """Custom callback after a audio stream has finished."""
        print("I'm finished.")

    def _finished_callback(self) -> None:
        """Private portaudio callback after a audio stream has finished."""
        self._stream_finished.set()
        self.finished_callback()
        self.stream.stop()

    def start(self):
        """Start the audio stream"""
        if self.stream.closed:
            print("Stream is closed. Try re-initializing.", file=sys.stderr)
            return

        elif not self.stream.active:
            self._stream_finished.clear()
            self.stream.start()
        else:
            print("Stream is already active.", file=sys.stderr)

    def wait(self):
        """Wait for the audio stream to finish."""
        self._stream_finished.wait(timeout=None)

    def abort(self):
        """Stop the audio steam without finishing remaining buffers."""
        if self.stream.active is True:
            self.stream.abort()
            self._stop_buffer()

    def close(self):
        """Close the audio device and release the sound card lock."""
        if self.stream is not None:
            self.stream.close()
            self._stop_buffer()

    def stop(self):
        """Stop the audio stream after finishing the current buffer."""
        if self.stream.active is True:
            self.stream.stop()
            self._stop_buffer()

    @abstractmethod
    def _stop_buffer(self):
        raise NotImplementedError()

    @abstractmethod
    def _reset_buffer(self):
        raise NotImplementedError()


class OutputAudioDevice(AudioDevice):

    def __init__(
            self,
            identifier=sd.default.device['output'],
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

        # First check the settings before continuing
        max_channel = np.max(channels)
        n_channels = len(channels)
        sd.check_output_settings(
            device=identifier,
            channels=np.max([n_channels, max_channel+1]),
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)

        super().__init__(
            identifier=identifier,
            sampling_rate=sampling_rate,
            block_size=block_size,
            dtype=dtype)

        self._output_channels = channels

        if output_buffer is None:
            output_buffer = SignalBuffer(
                self.block_size,
                pf.Signal(np.zeros(
                        (self.n_channels_output, self.block_size),
                        dtype=self.dtype),
                    self.sampling_rate, fft_norm='rms'))
        self.output_buffer = output_buffer
        self.initialize()

    def check_settings(
            self,
            n_channels=None,
            sampling_rate=None,
            dtype=None,
            extra_settings=None):
        """Check if settings are compatible with the physical devices.

        Parameters
        ----------
        n_channels : int
            The number of channels to be used
        sampling_rate : int
            The audio sampling rate
        dtype : np.float32, np.int8, np.int16, np.int32
            The audio buffer data type
        extra_settings : extra settings
            Audio API specific settings.

        Raises
        ------
        PortAudioError
            If the settings are incompatible with the device an exception is
            raised.
        """
        sd.check_output_settings(
            device=self.id,
            channels=n_channels,
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def n_channels_output(self):
        """The total number of output channels.

        Returns
        -------
        int
            The number of output channels
        """
        return len(self._output_channels)

    @property
    def _n_channels_stream(self):
        """The number of output channels required for the stream.

        This includes a number of unused pre-pended channels which need to be
        filled with zeros before writing the portaudio buffer. In case of
        using only the first channel, portaudio plays back a mono signal,
        which will be broadcasted to the first two channels. To avoid this,
        the minimum number of channels opened is always two, the unused second
        channel is filled with zeros.
        """
        return np.max((2, np.max(self._output_channels) + 1))

    @property
    def max_channels_output(self):
        """The number of output channels supported by the device"""
        return sd.query_devices(self.id, 'output')['max_output_channels']

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
            # Write a block to an array with all required output channels
            # including zeros for unused channels. Required as sounddevice does
            # not support routing matrices
            self._stream_block_out[self.output_channels] = next(
                self.output_buffer)
            outdata[:] = self._stream_block_out.T
        except StopIteration as e:
            raise sd.CallbackStop("Buffer empty") from e

    def initialize(self):
        """Initialize the playback stream for a given number of channels."""
        ostream = sd.OutputStream(
            self.sampling_rate,
            self.block_size,
            self.id,
            self._n_channels_stream,
            self.dtype,
            callback=self.output_callback,
            finished_callback=self._finished_callback)
        self._stream = ostream
        # Init array buffering a block of all required output channels
        # including zeros for unused channels. Required as sounddevice does
        # not support routing matrices
        self._stream_block_out = np.zeros(
            (self._n_channels_stream, self.block_size), dtype=self.dtype)

    def initialize_buffer(self):
        self.output_buffer._start()
        self.output_buffer._is_active.wait()

    @property
    def output_buffer(self):
        return self._output_buffer

    @output_buffer.setter
    def output_buffer(self, buffer):
        if buffer.block_size != self.block_size:
            raise ValueError(
                "The buffer's block size does not match. ",
                f"Needs to be {self.block_size}")

        if buffer.n_channels != self.n_channels_output:
            raise ValueError(
                "The buffer's channel number does not match the channel "
                f"mapping. Currently used channels are {self.output_channels}")

        self._output_buffer = buffer

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, value):
        if self.stream.active is True or self.output_buffer.is_active is True:
            raise ValueError(
                "The device is currently in use and needs to be closed first")
        self.output_buffer.block_size = value
        super(OutputAudioDevice, self.__class__).block_size.fset(self, value)

    def _stop_buffer(self):
        self._output_buffer._stop()

    def start(self):
        self.output_buffer._start()
        self.output_buffer._is_active.wait()
        super().start()

    def wait(self):
        super().wait()
        self.output_buffer._is_finished.wait()
