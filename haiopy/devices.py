
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
        pass

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


