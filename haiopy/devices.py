from multiprocessing import Event
import asyncio
from os import stat
import sys
import sounddevice as sd
import queue
import numpy as np
from arrayqueues import ArrayQueue
from abc import (ABCMeta, abstractmethod, abstractproperty)
from queue import Queue


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

        self._record_queue = None
        self._playback_queue = None
        self.initialize_playback_queue()

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

        if self.playback_queue.empty():
            print('Buffer is empty: Are we finished?', file=sys.stderr)
            raise sd.CallbackStop("Buffer empty")
        else:
            data = self._playback_queue.get()
            outdata[:] = data.T

    def playback(self, data, start=True):
        """Playback an array of audio data.
        This method initializes the playback device for the given audio
        data and starts playback. After playback is finished, the device
        is automatically stopped.

        Parameters
        ----------
        data : array, float32, int8, int16, int32
            Playback data with dimensions (n_channels, n_samples).
        start : bool, optional
            If ``True``, the playback is started right away, if ``False``.
            The default is ``True``

        """
        if data.ndim > 2:
            raise ValueError(
                "The data cannot can not have more than 2 dimensions.")
        n_channels_data = data.shape[0]

        # queue size in mega bytes
        # qsize = data.itemsize * data.size / 1000000
        # self.initialize_playback_queue(qsize)
        self.initialize_playback_queue(0)
        self.initialize_playback(n_channels_data)
        self.write_queue(data)

        if start is True:
            self.start()

    def record(n_samples):
        # stream start, read into the queue
        pass

    def playback_record(data):
        # see combination above
        pass

    @property
    def playback_queue(self):
        """The playback queue, storing audio data in blocks which is read
        by the playback callback.
        """
        return self._playback_queue

    def initialize_playback_queue(self, qsize=32):
        """Initialize an empty playback queue.

        Parameters
        ----------
        qsize : int, optional
            The queue size in mega-bytes, by default 32
        """
        if self._playback_queue is not None:
            self._playback_queue = None
        self._playback_queue = Queue(0)

    def write_queue(self, data):
        """Fill the playback queue with audio data.

        Parameters
        ----------
        data : array, float32, int32, int16, int8
            The audio data as numpy array

        """
        if self._playback_queue is None:
            raise ValueError("The Queue need to be initialized first.")

        n_blocks = int(np.floor(data.shape[-1]/self.block_size))

        for idb in range(n_blocks):
            sdx = np.arange(idb*self.block_size, (idb+1)*self.block_size)
            # if self._playback_queue.check_full():
            if self._playback_queue.full():
                raise MemoryError(
                    "The input queue is full. ",
                    "Try initializing a larger queue.")
            self._playback_queue.put(data[..., sdx])

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
            finished_callback=self._finished_callback
        )

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

    def initialize_record(channels):
        raise NotImplementedError()

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
