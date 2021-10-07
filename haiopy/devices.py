from os import stat
import sys
import sounddevice as sd
import queue
import numpy as np
from arrayqueues import ArrayQueue
from abc import (ABCMeta, abstractmethod, abstractproperty)


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
            id,
            sampling_rate=44100,
            block_size=512,
            dtype='float32',
            latency=None,
            extra_settings=None,
            # finished_callback=None,
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

        self.id = id
        self.dtype = dtype
        self._block_size = block_size
        self._sampling_rate = sampling_rate
        self._extra_settings = extra_settings

        self._callback = None
        self._stream = None

        self._record_queue = None
        self.initialize_playback_queue()

    @property
    def n_channels_input(self):
        return sd.query_devices(self.id)['max_input_channels']

    @property
    def n_channels_output(self):
        return sd.query_devices(self.id)['max_output_channels']

    def check_settings(
            self, sampling_rate, dtype, extra_settings,):
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
        return self._name

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        self.check_settings(value, self.dtype, self._extra_settings)

    @property
    def block_size(self):
        return self._block_size

    @property
    def stream(self):
        return self._stream

    def finished_callback(self) -> None:
        self.stream.stop()
        # self.stream.callback = self._callback

    def output_callback(self, outdata, frames, time, status) -> None:
        assert frames == self.block_size
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort('Buffer underflow')
        assert not status

        if self.playback_queue.empty():
            print('Buffer is empty: Are we finished?', file=sys.stderr)
            raise sd.CallbackStop("Buffer empty")
            # self.stream.abort()
        else:
            data = self._playback_queue.get()
            outdata[:] = data.T

    def playback(self, data, start=True):
        if data.ndim > 2:
            raise ValueError(
                "The data cannot can not have more than 2 dimensions.")
        n_channels_data = data.shape[0]

        # queue size in mega bytes
        qsize = data.itemsize * data.size / 1000000
        self.initialize_playback_queue(qsize)
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
        return self._playback_queue

    def initialize_playback_queue(self, qsize=32):
        self._playback_queue = ArrayQueue(qsize)

    def write_queue(self, data):
        if self._playback_queue is None:
            # self.initialize_playback_queue()
            raise ValueError("The Queue need to be initialized first.")

        n_blocks = int(np.floor(data.shape[-1]/self.block_size))

        for idb in range(n_blocks):
            sdx = np.arange(idb*self.block_size, (idb+1)*self.block_size)
            self._playback_queue.put(data[..., sdx])

    def initialize_playback(self, n_channels):
        self.initialize_playback_queue()
        ostream = sd.OutputStream(
            self.sampling_rate,
            self.block_size,
            self.id,
            n_channels,
            self.dtype,
            callback=self.output_callback,
            finished_callback=self.finished_callback
        )

        self._stream = ostream

    def start(self):
        if not self.stream.closed:
            if not self.stream.active:
                self.stream.start()
            else:
                print("Stream is already active.", file=sys.stderr)
        else:
            print("Stream is closed. Try re-initializing.", file=sys.stderr)

    def initialize_record(channels):
        raise NotImplementedError()

    def initialize_playback_record(input_channels, output_channels):
        raise NotImplementedError()

    def abort(self):
        if self.stream.active is True:
            self.stream.abort()

    def close(self):
        if self.stream is not None:
            self.stream.close()

    def stop(self):
        if self.stream.active is True:
            self.stream.stop()
