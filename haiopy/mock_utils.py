"""
Provides mocks of common haiopy classes, which help to test the hardware
communication.
"""
import numpy as np
from unittest import mock


class _Device(mock.Mock):
    def __init__(
            self,
            id=0,
            sampling_rate=44100,
            block_size=2**5,
            dtype=np.float64):
        super().__init__()
        self._sampling_rate = sampling_rate
        self._dtype = dtype

    @property
    def name(self):
        raise NotImplementedError('Abstract method')

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def dtype(self):
        return self._dtype

    def playback():
        pass

    def record():
        pass

    def playback_record():
        pass

    def initialize_playback():
        pass

    def initialize_record():
        pass

    def initialize_playback_record():
        pass

    def abort():
        pass


class AudioDevice(_Device, mock.Mock):
    def __init__(
            self,
            id=0,
            sampling_rate=44100,
            block_size=2**5,
            dtype=np.float64,
            latency=None,
            extra_settings=None,
            # finished_callback=None,
            clip_off=None,
            dither_off=None,
            never_drop_input=None,
            prime_output_buffers_using_stream_callback=None
            ):
        super().__init__(id, sampling_rate, block_size, dtype)

    @property
    def stream():
        pass

    @staticmethod
    def callback():
        pass

    def playback(data):
        # fill queue, stream.start()
        pass

    def record(n_samples):
        # stream start, read into the queue
        pass

    def playback_record(data):
        # see combination above
        pass

    def initialize_playback(channels):
        # init queue, define callback, init stream
        pass

    def initialize_record(channels):
        pass

    def initialize_playback_record(input_channels, output_channels):
        pass

    def abort():
        # abort
        pass

    def close():
        # remove stream
        pass