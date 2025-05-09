import pytest
from unittest.mock import MagicMock

import numpy as np
import sounddevice as sd


def output_stream_mock(block_size=512, sampling_rate=44100, channels=1):
    # pass
    stream = MagicMock(spec_set=sd.OutputStream)
    stream.samplerate = sampling_rate
    stream.blocksize = block_size
    stream.device = 0
    stream.channels = channels
    stream.dtype = np.float32
    stream.latency = 0.1
    # stream.extra_settings = None
    # stream.clip_off = False
    # stream.dither_off = False
    # stream.never_drop_input = False
    # stream.prime_output_buffers_using_stream_callback = False

    return stream
